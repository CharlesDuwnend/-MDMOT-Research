"""Independent CPU numerical contracts; no data, checkpoints or CUDA calls."""

import math
import unittest

import torch

from temporal_module import TemporalAdapter, validate_temporal_group, warp_past_features


torch.set_num_threads(1)


def level_shape(pad, stride):
    return tuple(int(math.ceil(v / float(stride))) for v in pad)


def ramp(count, height, width, channels=1, dtype=torch.float64):
    # Analytic sample value is 100 * y + x, without using grid_sample.
    y = torch.arange(height, dtype=dtype)[:, None]
    x = torch.arange(width, dtype=dtype)[None, :]
    return (100 * y + x)[None, None].repeat(count, channels, 1, 1)


class WarpContracts(unittest.TestCase):
    def test_zero_flow_exact_identity_odd_non_square_all_levels(self):
        torch.manual_seed(1)
        original, resized, padded = (173, 271), (75, 113), (96, 128)
        strides = (8, 16, 32, 64, 128)
        past = [torch.randn(3, 4, *level_shape(padded, s)) for s in strides]
        warped, masks = warp_past_features(past, torch.zeros(3, 2, 64, 96),
            original_hw=original, detector_img_hw=resized, pad_hw=padded)
        for source, result, mask, stride in zip(past, warped, masks, strides):
            h, w = source.shape[-2:]
            expected = torch.tensor([[i * stride < resized[0] and
                                      j * stride < resized[1]
                                      for j in range(w)] for i in range(h)])
            self.assertTrue(torch.equal(mask[0, 0], expected))
            expanded = mask.expand_as(source)
            self.assertTrue(torch.equal(source[expanded], result[expanded]))
            self.assertTrue(torch.equal(result[~expanded], torch.zeros_like(result[~expanded])))

    def test_positive_negative_displacement_and_separate_xy_units(self):
        stride, resized, padded, flow_hw = 8, (64, 96), (64, 96), (24, 32)
        refs = ramp(3, 8, 12)
        flow = torch.zeros(3, 2, *flow_hw, dtype=torch.float64)
        # Three different backward directions in feature-pixel units.
        shifts = [(1., 0.), (-1., 0.), (0., 1.)]
        for k, (dx, dy) in enumerate(shifts):
            flow[k, 0].fill_(dx * stride * flow_hw[1] / resized[1])
            flow[k, 1].fill_(dy * stride * flow_hw[0] / resized[0])
        out, valid = warp_past_features([refs], flow, original_hw=(137, 259),
            detector_img_hw=resized, pad_hw=padded, strides=(stride,))
        for k, (dx, dy) in enumerate(shifts):
            expected = refs[k, 0] + dx + 100 * dy
            mask = valid[0][k, 0]
            self.assertGreater(mask.sum().item(), 0)
            self.assertLess((out[0][k, 0][mask] - expected[mask]).abs().max().item(), 1e-10)
        self.assertFalse(valid[0][0, 0, 2, -1])
        self.assertFalse(valid[0][1, 0, 2, 0])
        self.assertFalse(valid[0][2, 0, -1, 2])

    def test_spatial_flow_pixelcentre_transform_against_analytic_ramp(self):
        stride, resized, padded = 8, (80, 120), (80, 120)
        original, flow_hw = (197, 313), (32, 48)
        fy = torch.arange(flow_hw[0], dtype=torch.float64)[:, None]
        fx = torch.arange(flow_hw[1], dtype=torch.float64)[None, :]
        flow = torch.zeros(1, 2, *flow_hw, dtype=torch.float64)
        flow[0, 0] = .02 * (fx + .5) + .1
        flow[0, 1] = .01 * (fy + .5) + .05
        refs = ramp(1, 10, 15)
        out, valid = warp_past_features([refs], flow, original_hw=original,
            detector_img_hw=resized, pad_hw=padded, strides=(stride,))
        errors = []
        for i in range(1, 8):
            for j in range(1, 13):
                native_x = (j * stride + .5) * original[1] / resized[1]
                native_y = (i * stride + .5) * original[0] / resized[0]
                flow_edge_x = native_x * flow_hw[1] / original[1]
                flow_edge_y = native_y * flow_hw[0] / original[0]
                dx = (.02 * flow_edge_x + .1) * resized[1] / flow_hw[1] / stride
                dy = (.01 * flow_edge_y + .05) * resized[0] / flow_hw[0] / stride
                self.assertTrue(valid[0][0, 0, i, j])
                errors.append(abs(out[0][0, 0, i, j].item() - (100 * (i + dy) + j + dx)))
        self.assertLess(max(errors), 1e-10)

    def test_no_padding_feature_contribution_at_fractional_endpoint(self):
        refs = ramp(1, 4, 5)
        flow = torch.zeros(1, 2, 16, 16, dtype=torch.float64)
        kwargs = dict(original_hw=(100, 100), detector_img_hw=(23, 23),
                      pad_hw=(32, 40), strides=(8,))
        zero, zero_valid = warp_past_features([refs], flow, **kwargs)
        self.assertTrue(zero_valid[0][0, 0, 1, 2])  # last valid integer node
        self.assertFalse(zero_valid[0][0, 0, 1, 3])
        flow[:, 0] = 2 * 16 / 23  # endpoint remains inside image, needs pad node
        _, fractional = warp_past_features([refs], flow, **kwargs)
        self.assertFalse(fractional[0][0, 0, 1, 2])
        self.assertTrue(fractional[0][0, 0, 1, 1])
        flow[:, 0] = -2 * 16 / 23
        _, negative = warp_past_features([refs], flow, **kwargs)
        self.assertFalse(negative[0][0, 0, 1, 0])

    def test_flow_border_sampling_does_not_attenuate_constant_motion(self):
        refs = ramp(1, 8, 8)
        flow = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
        flow[:, 0] = .25  # equals 8 detector pixels, including near image edge
        out, valid = warp_past_features([refs], flow, original_hw=(128, 128),
            detector_img_hw=(64, 64), pad_hw=(64, 64), strides=(8,))
        self.assertTrue(valid[0][0, 0, 0, 0])
        self.assertAlmostEqual(out[0][0, 0, 0, 0].item(), 1., places=10)

    def test_coarse_ceil_lattices_and_wrong_shape_rejection(self):
        resized, padded = (751, 1333), (768, 1344)
        strides = (64, 128)
        refs = [torch.ones(3, 1, *level_shape(padded, s)) for s in strides]
        out, masks = warp_past_features(refs, torch.zeros(3, 2, 360, 640),
            original_hw=(1080, 1920), detector_img_hw=resized,
            pad_hw=padded, strides=strides)
        self.assertEqual(out[1].shape[-2:], (6, 11))
        # Offset zero means the ceil node at x=1280 remains valid. A wrong
        # half-stride origin would silently reject this node as x=1344.
        self.assertTrue(masks[1][0, 0, 0, -1])
        self.assertTrue(masks[1][0, 0, -1, -2])
        with self.assertRaises(ValueError):
            warp_past_features([torch.ones(3, 1, 5, 10)], torch.zeros(3, 2, 5, 8),
                original_hw=(1080, 1920), detector_img_hw=resized,
                pad_hw=padded, strides=(128,))

    def test_nonfinite_flow_invalid_and_finite_warp(self):
        refs = ramp(3, 4, 4)
        flow = torch.zeros(3, 2, 16, 16, dtype=torch.float64)
        flow[0] = float("nan")
        flow[1] = float("inf")
        flow[2] = 1e5
        out, valid = warp_past_features([refs], flow, original_hw=(50, 50),
            detector_img_hw=(32, 32), pad_hw=(32, 32), strides=(8,))
        self.assertFalse(valid[0].any())
        self.assertTrue(torch.isfinite(out[0]).all())
        self.assertEqual(out[0].abs().sum().item(), 0)

    def test_float32_exact_boundary_shift_on_realistic_resize(self):
        # This catches a float32 failure that double-precision shift tests
        # miss: grid_sample's constant-flow result can vary by a few ulps,
        # incorrectly rejecting some exact x=0 endpoints after a -1 shift.
        refs = ramp(1, 96, 168, dtype=torch.float32)
        flow = torch.zeros(1, 2, 360, 640)
        flow[:, 0] = -8 * 640 / 1333
        out, valid = warp_past_features([refs], flow,
            original_hw=(1080, 1920), detector_img_hw=(751, 1333),
            pad_hw=(768, 1344), strides=(8,))
        self.assertTrue(valid[0][0, 0, :94, 1].all())
        error = (out[0][0, 0, :93, 2:165] - refs[0, 0, :93, 1:164]).abs()
        self.assertLess(error.max().item(), 2e-3)


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.current = [torch.randn(1, 4, 4, 6, requires_grad=True)]
        self.past = [torch.randn(3, 4, 4, 6)]
        self.flow = torch.zeros(3, 2, 16, 24)
        self.kwargs = dict(original_hw=(97, 145), detector_img_hw=(32, 48),
                           pad_hw=(32, 48))

    def test_single_and_no_history_exact_identity_at_initialization(self):
        for mode in ("single", "fgfa", "uniform", "unaligned"):
            model = TemporalAdapter(4, mode=mode, strides=(8,))
            output = model(self.current)
            self.assertTrue(torch.equal(output[0], self.current[0]))
            self.assertTrue(output[0].requires_grad)

    def test_all_invalid_past_exact_current_fallback_and_no_nan(self):
        model = TemporalAdapter(4, strides=(8,))
        output = model(self.current, self.past, self.flow + 1e5,
                       **self.kwargs, return_aux=True)
        self.assertTrue(torch.equal(output["features"][0], self.current[0]))
        self.assertFalse(output["valid"][0].any())
        self.assertTrue(torch.equal(output["weights"][0][0],
                                    torch.ones_like(output["weights"][0][0])))
        self.assertEqual(output["weights"][0][1:].sum().item(), 0)

    def test_zero_embedding_safe_and_uniform_on_available_frames(self):
        model = TemporalAdapter(4, strides=(8,))
        for parameter in model.embedding.parameters():
            parameter.data.zero_()
        output = model(self.current, self.past, self.flow,
                       **self.kwargs, return_aux=True)
        expected = torch.cat((self.current[0], self.past[0])).mean(dim=0, keepdim=True)
        self.assertTrue(torch.isfinite(output["features"][0]).all())
        self.assertTrue(torch.allclose(output["features"][0], expected, atol=1e-7))
        self.assertTrue(torch.equal(output["weights"][0],
                                    torch.full_like(output["weights"][0], .25)))

    def test_nonidentical_refs_backprop_to_value_embedding_and_current(self):
        model = TemporalAdapter(4, strides=(8,))
        saved = [x.clone() for x in self.past]
        output = model(self.current, self.past, self.flow, **self.kwargs)
        target = torch.randn_like(output[0])
        ((output[0] - target) ** 2).mean().backward()
        for module in (model.value_adapter, model.embedding):
            total = sum(p.grad.abs().sum().item() for p in module.parameters()
                        if p.grad is not None)
            self.assertGreater(total, 0)
        self.assertGreater(self.current[0].grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(saved[0], self.past[0]))

    def test_single_uses_same_value_adapter_and_discloses_unused_embedding(self):
        single = TemporalAdapter(4, mode="single", strides=(8,))
        fgfa = TemporalAdapter(4, mode="fgfa", strides=(8,))
        fgfa.load_state_dict(single.state_dict())
        single(self.current)[0].square().mean().backward()
        self.assertGreater(single.value_adapter.residual.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in single.embedding.parameters()))
        self.assertEqual(single.parameter_counts()["value_adapter"],
                         fgfa.parameter_counts()["value_adapter"])
        self.assertGreater(single.parameter_counts()["unused_embedding"], 0)
        self.assertEqual(fgfa.parameter_counts()["unused_embedding"], 0)

    def test_uniform_and_unaligned_controls_have_expected_semantics(self):
        uniform = TemporalAdapter(4, mode="uniform", strides=(8,))
        result = uniform(self.current, self.past, self.flow, **self.kwargs)
        expected = torch.cat((self.current[0], self.past[0])).mean(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(result[0], expected, atol=1e-7))
        unaligned = TemporalAdapter(4, mode="unaligned", strides=(8,))
        zero = unaligned(self.current, self.past, None, **self.kwargs)
        changed = unaligned(self.current, self.past, self.flow + 999, **self.kwargs)
        self.assertTrue(torch.equal(zero[0], changed[0]))

    def test_empty_past_supported(self):
        model = TemporalAdapter(4, strides=(8,))
        output = model(self.current, [self.past[0][:0]], self.flow[:0],
                       **self.kwargs, return_aux=True)
        self.assertTrue(torch.equal(output["features"][0], self.current[0]))
        self.assertEqual(output["weights"][0].shape[0], 1)


class TemporalGroupContracts(unittest.TestCase):
    def test_strict_past_same_sequence_and_declared_order(self):
        records = [{"pair": "23", "view": 1, "frame": f} for f in (8, 5, 1)]
        self.assertEqual(validate_temporal_group(pair="23", view=1,
                         current_frame=9, reference_records=records), (1, 4, 8))
        mutations = [
            [{"pair": "23", "view": 1, "frame": 9}] + records[1:],
            [{"pair": "23", "view": 1, "frame": 10}] + records[1:],
            [{"pair": "23", "view": 2, "frame": 8}] + records[1:],
            [{"pair": "24", "view": 1, "frame": 8}] + records[1:],
            list(reversed(records)), records[:2] + [records[1]], records[:2],
        ]
        for bad in mutations:
            with self.assertRaises(ValueError):
                validate_temporal_group(pair="23", view=1, current_frame=9,
                                        reference_records=bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
