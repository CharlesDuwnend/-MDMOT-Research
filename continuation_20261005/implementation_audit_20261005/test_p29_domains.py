"""Analytical contracts through the installed MSDA; no images or labels."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'p28_causal_fgfa'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def center_sampler(mod):
    adapter = mod.ResidualDeformableAdapter(channels=4, strides=(8,))
    with torch.no_grad():
        adapter.msda.value_proj.weight.copy_(torch.eye(4))
        adapter.msda.value_proj.bias.zero_()
        adapter.msda.output_proj.weight.copy_(torch.eye(4))
        adapter.msda.output_proj.bias.zero_()
        adapter.msda.attention_weights.weight.zero_()
        bias = adapter.msda.attention_weights.bias.view(4, 3, 9)
        bias.fill_(-1000.)
        bias[:, :, 4] = 0.
    return adapter.eval()


def output(mod, dx, detector_hw=(32, 32)):
    adapter = center_sampler(mod)
    current = torch.zeros(1, 4, 4, 4)
    # Source value is x+1, so erroneous masking is observable even at x=0.
    raw = (torch.arange(4, dtype=torch.float32) + 1).view(1, 1, 1, 4).expand(3, 4, 4, 4).clone()
    flow = torch.zeros(3, 2, 32, 32)
    flow[:, 0] = dx
    with torch.no_grad():
        return adapter([current], [raw], flow, original_hw=(32, 32),
                       detector_img_hw=detector_hw, pad_hw=(32, 32))[0]


def main():
    repaired = load(BASE / 'p29_repair_20261005/residual_deformable.py', 'repaired_p29')
    stride_only = load(Path(__file__).parent / 'attempts/p29_stride_only/residual_deformable.py', 'stride_only_p29')
    cases = []
    for dx, query_x, expected in ((8., 2, 4.), (-8., 1, 1.)):
        corrected = float(output(repaired, dx)[0, 0, 1, query_x])
        defective = float(output(stride_only, dx)[0, 0, 1, query_x])
        assert abs(corrected - expected) < 1e-5, (dx, corrected, expected)
        assert abs(defective - expected) > .9, (dx, defective, expected)
        cases.append(dict(flow_dx=dx, query_index=[1, query_x], expected=expected,
                          repaired=corrected, stride_only=defective))
    padded = output(repaired, 0., detector_hw=(24, 24))
    assert float(padded[0, 0, 1, 2]) == 3.
    assert float(padded[0, 0, 1, 3]) == 0.
    # Spatial flow ramp on an anisotropic detector resize. Independent
    # formula: raster coordinate ((j*stride+.5)*fw/dw-.5), border clamped,
    # displacement raster*dw/fw/stride.
    flow = torch.zeros(3, 2, 24, 40, dtype=torch.float64)
    flow[:, 0] = torch.arange(40, dtype=torch.float64)[None]
    flow[:, 1] = torch.arange(24, dtype=torch.float64)[:, None]
    disp, _ = repaired._flow_sample(flow, 4, 6, 8, (120, 200), (32, 48), 'cpu')
    for y in range(4):
        for x in range(6):
            expected_x = max(0., min(39., (x*8+.5)*40/48-.5))*48/40/8
            expected_y = max(0., min(23., (y*8+.5)*24/32-.5))*32/24/8
            assert abs(float(disp[0, y, x, 0])-expected_x) < 1e-10
            assert abs(float(disp[0, y, x, 1])-expected_y) < 1e-10
    result = dict(status='PASS_P29_TWO_DOMAIN_REPAIR_CONTRACTS', center_msda_cases=cases,
                  raw_padding_contract='PASS', anisotropic_resize_ramp='PASS_24_NODES',
                  labels_read=False, gpu_used=False, official_val_or_test_read=False)
    (Path(__file__).parent/'P29_TWO_DOMAIN_CONTRACT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
