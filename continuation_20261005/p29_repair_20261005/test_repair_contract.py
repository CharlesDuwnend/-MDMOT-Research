import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_deformable import _flow_sample

def main():
    # A spatially varying raster makes the coordinate domain observable.
    flow = torch.zeros(3, 2, 32, 32)
    flow[:, 0] = torch.arange(32, dtype=torch.float32)[None, :]
    disp, mask = _flow_sample(flow, 4, 4, 8, (32, 32), (32, 32), 'cpu')
    got = disp[0, 0, :, 0]
    expected = torch.tensor([0., 1., 2., 3.])
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    assert bool(mask.all())
    print({'status':'PASS_P29_REPAIR_FLOW_REFERENCE_CONTRACT',
           'expected_feature_displacements': expected.tolist(),
           'actual_feature_displacements': got.tolist(),
           'official_metric_authorized':False})
if __name__ == '__main__': main()
