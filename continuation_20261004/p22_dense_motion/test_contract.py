import json
from pathlib import Path
import numpy as np
from extract_dense import sample, grid, residual_field, project


def main():
    xy = grid((256, 320))
    assert np.allclose(sample(xy, xy), xy)
    backward = np.broadcast_to(np.array([-2., 1.], np.float32), xy.shape).copy()
    forward = -backward
    current = [{'bbox': [100,80,130,110]}]
    previous = [{'bbox': [98,81,128,111]}]
    residual, valid, diagnostics, matrix = residual_field(backward, forward, current, previous)
    assert diagnostics['status'] == 'BACKGROUND_MODEL_ESTIMATED'
    assert np.max(np.abs(residual[valid])) < 1e-3
    altered = backward.copy()
    altered[85:105,105:125] += [4.,-3.]
    target, _, _, _ = residual_field(altered, forward, current, previous)
    assert np.allclose(np.median(target[85:105,105:125],axis=(0,1)),[4.,-3.],atol=1e-3)
    transform = np.array([[.98,-.1,12],[.1,.98,-8],[0,0,1.]])
    points = xy[90:100,110:120]
    background, _ = project(matrix, points)
    endpoint = background + np.array([4.,-3.])
    changed_endpoint, _ = project(transform, endpoint)
    changed_background, _ = project(transform, background)
    recovered = (changed_endpoint-changed_background) @ np.linalg.inv(transform[:2,:2]).T
    assert np.allclose(recovered, [4.,-3.], atol=1e-4)
    out = {'status':'PASS','full_field_remap':True,'pure_camera_zero_residual':True,
           'foreground_displacement_recovered':True,'affine_endpoint_pullback':True,
           'model_forward_run':False,'real_image_intervention_validated':False}
    Path(__file__).with_name('TESTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
