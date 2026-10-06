"""Regression checks for actual validity and endpoint coordinate contracts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
from correct_support import composed_support, sha

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'p22_dense_motion'))
from extract_dense import grid,residual_field
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(HERE.parent/'p22_dense_analysis'))
from temporal_analysis import pull_previous,project


def main():
    valid=np.zeros((1,2,2,2),bool)
    valid[0,0]=[[True,False],[True,False]]
    valid[0,1]=[[False,True],[True,False]]
    history,current=composed_support(valid)
    assert np.array_equal(history[0],[[False,False],[True,False]])
    assert np.array_equal(current,valid[:,1])
    xy=grid((256,320))
    backward=np.broadcast_to(np.array([-2.,1.],np.float32),xy.shape).copy()
    forward=-backward.copy()
    current_rows=[{'bbox':[100,80,130,110]}]
    previous_rows=[{'bbox':[98,81,128,111]}]
    backward[85:105,105:125]+=[4.,-3.]
    # Actual foreground backward endpoint is (x+2,y-2). Its corresponding
    # previous-frame rectangle therefore needs forward flow (-2,+2).
    forward[83:103,107:127]=[-2.,2.]
    residual,visible,_,_=residual_field(backward,forward,current_rows,previous_rows)
    assert visible[87:103,107:123].all()
    assert np.allclose(residual[87:103,107:123],[4.,-3.],atol=1e-3)
    inconsistent=np.broadcast_to(np.array([2.,-1.],np.float32),xy.shape).copy()
    _,rejected,_,_=residual_field(backward,inconsistent,current_rows,previous_rows)
    assert not rejected[87:103,107:123].any()
    # Independent non-affine endpoint construction tests the temporal pullback.
    points=xy[90:106,110:126].astype(float)
    Hc=np.array([[.99,.03,4.],[-.02,1.01,-3.],[.00003,-.00002,1.]])
    Hp=np.array([[1.01,-.02,-2.],[.04,.98,1.],[-.00004,.00001,1.]])
    r_current=np.broadcast_to([2.,-1.],points.shape)
    previous_points=project(Hc,points)[0]+r_current
    expected=np.broadcast_to([1.7,-.8],points.shape)
    r_previous=project(Hp,previous_points+expected)[0]-project(Hp,previous_points)[0]
    pulled,_,supported,error=pull_previous(r_previous,r_current,points,Hc,Hp)
    assert supported.all() and np.max(error)<1e-8
    assert np.max(np.abs(pulled-expected))<1e-8
    result={'status':'PASS_CORRECTED_CONTRACT','composed_path_mask':True,
            'physically_consistent_forward_backward_foreground_visible':True,
            'original_inconsistent_foreground_correctly_rejected':True,'projective_temporal_endpoint_pullback':True,
            'real_image_invariance_validated':False,'GPU_model_inference':False,
            'sources':{str(p):sha(p) for p in [Path(__file__),HERE/'correct_support.py',
                       HERE.parent/'p22_dense_motion/extract_dense.py',HERE.parent/'p22_dense_analysis/temporal_analysis.py']}}
    (HERE/'TESTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
