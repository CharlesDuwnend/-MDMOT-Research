#!/usr/bin/env python3
import hashlib,json,sys,inspect
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from co_movement import CrossViewTemporalDifferentialField

def check(name,ok,evidence):
 if not ok: raise AssertionError(f'{name}: {evidence}')
 return {'check':name,'status':'PASS','evidence':evidence}

def main():
 torch.manual_seed(991)
 m=CrossViewTemporalDifferentialField(8,hidden=16,output_dim=8)
 b,s,d=4,5,8
 current_t=torch.randn(b,d,requires_grad=True); current_s=torch.randn(b,d,requires_grad=True)
 st=torch.randn(b,s,d,requires_grad=True); ss=torch.randn(b,s,d,requires_grad=True)
 offsets=torch.tensor([[0.,1.,2.,3.,4.],[0.,2.,4.,6.,8.],[0.,1.,1.,3.,5.],[0.,1.,2.,3.,4.]])
 valid=torch.tensor([[1,1,1,1,1],[1,1,0,1,1],[1,0,0,0,0],[0,0,0,0,0]],dtype=torch.bool)
 out,stats=m(current_t,current_s,st,ss,offsets,valid)
 checks=[]
 checks.append(check('zero_initialized_residual_is_base_only',float(stats['residual_norm'].max())==0.0,{'residual_norm':stats['residual_norm'].tolist(),'scale':float(m.residual_scale)}))
 checks.append(check('explicit_missing_mask_has_no_nan_and_no_fake_pairs',torch.isfinite(out).all().item() and stats['valid_pair_count'].tolist()==[4,2,0,0],{'valid_pair_count':stats['valid_pair_count'].tolist(),'support_valid_count':stats['support_valid_count'].tolist()}))
 # One valid support frame is legal but has no differential pair; it must not
 # manufacture a residual token.
 one=torch.ones(1,d); zero=torch.zeros(1,1,d); one_off=torch.zeros(1,1); one_mask=torch.ones(1,1,dtype=torch.bool)
 one_out,one_stats=m(one,one,zero,zero,one_off,one_mask)
 checks.append(check('S1_no_derivative_is_defined',torch.isfinite(one_out).all().item() and one_stats['valid_pair_count'].item()==0,{'valid_pair_count':one_stats['valid_pair_count'].item()}))
 # Chronology is part of the contract.
 try: m(current_t[:1],current_s[:1],st[:1],ss[:1],torch.tensor([[0.,2.,1.,3.,4.]]),valid[:1]); order_rejected=False
 except ValueError: order_rejected=True
 checks.append(check('support_order_is_not_silently_permuted',order_rejected,{}))
 # Random-order control should change a nonzero differential field, while the
 # support validity and offsets remain explicit.
 with torch.no_grad(): m.residual_scale.fill_(0.4)
 ordered,_=m(current_t[:1],current_s[:1],st[:1],ss[:1],offsets[:1],valid[:1])
 perm=torch.tensor([0,2,1,3,4]); shuffled,_=m(current_t[:1],current_s[:1],st[:1,perm],ss[:1,perm],offsets[:1],valid[:1])
 checks.append(check('temporal_order_changes_differential_field',float((ordered-shuffled).abs().max())>1e-7,{'max_abs_delta':float((ordered-shuffled).abs().max())}))
 dup_offsets=torch.tensor([[0.,1.,1.,3.,4.]])
 _,dup_stats=m(current_t[:1],current_s[:1],st[:1],ss[:1],dup_offsets,valid[:1])
 checks.append(check('repeated_timestamp_is_not_differential_evidence',dup_stats['valid_pair_count'].item()==3,{'valid_pairs':dup_stats['valid_pair_count'].item()}))
 bad_t=st.detach().clone();bad_s=ss.detach().clone()
 bad_t[~valid]=float('nan');bad_s[~valid]=float('nan')
 clean_output,_=m(current_t,current_s,st,ss,offsets,valid)
 masked_output,_=m(current_t,current_s,bad_t,bad_s,offsets,valid)
 checks.append(check('masked_missing_values_cannot_contaminate_output',torch.isfinite(masked_output).all().item() and torch.allclose(clean_output,masked_output,atol=1e-7),{'max_error':float((clean_output-masked_output).abs().max())}))
 names=list(inspect.signature(m.forward).parameters)
 checks.append(check('model_api_has_no_gt_or_tracker_id',not any('id'==n or 'gt' in n or 'track' in n for n in names),{'input_names':names}))
 m.zero_grad(set_to_none=True); loss=out.square().sum()+stats['field_norm'].sum(); loss.backward()
 finite_grad=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in m.parameters())
 checks.append(check('backward_gradients_finite',finite_grad,{'parameter_gradients':{n:(p.grad is None or float(p.grad.abs().sum())>=0) for n,p in m.named_parameters()}}))
 payload={'status':'PASS_CVTDCF_CPU_MECHANISM_CONTRACT','rounds':[{'round':'R1_INPUT_AND_MASK','checks':checks[:3]},{'round':'R2_TEMPORAL_ORDER','checks':checks[3:6]},{'round':'R3_BACKWARD_AND_INPUT_BOUNDARY','checks':checks[6:]}],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'official_val_test_access':False,'effectiveness_claim':False,'next':'complete collision and signal audit before real-feature learning; no official evaluation'}
 (ROOT/'CPU_CONTRACT.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
