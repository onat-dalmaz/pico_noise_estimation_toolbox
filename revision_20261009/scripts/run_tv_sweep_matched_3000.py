"""Matched 3000-replica/3000-probe TV sweep; preserves the reviewed fixed recon.

Uses the existing manuscript reconstruction and validated functional batching.
PMR is the unbiased variance across exactly N independently perturbed reconstructions.
Checkpointed batch-Welford statistics are accumulated in float64. No runtime claim.
"""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
import torch
from torch.func import vmap,jvp
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_lambda_tv_sweep as sweep
import nonlinear_cs_noise_sweep as ncs
from cs_batched_timing import install_functional_gridding,draw_noise,ORIGINAL_FISTA

LAMS=[3e-7,9e-7,3e-6,9e-6,3e-5,9e-5]
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--N',type=int,default=3000);p.add_argument('--batch-size',type=int,default=100);args=p.parse_args()
assert 0<=args.index<6 and args.N>=2
root=Path(__file__).resolve().parents[1]
out=root/'experiments/fastmri_knee/results/lambda_tv_sweep_matched3000_20260929'/f'lambda_{args.index}'
out.mkdir(parents=True,exist_ok=True)
start=time.time();validation={};progress={'lambda_tv':LAMS[args.index],'N_requested':args.N,'batch_size':args.batch_size,'pmr_N':0,'pico_N':0,'complete':False}

def status():
 progress['elapsed_seconds']=time.time()-start
 temp=out/'status.tmp';temp.write_text(json.dumps(progress,indent=2));temp.replace(out/'status.json')
 print(json.dumps(progress),flush=True)

def save_state(path,state):
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def pmr(ksp,A,prox,K,eigen,sigma,n_replicas):
 assert n_replicas==args.N
 # Compare the functional reconstruction and JVP with the original library.
 legacy=ncs.fista_reconstruct(ksp,A,prox,K,eigen)
 torch.manual_seed(8101)
 old_pico=ncs.jvp_hutch_variance(ksp,A,prox,K,eigen,num_samples=8,batch_size=8)
 install_functional_gridding()
 new=ncs.fista_reconstruct(ksp,A,prox,K,eigen)
 torch.manual_seed(8101)
 new_pico=ncs.jvp_hutch_variance(ksp,A,prox,K,eigen,num_samples=8,batch_size=8)
 validation['reconstruction_relative_difference']=float(torch.linalg.norm(new-legacy)/torch.linalg.norm(legacy))
 validation['same_probe_pico_relative_difference']=float(torch.linalg.norm(new_pico-old_pico)/torch.linalg.norm(old_pico))
 noise=draw_noise(ksp,8,sigma,8102)
 recon=vmap(lambda noise:ncs.fista_reconstruct(ksp+noise,A,prox,K,eigen))
 check=recon(noise).cpu().to(torch.complex128);mean=check.mean(0)
 vb=((check-mean).abs()**2).sum(0)/7
 seq=torch.stack([ORIGINAL_FISTA(lambda x:A.normal(x)/eigen,A.adjoint(ksp+q)/(eigen**0.5),prox,num_iters=K,ptol_exit=-1.0,verbose=False).cpu().to(torch.complex128) for q in noise]);vs=((seq-seq.mean(0)).abs()**2).sum(0)/7
 validation['same_noise_pmr_relative_difference']=float(torch.linalg.norm(vb-vs)/torch.linalg.norm(vs))
 assert max(validation.values())<1e-3,validation
 validation['threshold']=1e-3;validation['gpu']=torch.cuda.get_device_name();validation['max_eigen']=eigen
 (out/'batching_validation.json').write_text(json.dumps(validation,indent=2))
 del check,noise,legacy,new,old_pico,new_pico,vb,vs
 gen=torch.Generator(device=ksp.device).manual_seed(20260929)
 path=out/'pmr_checkpoint.pt';n=0;mean=torch.zeros(A.im_size,dtype=torch.complex128);m2=torch.zeros(A.im_size,dtype=torch.float64)
 if path.exists():
  ck=torch.load(path,map_location='cpu');n=ck['n'];mean=ck['mean'];m2=ck['m2'];gen.set_state(ck['rng'])
 progress['pmr_N']=n;status()
 while n<n_replicas:
  b=min(args.batch_size,n_replicas-n);shape=(b,)+tuple(ksp.shape)
  noise=(torch.randn(shape,device=ksp.device,generator=gen)+1j*torch.randn(shape,device=ksp.device,generator=gen))*sigma
  rr=recon(noise.to(ksp.dtype)).cpu().to(torch.complex128);bm=rr.mean(0);bm2=((rr-bm).abs()**2).sum(0);delta=bm-mean
  m2+=bm2+delta.abs()**2*(n*b/(n+b));mean+=delta*(b/(n+b));n+=b
  save_state(path,{'n':n,'mean':mean.cpu(),'m2':m2.cpu(),'rng':gen.get_state()})
  progress['pmr_N']=n;status()
 return m2/(n-1)/(sigma**2)

def pico(ksp,A,prox,K,eigen,num_samples,batch_size):
 assert num_samples==args.N
 gen=torch.Generator(device=ksp.device).manual_seed(20260930)
 path=out/'pico_checkpoint.pt';n=0;total=torch.zeros(A.im_size,device=ksp.device,dtype=torch.float64)
 if path.exists():
  ck=torch.load(path,map_location='cpu');n=ck['n'];total=ck['total'].to(ksp.device);gen.set_state(ck['rng'])
 def sample(q):
  _,u=jvp(lambda y:ncs.fista_reconstruct(y,A,prox,K,eigen),(ksp,),(q,))
  return (u.real.double()**2+u.imag.double()**2)
 batched=vmap(sample)
 progress['pico_N']=n;status()
 while n<num_samples:
  b=min(args.batch_size,num_samples-n);shape=(b,)+tuple(ksp.shape)
  re=torch.randint(0,2,shape,device=ksp.device,generator=gen).float()*2-1
  im=torch.randint(0,2,shape,device=ksp.device,generator=gen).float()*2-1
  total+=batched((re+1j*im).to(ksp.dtype)).sum(0);n+=b
  save_state(path,{'n':n,'total':total.cpu(),'rng':gen.get_state()})
  progress['pico_N']=n;status()
 return total/n

ncs.FISTA_PTOL_EXIT=None;sweep.TV_PROX='complex';sweep.TV_INNER=50;sweep.TV_TOL=0.0
sweep.pmr_variance_map=pmr;sweep.jvp_hutch_variance=pico
cfg=argparse.Namespace(seed=20260830,slice_idx=20,accel=2,lamdas=[LAMS[args.index]],n_pmr=args.N,n_pico=args.N,sigma=1e-7,num_iters=30,batch_size=args.batch_size,roi='central',roi_size=128,data_file=str(root/'experiments/fastmri_knee/data/file1000000.h5'),save_dir=str(out.parent),run_tag=out.name,tv_stable_step=False)
(out/'requested_configuration.json').write_text(json.dumps(vars(cfg),indent=2))
status();sweep.run(cfg)
old=root/'experiments/fastmri_knee/results'/('lambda_tv_sweep_ctv_a' if args.index<3 else 'lambda_tv_sweep_ctv_b')/'maps'/f'recon_mag_L{LAMS[args.index]:g}.npy'
a=np.load(old);b=np.load(out/'maps'/old.name)
validation['reconstruction_vs_previous_sweep_relative_difference']=float(np.linalg.norm(a-b)/np.linalg.norm(a))
scale=float(np.vdot(b.astype(float),a.astype(float))/np.vdot(b.astype(float),b.astype(float)))
validation["legacy_fitted_global_scale"]=scale
validation["legacy_scale_aligned_reconstruction_relative_difference"]=float(np.linalg.norm(a-scale*b)/np.linalg.norm(a))
assert validation["legacy_scale_aligned_reconstruction_relative_difference"]<1e-4,validation
(out/'batching_validation.json').write_text(json.dumps(validation,indent=2))
progress['complete']=True;status()
