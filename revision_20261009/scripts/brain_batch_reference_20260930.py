"""Reference-only independent PMR, batched CG with per-RHS stopping.

This does not supply the timed PMR comparison. Validate exact operator and
serial-solver agreement before generating any reference samples.
"""
from pathlib import Path
import argparse,sys,math,json,time,os,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.run_brain_noncart import build_brain_problem
from scripts.noncart_audit_core import solve_cg_true_residual
from mr_recon.pad import PadLast
from mr_recon.fourier import fft,ifft

ap=argparse.ArgumentParser();ap.add_argument('--R',type=int,default=2);ap.add_argument('--lam',type=float,default=.002);ap.add_argument('--N',type=int,default=0);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--seed',type=int,default=20263930);ap.add_argument('--label',default='validation');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
base=ROOT/'experiments/brain_spiral/operating_point_20260930';out=base/f'reference_batch_R{a.R}_lam{a.lam:g}_{a.label}'
out.mkdir(parents=True,exist_ok=True)
run_source=out/'run_source_20260930.py'
if run_source.exists():assert run_source.read_bytes()==Path(__file__).read_bytes(),'Refusing changed experiment code on resume'
else:run_source.write_bytes(Path(__file__).read_bytes())
if (out/'complete.json').exists():
 assert a.resume,'A completed reference can only be explicitly extended'
 previous=json.loads((out/'config.json').read_text());assert a.N>previous['N']
 assert previous['seed']==a.seed and previous['batch']==a.batch
 (out/f"complete_N{previous['N']}.json").write_text((out/'complete.json').read_text())
 (out/f"config_N{previous['N']}.json").write_text((out/'config.json').read_text())
 (out/'complete.json').unlink()
assert a.resume or not (out/'config.json').exists(),'Use explicit resume for a started reference'
p=build_brain_problem(a.R,torch.device('cuda:0'),out=out/'setup',reference='gridding')
actualR=p.ksp_ref.shape[-1]/p.ksp_acc.shape[-1];sigma=10.;rtol=1e-5
assert a.R==2 and a.lam==.002,'Reference is frozen to the selected R2 lambda_norm=.002 operating point'
lambda_physical=0.06097300155639648
def makeH(A,mu,lam):
 assert A.imperf_model is None and A.use_toeplitz
 assert len(A.im_size)==2
 pad=PadLast(A.toep_kerns.shape[-2:],A.im_size)
 def H(x):
  sx=A.mps[None,:,:,:]*x[:,None,:,:]
  y=pad.adjoint(ifft(A.toep_kerns[None,None,:,:]*fft(pad.forward(sx),dim=(-2,-1)),dim=(-2,-1)))
  return (y*A.mps.conj()[None,:,:,:]).sum(1)/mu+lam*x
 return H
def cg_batch(H,b):
 x=torch.zeros_like(b);r=b.clone();v=r.clone();rho=(r.conj()*r).real.sum((-2,-1));norm=torch.linalg.vector_norm(b,dim=(-2,-1));active=norm>0;used=torch.zeros(b.shape[0],device=b.device,dtype=torch.long)
 last=torch.ones_like(norm);restarts=torch.zeros_like(used)
 for k in range(1,1201):
  hv=H(v);curv=(v.conj()*hv).real.sum((-2,-1))
  if bool((active&((curv<=0)|~torch.isfinite(curv))).any()):raise RuntimeError('Invalid active curvature')
  alpha=torch.where(active,rho/torch.where(active,curv,torch.ones_like(curv)),torch.zeros_like(curv))
  x=x+alpha[:,None,None]*v;rec=r-alpha[:,None,None]*hv;used[active]=k
  restart=torch.zeros_like(active)
  if k%5==0 or k==1200:
   true=b-H(x);tn=torch.linalg.vector_norm(true,dim=(-2,-1));last=tn/norm.clamp_min(1e-30)
   active=active&(last>rtol)
   if not bool(active.any()):break
   drift=torch.linalg.vector_norm(true-rec,dim=(-2,-1))/tn.clamp_min(1e-30)
   restart=active&(drift>.1);restarts+=restart.long()
  rho_new=(rec.conj()*rec).real.sum((-2,-1));beta=rho_new/rho.clamp_min(1e-30)
  rr=torch.where(restart[:,None,None],true if k%5==0 or k==1200 else rec,rec)
  vv=torch.where(restart[:,None,None],rr,rec+beta[:,None,None]*v)
  r=rr*active[:,None,None];v=vv*active[:,None,None];rho=(r.conj()*r).real.sum((-2,-1))
 if bool(active.any()):raise RuntimeError(f'Batch failed true-residual certificate: {last.max().item()}')
 return x,{'max_iterations':int(used.max()),'max_true_residual':float(last.max()),'max_restarts':int(restarts.max())}

arms={}
for n,A,mu in [('ref',p.A_ref,p.max_eigen_ref),('acc',p.A_acc,p.max_eigen_acc)]:
 lam=lambda_physical/mu;H=makeH(A,mu,lam);hs=lambda x,A=A,mu=mu,lam=lam:A.normal(x)/mu+lam*x
 arms[n]=(A,mu,H,hs)
validation={};torch.manual_seed(37701)
for n,(A,mu,H,hs) in arms.items():
 z=torch.randn((2,*p.image.shape),device='cuda',dtype=p.image.dtype);expected=torch.stack([hs(zz) for zz in z]);observed=H(z)
 op=float(torch.linalg.norm(observed-expected)/torch.linalg.norm(expected));assert op<1e-6,op
 noise=torch.randn((2,*(p.ksp_ref.shape if n=='ref' else p.ksp_acc.shape)),device='cuda',dtype=p.ksp_ref.dtype)*sigma
 rhs=torch.stack([A.adjoint(nn)/math.sqrt(mu) for nn in noise]);xb,info=cg_batch(H,rhs);errors=[];residuals=[]
 for i in range(2):
  xs,si=solve_cg_true_residual(hs,rhs[i],max_iter=1200,rtol=rtol,true_residual_every=5);assert si['converged']
  errors.append(float(torch.linalg.norm(xs-xb[i])/torch.linalg.norm(xs)));residuals.append(float(torch.linalg.norm(hs(xb[i])-rhs[i])/torch.linalg.norm(rhs[i])))
 assert max(errors)<2e-4,errors
 assert max(residuals)<1.5e-5,residuals
 validation[n]={'normal_operator_relative_difference':op,'batch_vs_serial_image_relative_errors':errors,'batch_solution_serial_operator_residuals':residuals,**info}
(out/'validation.json').write_text(json.dumps(validation,indent=2));print('VALIDATION',json.dumps(validation),flush=True)
config={'reference_only':True,'timing_comparison_uses_separate_production_implementations':True,'R':actualR,'lambda_norm_acc':a.lam,'common_physical_lambda':lambda_physical,'mu_ref':p.max_eigen_ref,'mu_acc':p.max_eigen_acc,'N':a.N,'batch':a.batch,'seed':a.seed,'complex_RMS':sigma,'relative_true_residual':rtol,'max_iter':1200,'operator':'same Toeplitz normal as serial reconstruction, no B0','shared_noise_between_arms':True,'gpu':torch.cuda.get_device_name(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
config.update({'hostname':os.uname().nodename,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'dependency_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'scripts/run_brain_noncart.py',ROOT/'scripts/noncart_audit_core.py',ROOT/'src/mr_recon/covariance_apply.py']}})
(out/'config.json').write_text(json.dumps(config,indent=2))
if a.N==0:(out/'validation_complete.json').write_text(json.dumps(validation,indent=2));sys.exit(0)
gen=torch.Generator(device='cuda').manual_seed(a.seed);shape=p.image.shape;mask_path=base/'R2_lam0.002_converged_K0_structure1000_tol1e5/analysis_mask.npy';mask=np.load(mask_path);assert np.array_equal(mask,p.support.cpu().numpy()),'Foreground changed from frozen candidate';np.save(out/'analysis_mask.npy',mask)
means={n:torch.zeros(shape,dtype=torch.complex128) for n in arms};m2={n:torch.zeros(shape,dtype=torch.float64) for n in arms};cross_m2=torch.zeros(shape,dtype=torch.complex128);n=0;elapsed=0;rows=[];solver_max={'max_iterations':0,'max_true_residual':0.,'max_restarts':0}
if a.resume and (out/'checkpoint.pt').exists():
 ck=torch.load(out/'checkpoint.pt',map_location='cpu');means=ck['means'];m2=ck['m2'];cross_m2=ck['cross_m2'];solver_max=ck['solver_max'];n=ck['n'];elapsed=ck['seconds'];gen.set_state(ck['rng']);rows=json.loads((out/'progress.json').read_text())
def now():torch.cuda.synchronize();return time.perf_counter()
start=now();next_save=min(a.N,((n//1000)+1)*1000)
while n<a.N:
 bs=min(a.batch,a.N-n,next_save-n);noise=torch.randn((bs,*p.ksp_ref.shape),device='cuda',dtype=p.ksp_ref.dtype,generator=gen)*sigma
 batch_x={};batch_means={};batch_deltas={}
 for name,(A,mu,H,hs) in arms.items():
  nn=noise if name=='ref' else noise[:,:,:,::a.R];rhs=torch.stack([A.adjoint(q)/math.sqrt(mu) for q in nn]);xx,info=cg_batch(H,rhs);xx=(xx/math.sqrt(mu)).detach().cpu().to(torch.complex128)
  for key,value in info.items():solver_max[key]=max(solver_max[key],value)
  bm=xx.mean(0);batch_x[name]=xx;batch_means[name]=bm;batch_deltas[name]=bm-means[name];bm2=((xx-bm).conj()*(xx-bm)).real.sum(0);delta=bm-means[name]
  m2[name]+=bm2+delta.abs().square()*(n*bs/(n+bs));means[name]+=delta*(bs/(n+bs))
 cross_m2+=((batch_x['ref']-batch_means['ref']).conj()*(batch_x['acc']-batch_means['acc'])).sum(0)+batch_deltas['ref'].conj()*batch_deltas['acc']*(n*bs/(n+bs))
 n+=bs
 if n==next_save:
  elapsed+=now()-start
  dr=m2['ref'].numpy()/((n-1)*sigma*sigma);da=m2['acc'].numpy()/((n-1)*sigma*sigma);g=np.sqrt(np.maximum(da,0)/np.maximum(dr,1e-30)/actualR)
  for label,x in [('var_ref',dr),('var_acc',da),('g',g)]:np.save(out/f'pmr_N{n}_{label}.npy',x)
  checkpoint_tmp=out/'checkpoint.pt.tmp';torch.save({'n':n,'means':means,'m2':m2,'cross_m2':cross_m2,'solver_max':solver_max,'rng':gen.get_state(),'seconds':elapsed},checkpoint_tmp);checkpoint_tmp.replace(out/'checkpoint.pt');rows.append({'N':n,'seconds':elapsed});(out/'progress.json').write_text(json.dumps(rows,indent=2));print('REFERENCE_COUNT',n,flush=True);next_save=min(a.N,n+1000);start=now()
(out/'solver_diagnostics.json').write_text(json.dumps(solver_max,indent=2))
(out/'complete.json').write_text(json.dumps(config,indent=2))
