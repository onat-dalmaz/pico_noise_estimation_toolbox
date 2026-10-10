#!/usr/bin/env python3
"""(RTX 8000 run of 2026-10-07; copy of audit_cs_cohort_timing_20260930.py with its paths, device and field checks.)
Validate saved CS maps and aggregate slices -> volumes -> participants.

No reconstruction or GPU work. Participant linkage is derived from authorized
HDF5 headers in memory on the data host. No identifier, identifier/header hash,
volume-to-participant mapping, or per-participant volume count is exported.
"""
from pathlib import Path
import argparse,csv,json,hashlib,random,xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import h5py

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/dataNAS/people/onat/mr_recon'))
p.add_argument('--out',type=Path)
p.add_argument('--require-complete',action='store_true')
a=p.parse_args()
manifest_path=a.root/'scripts/cs_cohort_timing_manifest_20260930.csv'
manifest=list(csv.DictReader(manifest_path.open()))
assert len(manifest)==36 and len({m['subject'] for m in manifest})==12
assert len({(m['subject'],m['slice_idx']) for m in manifest})==36
manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
runner_hash=hashlib.sha256((a.root/'scripts/run_cs_cohort_timing_rtx8000_20261007.py').read_bytes()).hexdigest()
base=a.root/'experiments/fastmri_knee/results/cs_cohort_timing_equalN2000_rtx8000_20261007'
result={'expected_cases':36,'expected_volumes':12,'expected_participants':11,'complete_cases':0,'validated_cases':[],'pending_cases':[],'invalid_cases':[],'statistical_unit':'equal-weight mean of three slices within volume, then equal-weight mean of volumes within participant; seconds','agreement_regions':{'primary':'central 128 x 128 pixels; predeclared cohort and sampling-convergence metric','full_image_descriptive':'all image pixels; separately reported from unchanged maps to match Figure 6 support'}}
for m in manifest:
 case=base/m['subject']/f"slice{int(m['slice_idx']):04d}"
 marker=case/'cs_subject_metrics.json'
 if not marker.exists():result['pending_cases'].append({'volume_id':m['subject'],'slice':int(m['slice_idx'])});continue
 try:
  r=json.loads(marker.read_text())
  for key,val in {'subject':m['subject'],'slice_idx':int(m['slice_idx']),'nslices':int(m['nslices']),'n_pico':2000,'n_pmr':2000,'accel':2,'lamda':3e-7,'sigma':1e-7,'num_iters':30,'tv_prox':'complex','tv_inner':50,'tv_tol':0,'batch_size':25,'roi_size':128,'seed':123,'method_order':m['method_order'],'script_sha256':runner_hash}.items():assert r[key]==val,(key,r.get(key),val)
  assert r['device']=='Quadro RTX 8000',r['device']
  assert np.isfinite(r['t_pico_s']) and np.isfinite(r['t_pmr_s']) and r['t_pico_s']>0 and r['t_pmr_s']>0
  assert r['timing_clock']=='perf_counter; CUDA synchronize before and after each method'
  pico=np.load(case/'pico_var.npy');pmr=np.load(case/'pmr_var_f64.npy');recon=np.load(case/'recon_mag.npy')
  assert pico.shape==pmr.shape==recon.shape==tuple(r['image_shape'])
  for arr in (pico,pmr,recon):assert np.isfinite(arr).all() and np.all(arr>=0)
  cy,cx=np.array(pico.shape)//2;h=r['roi_size']//2;rs=np.s_[cy-h:cy+h,cx-h:cx+h]
  ps,ms=np.sqrt(pico[rs]),np.sqrt(pmr[rs]);nr=100*float(np.linalg.norm(ps-ms)/np.linalg.norm(ms))
  assert abs(nr-r['nrmse_std_roi_pct'])<1e-8,(nr,r['nrmse_std_roi_pct'])
  ps,ms=np.sqrt(pico),np.sqrt(pmr);nr_full=100*float(np.linalg.norm(ps-ms)/np.linalg.norm(ms));assert np.isfinite(nr_full)
  result['validated_cases'].append({'subject':m['subject'],'volume_id':m['subject'],'slice':int(m['slice_idx']),'t_pico_s':r['t_pico_s'],'t_pmr_s':r['t_pmr_s'],'nrmse_std_roi_pct':nr,'nrmse_std_full_image_pct':nr_full,'device':r['device'],'method_order':r['method_order']})
 except Exception as exc:result['invalid_cases'].append({'volume_id':m['subject'],'slice':int(m['slice_idx']),'error':str(exc)})
result['legacy_subject_field_note']='The case field subject is the historical volume-file basename, not an independent participant ID.'
result['complete_cases']=len(result['validated_cases'])
volumes={}
for r in result['validated_cases']:volumes.setdefault(r['volume_id'],[]).append(r)
result['volume_means']=[];result['participant_means']=[]
for volume,rows in sorted(volumes.items()):
 if sorted(x['slice'] for x in rows)==[13,17,21]:
  x=float(np.mean([r['t_pico_s'] for r in rows]));y=float(np.mean([r['t_pmr_s'] for r in rows]))
  result['volume_means'].append({'volume_id':volume,'n_slices':3,'N_each':2000,'mean_pico_s':x,'mean_pmr_s':y,'pmr_over_pico':y/x,'mean_nrmse_std_roi_pct':float(np.mean([r['nrmse_std_roi_pct'] for r in rows])),'mean_nrmse_std_full_image_pct':float(np.mean([r['nrmse_std_full_image_pct'] for r in rows]))})
result['status']='complete' if result['complete_cases']==36 and not result['invalid_cases'] else 'incomplete'
if result['status']=='complete':
 assert len(result['volume_means'])==12
 # Sensitive linkage never leaves this local in-memory block.
 def scalar_string(v):
  if isinstance(v,np.ndarray):assert v.size==1;v=v.item()
  return (v.decode('utf-8') if isinstance(v,(bytes,np.bytes_)) else str(v)).strip()
 private=[]
 for vr in result['volume_means']:
  with h5py.File(a.root/'experiments/fastmri_knee/data'/f"{vr['volume_id']}.h5",'r') as f:
   patient=scalar_string(f.attrs['patient_id']);header=f['ismrmrd_header'][()]
   if isinstance(header,np.ndarray):assert header.size==1;header=header.item()
   tree=ET.fromstring(header)
   uids=[(el.text or '').strip() for el in tree.iter() if el.tag.split('}')[-1]=='frameOfReferenceUID']
   assert patient and len(uids)==1 and uids[0],'Required grouping metadata missing'
   private.append((patient,uids[0],vr))
 for i in range(12):
  for j in range(12):assert (private[i][0]==private[j][0])==(private[i][1]==private[j][1]),'Metadata grouping partitions disagree'
 grouped={}
 for patient,uid,vr in private:grouped.setdefault(patient,[]).append(vr)
 assert len(grouped)==11
 sizes={str(k):v for k,v in sorted(Counter(map(len,grouped.values())).items())};assert sizes=={'1':10,'2':1}
 groups=list(grouped.values());random.SystemRandom().shuffle(groups)
 for i,rr in enumerate(groups,1):
  x=float(np.mean([r['mean_pico_s'] for r in rr]));y=float(np.mean([r['mean_pmr_s'] for r in rr]))
  result['participant_means'].append({'participant_index':i,'N_each':2000,'mean_pico_s':x,'mean_pmr_s':y,'pmr_over_pico':y/x,'mean_nrmse_std_roi_pct':float(np.mean([r['mean_nrmse_std_roi_pct'] for r in rr])),'mean_nrmse_std_full_image_pct':float(np.mean([r['mean_nrmse_std_full_image_pct'] for r in rr]))})
 del private,grouped,groups,patient,uid,rr,vr
 redacted={'status':'passed','volume_count':12,'participant_group_count':11,'partitions_concordant':True,'group_size_counts':sizes,'all36_cases_included':True,'source_manifest_sha256':manifest_sha,'aggregation':'slice_to_volume_to_participant_equal_weights','identifier_values_exported':False,'identifier_or_header_hashes_exported':False,'mapping_exported':False}
 raw=(json.dumps(redacted,indent=2)+'\n').encode();out=a.root/'reports/cs_rtx8000_20261007/cs_participant_grouping_validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
 result['participant_grouping_validation_sha256']=hashlib.sha256(raw).hexdigest()
 result['participant_grouping_validation']=redacted
 result['inference_revision']='Before final cohort aggregation, metadata revealed 12 volumes from 11 participants. All 36 cases were retained; the planned volume-level test was replaced by participant-level inference. Volume summaries remain descriptive.'
 from scipy.stats import wilcoxon
 def stats(v):
  v=np.asarray(v,dtype=float);return {'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1)),'min':float(v.min()),'max':float(v.max())}
 def aggregate(rr,unit):
  x=[r['mean_pico_s'] for r in rr];y=[r['mean_pmr_s'] for r in rr]
  return {'pico_seconds':stats(x),'pmr_seconds':stats(y),unit+'_runtime_ratio':stats([r['pmr_over_pico'] for r in rr]),'agreement_'+unit+'_mean_nrmse_std_roi_pct':stats([r['mean_nrmse_std_roi_pct'] for r in rr]),'agreement_'+unit+'_mean_nrmse_std_full_image_pct':stats([r['mean_nrmse_std_full_image_pct'] for r in rr]),'ratio_of_group_mean_times':float(np.mean(y)/np.mean(x))}
 rr=result['participant_means'];w=wilcoxon([r['mean_pico_s'] for r in rr],[r['mean_pmr_s'] for r in rr],alternative='two-sided',method='exact')
 result['final_summary']={'n_participants':11,'n_volumes':12,'n_slices':36,'N_each':2000,'statistical_unit':result['statistical_unit'],**aggregate(rr,'participant'),'wilcoxon_two_sided_exact':{'W':float(w.statistic),'p':float(w.pvalue)},'agreement_case_nrmse_std_roi_pct':stats([r['nrmse_std_roi_pct'] for r in result['validated_cases']]),'agreement_case_nrmse_std_full_image_pct':stats([r['nrmse_std_full_image_pct'] for r in result['validated_cases']]),'agreement_aggregation':'Primary and full-image descriptive SD NRMSE: average three slices within each volume, then average volumes within each participant; summarize 11 participant means. Case ranges describe all 36 saved maps.','scope':'equal-N measured runtime; no equal-precision or time-to-target claim'}
 result['volume_descriptive_summary']={'n_volumes':12,'n_slices':36,'N_each':2000,'inference':False,**aggregate(result['volume_means'],'volume')}
if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if a.require_complete and result['status']!='complete':raise SystemExit(2)
