import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json, csv
import numpy as np
import scipy.io as sio
import scipy.signal as ss
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, log_loss
from threadpoolctl import threadpool_limits
threadpool_limits(1)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
fs=1200
source=Path(r'C:\Users\AbriK\Documents\ecog-hand-pose\ECoG_Handpose.mat')
y=sio.loadmat(source)['y']; cue=y[61]; glove=y[62:67]
on=np.where((np.diff(cue)!=0)&(cue[1:]!=0))[0]+1
off=np.where((np.diff(cue)!=0)&(cue[1:]==0))[0]+1
labels=cue[on].astype(int); prev=np.r_[0,labels[:-1]]
assert len(on)==len(off)==90 and np.all(off>on)
print('Loaded',y.shape,'class counts',np.bincount(labels)[1:],flush=True)
e=y[1:61]-y[1:61].mean(axis=0)
e=ss.sosfiltfilt(ss.butter(4,1,'highpass',fs=fs,output='sos'),e,axis=1)
for f in [50,100,150,200,250,300]:
    b,a=ss.iirnotch(f,30,fs=fs); e=ss.filtfilt(b,a,e,axis=1)
hg=ss.sosfiltfilt(ss.butter(4,[50,300],'bandpass',fs=fs,output='sos'),e,axis=1)**2
bins=np.arange(.25,2.26,.25)
X=np.array([np.log10(np.stack([hg[:,o+int(a*fs):o+int(b*fs)].mean(1) for a,b in zip(bins[:-1],bins[1:])],axis=1)).ravel() for o in on])
model=lambda:LDA(solver='lsqr',shrinkage='auto')
acc=cross_val_score(model(),X,labels,cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=10,random_state=0))
fp=labels!=3
accfp=cross_val_score(model(),X[fp],labels[fp],cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=10,random_state=0))
probs=np.zeros((len(on),3)); correct=np.zeros(len(on)); first=None
for seed in range(20):
    pred=np.zeros(len(on),int)
    for tr,te in StratifiedKFold(10,shuffle=True,random_state=seed).split(X,labels):
        m=model().fit(X[tr],labels[tr]); pred[te]=m.predict(X[te]); probs[te]+=m.predict_proba(X[te])/20
    correct+=(pred==labels)/20
    if seed==0:first=pred.copy()
print('Reproduced decoder',acc.mean(),acc.std(),accfp.mean(),flush=True)
G0=np.array([glove[:,o-fs:o].mean(1) for o in on])
Ghold=np.array([glove[:,o+int(.75*fs):o+int(1.75*fs)].mean(1) for o in on])
# All behavior summaries defined without consulting neural errors.
gs=ss.sosfiltfilt(ss.butter(2,10,fs=fs,output='sos'),glove,axis=1)
amp=[]; rt=[]; vel=[]
for o,base in zip(on,G0):
    dist=np.sqrt(((gs[:,o:o+2*fs]-base[:,None])**2).mean(0))
    a=np.percentile(dist,95); amp.append(a)
    hits=np.where(dist>max(.02,.3*a))[0]; rt.append(hits[0]/fs if len(hits) else np.nan)
    vel.append(np.percentile(np.sqrt((np.diff(gs[:,o:o+2*fs],axis=1)**2).mean(0))*fs,95))
amp=np.array(amp); rt=np.array(rt); vel=np.array(vel)
beta=ss.sosfiltfilt(ss.butter(4,[13,30],'bandpass',fs=fs,output='sos'),e,axis=1)**2
motor=[15,25,26,36,46]
B=np.array([10*np.log10(beta[motor,o-fs:o].mean(1)).mean() for o in on])
H=np.array([10*np.log10(hg[motor,o+int(.4*fs):o+int(1.2*fs)].mean()) for o in on])
gap=np.r_[np.nan,(on[1:]-off[:-1])/fs]
rows=[]
for i in range(len(on)):
    row=dict(trial=i+1,cue=int(labels[i]),previous=int(prev[i]),onset_s=on[i]/fs,rest_s=gap[i],movement_rms=amp[i],reaction_s=rt[i],velocity_rms=vel[i],wrong_fraction=1-correct[i],prediction=int(first[i]),beta_db=B[i],response_db=H[i],true_class_probability=probs[i,labels[i]-1])
    for j in range(5):row[f'pre_finger_{j+1}']=G0[i,j];row[f'hold_finger_{j+1}']=Ghold[i,j]
    rows.append(row)
with (OUT/'trial_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
transition=np.array([[np.sum((prev==a)&(labels==b)) for b in (1,2,3)] for a in (1,2,3)])
print('Transition table',transition.tolist(),flush=True)

# Paired cross-context versus within-context evaluation. Same held-out target
# trials and same training count for each current class, for both models.
# Exclude trial zero because it has no observed history.
states=np.where(prev==3,0,1); valid=prev>0; rng=np.random.default_rng(42)
history=[]
for target in [0,1]:
    targetidx=np.where(valid&(states==target))[0]; sourceidx=np.where(valid&(states!=target))[0]
    bytarget=[targetidx[labels[targetidx]==c] for c in (1,2,3)]
    bysource=[sourceidx[labels[sourceidx]==c] for c in (1,2,3)]
    ntrain=min(min(map(len,bysource)),min(map(len,bytarget))-2)
    assert ntrain>=2
    values=[]
    for repeat in range(100):
        te=np.concatenate([rng.choice(k,2,replace=False) for k in bytarget])
        tw=np.concatenate([rng.choice(np.setdiff1d(k,te),ntrain,replace=False) for k in bytarget])
        tc=np.concatenate([rng.choice(k,ntrain,replace=False) for k in bysource])
        out=[]
        for tr in (tw,tc):
            m=model().fit(X[tr],labels[tr]); p=m.predict_proba(X[te]); pr=m.classes_[p.argmax(1)]
            out.extend([balanced_accuracy_score(labels[te],pr),log_loss(labels[te],p,labels=[1,2,3])])
        values.append(out)
    values=np.array(values)
    result=dict(target='after open' if target==0 else 'after movement',train_per_class=int(ntrain),test_per_class=2,within_accuracy=float(values[:,0].mean()),cross_accuracy=float(values[:,2].mean()),accuracy_difference=float((values[:,2]-values[:,0]).mean()),within_log_loss=float(values[:,1].mean()),cross_log_loss=float(values[:,3].mean()),split_difference_5_95=np.percentile(values[:,2]-values[:,0],[5,95]).tolist())
    history.append(result); print('History',result,flush=True)

chron=[]
for block in np.array_split(np.arange(90),5):
    tr=np.setdiff1d(np.arange(90),np.arange(max(0,block[0]-1),min(90,block[-1]+2)))
    m=model().fit(X[tr],labels[tr]); chron.append(dict(test_trials=[int(block[0]+1),int(block[-1]+1)],accuracy=float(np.mean(m.predict(X[block])==labels[block]))))
openidx=np.where(labels==3)[0]
rho,p=stats.spearmanr(amp[openidx],1-correct[openidx])
summary=dict(source=str(source),shape=list(y.shape),class_counts=np.bincount(labels)[1:].tolist(),trial_duration_range=((off-on)/fs).tolist(),decoder=dict(accuracy=float(acc.mean()),fold_sd=float(acc.std()),fist_peace=float(accfp.mean()),confusion=confusion_matrix(labels,first).tolist()),transition_counts=transition.tolist(),history=history,chronological_blocks=chron,chronological_mean=float(np.mean([z['accuracy'] for z in chron])),open_movement_error_spearman=dict(rho=float(rho),p=float(p)),open_trials=[rows[i] for i in openidx],notes=['Split percentiles are descriptive split variability, not confidence intervals.','All contexts come from one continuous recording.','Behavioral threshold is exploratory, not calibrated to physical joint angles.','Zero-phase preprocessing retained for exact baseline reproduction; no causal pre-cue claim.'])
np.savez_compressed(ROOT/'work/features.npz',X=X,labels=labels,onsets=on,prev=prev)
with (OUT/'analysis_results.json').open('w') as f:json.dump(summary,f,indent=2,allow_nan=True)
fig,ax=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
for c,name,col in [(1,'Fist','#2878b5'),(2,'Peace','#e87532'),(3,'Open','#38956d')]:
    k=labels==c; ax[0].scatter(amp[k],1-correct[k],label=name,color=col,alpha=.75)
ax[0].set(xlabel='Glove movement amplitude (RMS)',ylabel='Fraction of decoder runs wrong',title='Does actual movement explain errors?');ax[0].legend()
for i,r in enumerate(history):
    ax[1].bar(i-.17,r['within_accuracy']*100,.34,color='#2878b5',label='Same history' if i==0 else None)
    ax[1].bar(i+.17,r['cross_accuracy']*100,.34,color='#e87532',label='Different history' if i==0 else None)
ax[1].set(xticks=[0,1],xticklabels=['Test after open','Test after movement'],ylabel='Balanced accuracy (%)',ylim=(0,100),title='Matched training size, same test trials');ax[1].axhline(100/3,color='gray',ls='--');ax[1].legend()
for c,name in [(1,'Fist'),(2,'Peace'),(3,'Open')]:
    ids=np.where(labels==c)[0]; ax[2].scatter(ids+1,correct[ids]*100,label=name)
ax[2].set(xlabel='Trial in recording',ylabel='Mean held-out accuracy (%)',title='Where errors occur');ax[2].legend()
fig.savefig(OUT/'first_tests.png',dpi=180);plt.close(fig)
print('DONE; open movement/error association',rho,p,'chronological',summary['chronological_mean'],flush=True)
