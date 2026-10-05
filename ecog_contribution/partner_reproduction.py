"""Reproduce the provided G25 whitening checks without editing or running unrelated analyses."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from .artifacts import ROOT,sha256
from .validation import classification_summary

SOURCES=[ROOT/'ECoG_G25_code/3_oct5_analyses/common.py',
         ROOT/'ECoG_G25_code/4_partner_checks/partner_checks.py',
         ROOT/'ECoG_G25_code/4_partner_checks/whitening_audit.py']


def predictions(X,labels,lda):
    pred=np.zeros((20,len(labels)),int);c=np.zeros(len(labels))
    for seed in range(20):
        for train,test in StratifiedKFold(10,shuffle=True,random_state=seed).split(X,labels):
            p=lda().fit(X[train],labels[train]).predict(X[test]);pred[seed,test]=p
            c[test]+=(p==labels[test])/20
    return pred,c


def forward(X,labels,lda):
    out=[]
    for start in range(30,90,10):
        train,test=np.arange(start),np.arange(start,start+10)
        out.extend(lda().fit(X[train],labels[train]).predict(X[test]))
    return np.array(out)


def reproduce(data,arrays,records):
    """Execute only the supplied definitions; controlled reproductions export every prediction."""
    spec=importlib.util.spec_from_file_location('common',SOURCES[0]);common=importlib.util.module_from_spec(spec)
    old=sys.modules.get('common');sys.modules['common']=common
    try:
        spec.loader.exec_module(common)
        common._cache['y']=data
        source=SOURCES[1].read_text();definitions=source.split('half = int(on[44] + 4 * fs)')[0]
        g={'__name__':'g25_whitening_reproduction'}
        exec(compile(definitions,str(SOURCES[1]),'exec'),g)
        on,labels=g['on'],g['labels'];half=int(on[44]+4*common.fs)
        results={'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in SOURCES},
                 'cv':'20 independent StratifiedKFold seeds 0–19, 10 folds each; same held-out trial assignments in every variant',
                 'paired_test':'One-sided Monte Carlo sign-flip on the 90 per-trial correctness-fraction differences, 20,000 draws, seed 41; zero-phase test consumes RNG first',
                 'half_cutoff_seconds':half/common.fs,'causal':{},'zero_phase':{},'sensitivity':[],
                 'limitations':['Whole-record AR fitting uses held-out signal values: causal filter application is not fully prospective calibration.',
                    'Original first-half-calibrated expanding evaluation includes trials 31–45 before calibration finishes; it is retrospective for those trials.',
                    'The first-45 → last-45 causal half-split has calibration finished before test cue 46, and is the relevant first-half check.',
                    'The source p-value uses a one-sided sign-flip, not two-sided McNemar; time dependence and prior exploration limit inference.']}
        for causal,tag in [(False,'zero_phase'),(True,'causal')]:
            print(f'  reproducing G25 {tag}: 20 seeds, paired predictions and original sign-flip',flush=True)
            e=common.clean(causal)
            X0=common.decoder_features(g['hg_from'](e,causal),on)
            Xw=common.decoder_features(g['hg_from'](g['whiten'](e),causal),on)
            Xh=common.decoder_features(g['hg_from'](g['whiten'](e,slice(0,half)),causal),on)
            p0,c0=predictions(X0,labels,g['lda']);pw,cw=predictions(Xw,labels,g['lda']);d=cw-c0
            r0,rw,rh=[forward(x,labels,g['lda']) for x in [X0,Xw,Xh]]
            h0=g['lda']().fit(X0[:45],labels[:45]).predict(X0[45:])
            hw=g['lda']().fit(Xh[:45],labels[:45]).predict(Xh[45:])
            results[tag]={'accuracy_without_percent':float(c0.mean()*100),'accuracy_with_percent':float(cw.mean()*100),
                'gain_percentage_points':float(d.mean()*100),'trials_helped':int((d>0).sum()),'trials_hurt':int((d<0).sum()),
                'helped_trials':(np.flatnonzero(d>0)+1).tolist(),'hurt_trials':(np.flatnonzero(d<0)+1).tolist(),
                'p_signflip_one_sided':g['signflip'](d),'unique_trials':90,'prediction_exposures':1800,
                'forward_retrospective':{name:classification_summary(labels[30:],p) for name,p in [('off',r0),('whole_record',rw),('first_half',rh)]},
                'first_half_holdout':{'train_trials':list(range(1,46)),'test_trials':list(range(46,91)),
                    'without':classification_summary(labels[45:],h0),'first_half':classification_summary(labels[45:],hw)}}
            arrays.update({f'partner_{tag}_off':p0,f'partner_{tag}_whole':pw,f'partner_{tag}_trial_gain':d,
                           f'partner_{tag}_half_off':h0,f'partner_{tag}_half_on':hw,
                           f'partner_{tag}_forward_off':r0,f'partner_{tag}_forward_whole':rw,f'partner_{tag}_forward_half':rh})
            for method,p in [('off',p0),('whole_record',pw)]:
                for seed in range(20):
                    for fold,(train,test) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(X0,labels)):
                        if causal and method=='off':
                            arrays[f'partner_seed_{seed}_fold_{fold}_train']=train
                            arrays[f'partner_seed_{seed}_fold_{fold}_test']=test
                        for trial in test:
                            records.append({'condition':f'partner_{tag}_{method}','protocol':'partner_20seed_cv',
                                'repeat':seed+1,'fold':fold+1,'trial':int(trial+1),'cue_code':int(labels[trial]),
                                'prediction_code':int(p[seed,trial]),'correct':bool(p[seed,trial]==labels[trial]),
                                'deadline':2.25,'feature_channels':60,'whitening':method,'calibration_scope':'whole_record' if method!='off' else 'none'})
            for method,p in [('off',h0),('first_half',hw)]:
                for trial,pred in zip(range(45,90),p):
                    records.append({'condition':f'partner_{tag}_half_{method}','protocol':'partner_half_holdout','repeat':1,'fold':1,
                        'trial':trial+1,'cue_code':int(labels[trial]),'prediction_code':int(pred),'correct':bool(pred==labels[trial]),
                        'deadline':2.25,'feature_channels':60,'whitening':method})
            if causal:
                # A fixed difference is a fitted-free mechanism control, not a new headline selected after searching.
                for name,transformed in [('first_difference',np.c_[e[:,:1],np.diff(e,axis=1)])]:
                    X=common.decoder_features(g['hg_from'](transformed,True),on)
                    p,c=predictions(X,labels,g['lda']);arrays[f'partner_causal_{name}']=p
                    results['sensitivity'].append({'variant':name,'summary':classification_summary(np.tile(labels,20),p.ravel()),
                                                  'calibration':'No fitting; retrospective 20-seed CV comparison'})
            del e,X0,Xw,Xh
        # AR-order sensitivity follows the supplied whitening_audit.py definitions.
        audit_defs=SOURCES[2].read_text().split('e_zp = C.clean(False)')[0]
        ag={'__name__':'g25_order_reproduction'};exec(compile(audit_defs,str(SOURCES[2]),'exec'),ag)
        e=common.clean(True)
        for order in [5,20]:
            X=ag['feats'](ag['whiten'](e,order),True)
            p,c=predictions(X,labels,g['lda']);arrays[f'partner_causal_ar{order}']=p
            results['sensitivity'].append({'variant':f'AR({order}) whole-record','summary':classification_summary(np.tile(labels,20),p.ravel()),
                                          'calibration':'Whole recording; retrospective sensitivity, not a deployment choice'})
        results['matches_reported_followup']={
            'rounded_accuracy':round(results['causal']['accuracy_without_percent'],1)==94.4 and round(results['causal']['accuracy_with_percent'],1)==97.3,
            'helped_hurt':results['causal']['trials_helped']==11 and results['causal']['trials_hurt']==0,
            'rounded_p':round(results['causal']['p_signflip_one_sided'],4)==.0004}
        return results
    finally:
        common._cache.clear()
        if old is None:sys.modules.pop('common',None)
        else:sys.modules['common']=old
