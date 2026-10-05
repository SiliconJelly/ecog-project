"""Matched causal whitening ablations with explicit calibration-time controls."""
import numpy as np
from scipy import signal
from scipy.stats import binomtest
from .signal import FS, DEADLINES, fit_ar, whiten, band_filter, power_features
from .validation import fit_model, calibrated_fold, classification_summary

METHODS={"off":"Without whitening", "pre_task":"Pre-task AR(10)", "first_half":"First-half AR(10)"}
REPORTED={"source":"ECoG_G25_code/4_partner_checks/partner_checks.py with 3_oct5_analyses/common.py",
          "accuracy_without_percent":94.4,"accuracy_with_percent":97.3,"helped":11,"hurt":0,"p_value":.0004,
          "status":"Pending source reproduction in this suite run; do not substitute reported figures for measured exports"}


def paired_summary(labels, before, after, trials, bootstrap_samples=2000):
    """Pair by unique trial; repeats are averaged within trial, never treated as new subjects."""
    labels=np.asarray(labels);before=np.atleast_2d(before);after=np.atleast_2d(after)
    trials=np.asarray(trials)
    if before.shape!=after.shape or before.shape[1]!=len(labels) or len(trials)!=len(labels):
        raise ValueError("Paired predictions require matching trials and repeat counts")
    b=before==labels; a=after==labels
    delta=(a.astype(float)-b).mean(axis=0)
    rng=np.random.default_rng(0);n=len(labels);length=min(5,n);samples=[]
    for _ in range(bootstrap_samples):
        starts=rng.integers(0,n-length+1,size=int(np.ceil(n/length)))
        ids=np.concatenate([np.arange(s,s+length) for s in starts])[:n]
        samples.append(delta[ids].mean()*100)
    helped=(~b[0])&a[0];hurt=b[0]&(~a[0]);h=int(helped.sum());r=int(hurt.sum())
    discordant=h+r
    # First repeat only, with one OOF decision per original trial. Serial independence is not assured.
    p=float(binomtest(h,discordant,.5,alternative="two-sided").pvalue) if discordant else 1.
    return {"accuracy_without_percent":float(b.mean()*100),"accuracy_with_percent":float(a.mean()*100),
            "gain_percentage_points":float(delta.mean()*100),
            "gain_block_bootstrap_95_interval":np.percentile(samples,[2.5,97.5]).tolist(),
            "unique_trials":n,"repeat_count":len(before),"prediction_exposures":int(before.size),
            "first_repeat":{"helped":h,"hurt":r,"both_correct":int((b[0]&a[0]).sum()),
                            "both_wrong":int((~b[0]&~a[0]).sum()),
                            "helped_trials":trials[helped].tolist(),"hurt_trials":trials[hurt].tolist(),
                            "exact_mcnemar_two_sided_p":p},
            "uncertainty":"Five-trial moving-block bootstrap over unique chronological trial IDs; repeat outcomes averaged within trial",
            "p_value_scope":"Descriptive exact two-sided McNemar on first-repeat paired correctness; assumes independent discordances, unverified for this recording"}


def evaluate(features, labels, splits, bins, repeats):
    pred=np.zeros((repeats,len(labels)),dtype=int);coverage=np.zeros_like(pred)
    folds=len(splits)//repeats
    for f,(train,test) in enumerate(splits):
        model,_,x=fit_model(features,labels,train,bins,60)
        r=f//folds;pred[r,test]=model.predict(x[test]);coverage[r,test]+=1
    assert np.all(coverage==1)
    return pred


def first_half_split(onsets, samples):
    cutoff=samples//2
    test=np.flatnonzero(onsets>=cutoff)
    train=np.flatnonzero(onsets+int(2.25*FS)<=cutoff)
    # Match the adjacent-trial purge used in the primary suite.
    train=train[train<test[0]-1]
    if not len(train) or not len(test):raise ValueError("No eligible first-half training / second-half test trials")
    return cutoff,train,test


def fixed_pair(features_by_method, labels, train, test, deadline):
    bins=int(round((deadline-.25)/.25));pred={}
    for name,features in features_by_method.items():
        model,_,x=fit_model(features,labels,train,bins,60);pred[name]=model.predict(x[test])
    return pred


def evidence(cleaned, onsets, labels, pre_task_features, whitened_hg, random_splits, blocks, arrays, records):
    """Every fitted choice uses ECoG calibration or training labels, never glove data."""
    print("  whitening ablation: identical causal preprocessing, features, LDA and splits",flush=True)
    baseline_hg=band_filter(cleaned,[50,300]);baseline=power_features(baseline_hg,onsets)
    methods={"off":baseline,"pre_task":pre_task_features}
    result={"central_question":"Does AR whitening improve gesture decoding when other processing and evaluation choices are held fixed?",
            "primary_deadline":2.25,"feature_channels":60,"ar_order":10,
            "comparison":"Only whitening changes; CAR, causal HP/notch/band filters, bins, shrinkage LDA and split indices match",
            "reported_followup":REPORTED,"deadlines":{},"second_half":[],"forward":[],
            "calibration_intervals":{"pre_task":[2.,float((onsets[0]-2*FS)/FS)]},
            "primary_status":"Exploratory within-recording ablation; the established submission stays separate",
            "withdrawn_hypothesis":{"claim":"Still hand versus cued Paper is reliably distinguishable",
                "status":"Not established; user reports failed follow-up. No quantitative replication artifact supplied",
                "interpretation":"Withdrawn from the project headline; this does not establish absence of a task signal"}}
    arrays["without_whitening_features"]=baseline
    for deadline in DEADLINES:
        bins=int(round((deadline-.25)/.25));entry={}
        for protocol,splits,repeats in [("random",random_splits,10),("blocked",blocks,1)]:
            off=evaluate(baseline,labels,splits,bins,repeats)
            on=arrays[f"{deadline:.2f}s_60ch_{protocol}"]
            entry[protocol]={"without":classification_summary(np.tile(labels,repeats),off.ravel()),
                             "with":classification_summary(np.tile(labels,repeats),on.ravel()),
                             "paired":paired_summary(labels,off,on,np.arange(1,len(labels)+1))}
            arrays[f"whitening_off_{deadline:.2f}_{protocol}"]=off
            for f,(train,test) in enumerate(splits):
                repeat=f//(len(splits)//repeats)
                for trial in test:
                    prediction=int(off[repeat,trial])
                    records.append({"condition":f"whitening_off_{deadline:.2f}","protocol":protocol,"repeat":repeat+1,
                        "fold":f%(len(splits)//repeats)+1,"trial":int(trial+1),"cue_code":int(labels[trial]),
                        "prediction_code":prediction,"correct":bool(prediction==labels[trial]),
                        "deadline":deadline,"feature_channels":60,"whitening":"off"})
        result["deadlines"][str(deadline)]=entry
    # Calibrate the off baseline with the same training-only inner-block procedure used by the on path.
    probabilities=np.zeros((7,90,3));predictions=np.zeros((7,90),int);temperatures=[]
    for fold,(train,test) in enumerate(blocks):
        temps={}
        for i,d in enumerate(DEADLINES):
            pred,p,t,_,_,_=calibrated_fold(baseline,labels,train,test,int(round((d-.25)/.25)))
            predictions[i,test]=pred;probabilities[i,test]=p;temps[str(d)]=t
        temperatures.append({"fold":fold+1,"temperatures":temps})
    arrays["without_whitening_calibrated_predictions"]=predictions
    arrays["without_whitening_calibrated_probabilities"]=probabilities
    result["without_whitening_temperatures"]=temperatures
    result["confidence_comparison"]={str(d):classification_summary(labels,predictions[i],probabilities[i]) for i,d in enumerate(DEADLINES)}
    for row in records:
        if row.get("whitening")=="off" and row["protocol"]=="blocked":
            p=probabilities[DEADLINES.index(row["deadline"]),row["trial"]-1]
            row.update(confidence=float(p.max()),prob_rock=float(p[0]),prob_scissors=float(p[1]),prob_paper=float(p[2]))
    print("  first-half calibration: score only trials whose cues occur after the calibration cutoff",flush=True)
    cutoff,train,test=first_half_split(onsets,cleaned.shape[1])
    coefficients=fit_ar(cleaned[:,2*FS:cutoff])
    half_whitened=whiten(cleaned,coefficients,chunk=12000)
    half_features=power_features(band_filter(half_whitened,[50,300]),onsets)
    methods["first_half"]=half_features
    result["calibration_intervals"]["first_half"]=[2.,float(cutoff/FS)]
    result["first_half_scope"]="AR coefficients fixed after the first half; only second-half trials are evaluated. Earlier features are retrospectively transformed, not claimed as causal decisions before calibration was available."
    arrays.update(first_half_ar_coefficients=coefficients,first_half_features=half_features,
                  first_half_train=train,first_half_test=test)
    for deadline in [1.,2.25]:
        pred=fixed_pair(methods,labels,train,test,deadline)
        item={"deadline":deadline,"train_trials":(train+1).tolist(),"test_trials":(test+1).tolist(),
              "cutoff_seconds":cutoff/FS,"summaries":{},"paired":{}}
        for name,p in pred.items():
            item["summaries"][name]=classification_summary(labels[test],p)
            arrays[f"second_half_{name}_{deadline:.2f}"]=p
            for trial,prediction in zip(test,p):
                records.append({"condition":f"second_half_{name}","protocol":"first_half_holdout","repeat":1,"fold":1,
                    "trial":int(trial+1),"cue_code":int(labels[trial]),"prediction_code":int(prediction),
                    "correct":bool(prediction==labels[trial]),"deadline":deadline,"feature_channels":60,"whitening":name})
        for name in ["pre_task","first_half"]:item["paired"][name]=paired_summary(labels[test],pred['off'],pred[name],test+1)
        result["second_half"].append(item)
    # Match established expanding windows for off/on/first-half paths on later trials.
    for deadline in [1.,2.25]:
        collected={k:[] for k in methods};ids=[];folds=[]
        for f,start in enumerate([45,60,75]):
            tr,te=np.arange(start-1),np.arange(start,start+15)
            assert onsets[te].min()>=cutoff
            pred=fixed_pair(methods,labels,tr,te,deadline);ids.extend(te)
            folds.append({"train_trials":(tr+1).tolist(),"test_trials":(te+1).tolist()})
            for name,p in pred.items():
                collected[name].extend(p)
                for trial,prediction in zip(te,p):
                    records.append({"condition":f"whitening_forward_{name}","protocol":"whitening_forward","repeat":1,"fold":f+1,
                        "trial":int(trial+1),"cue_code":int(labels[trial]),"prediction_code":int(prediction),
                        "correct":bool(prediction==labels[trial]),"deadline":deadline,"feature_channels":60,"whitening":name})
        ids=np.array(ids);item={"deadline":deadline,"folds":folds,"summaries":{},"paired":{}}
        for name,p in collected.items():
            arrays[f"whitening_forward_{name}_{deadline:.2f}"]=np.asarray(p)
            item["summaries"][name]=classification_summary(labels[ids],p)
        for name in ["pre_task","first_half"]:item["paired"][name]=paired_summary(labels[ids],collected['off'],collected[name],ids+1)
        result["forward"].append(item)
    # Predetermined electrode 26 (team's original example), and all-channel diagnostic summaries.
    frequencies,psd=signal.welch(cleaned[:,cutoff:],fs=FS,nperseg=2400,axis=1)
    _,white_psd=signal.welch(half_whitened[:,cutoff:],fs=FS,nperseg=2400,axis=1)
    frequencies_mask=(frequencies>=2)&(frequencies<=300)
    arrays.update(whitening_spectrum_hz=frequencies[frequencies_mask],
                  whitening_spectrum_without=psd[:,frequencies_mask],whitening_spectrum_first_half=white_psd[:,frequencies_mask])
    # No amplitude/flatness statistic is itself evidence of more neural information.
    result["spectrum_scope"]="Second-half PSD after causal cleaning and before the high-gamma band filter; first-half AR fixed before this period. Electrode 26 is a fixed illustration, not selected for an effect."
    cut=int(onsets[0]+FS)
    np.testing.assert_allclose(baseline_hg[:,:cut],band_filter(cleaned[:,:cut],[50,300]),atol=1e-9,rtol=1e-9)
    result["without_whitening_future_truncation_passed"]=True
    result["limitations"]=["One previously explored participant recording; all new paired comparisons are exploratory.",
        "Repeated CV has 900 prediction exposures but only 90 distinct trials; its p-value is not computed over 900 independent observations.",
        "McNemar is descriptive here; serial dependence and prior model exploration limit inferential claims.",
        "Whitening reshapes predictable temporal structure and frequency weighting; it adds no recorded information and does not remove every artifact.",
        "Whitening supports requested-gesture decoding, not proof of movement-independent intention or separation of still hand from cued Paper.",
        "Causal filtering and past-only calibration are separate requirements. First-half calibration cannot justify first-half live predictions.",
        "This fixed CAR ablation does not establish superiority to every possible decoder or montage."]
    return result
