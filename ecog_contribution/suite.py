"""Full deterministic research suite; writes only to its explicit output folder."""
from datetime import datetime,timezone
from pathlib import Path
import importlib.metadata
import json
import time
import platform
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal as scipy_signal
from scipy.special import softmax
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits
from .signal import FS,EDGES,DEADLINES,BUDGETS,build_features,clean,whiten,band_filter
from .validation import blocked,fit_model,rank_channels,calibrated_fold,classification_summary,adaptive,temperature
from .behavior import neuroscience,history_analysis,NAMES
from .artifacts import ROOT,DEFAULT_OUTPUT,sha256,code_hash

CONFIG={"deadlines":DEADLINES,"budgets":BUDGETS,"seed":0,"random_folds":10,"random_repeats":10,
        "temporal_folds":5,"purge_trials":1,"inner_folds":3,"confidence_threshold":.9,
        "primary_analysis":"whitening_on_vs_off","secondary_history":"previous_movement_amount","history_bootstrap_samples":1000,"history_block_length":5}


def load_recording(path):
    data=loadmat(path)["y"]
    if data.shape!=(67,507025):
        raise ValueError("This case study requires the documented 67 x 507025 recording")
    if not np.isfinite(data).all():
        raise ValueError(f"Recording contains {np.sum(~np.isfinite(data))} nonfinite values; no trials silently removed")
    if not np.allclose(np.diff(data[0]),1/FS,atol=1e-8):
        raise ValueError("Unexpected sample-time spacing")
    cue=data[61]
    if not set(np.unique(cue)).issubset({0,1,2,3}):
        raise ValueError("Unexpected cue codes")
    onsets=np.flatnonzero((np.diff(cue)!=0)&(cue[1:]!=0))+1
    offsets=np.flatnonzero((np.diff(cue)!=0)&(cue[1:]==0))+1
    labels=cue[onsets].astype(int)
    if len(onsets)!=90 or len(offsets)!=90 or not np.array_equal(np.bincount(labels)[1:],[30,30,30]):
        raise ValueError("Expected 90 complete balanced cue trials")
    if np.any(offsets<=onsets) or np.any(onsets[1:]<=offsets[:-1]):
        raise ValueError("Cue intervals overlap or are incomplete")
    return data,onsets,offsets,labels


def quality(raw,onsets):
    pre=raw[:,2*FS:onsets[0]-2*FS]
    std=pre.std(axis=1)
    differences=np.diff(pre,axis=1)
    center=np.median(differences,axis=1)
    scale=1.4826*np.median(np.abs(differences-center[:,None]),axis=1)
    scale=np.maximum(scale,np.finfo(float).eps)
    frequencies,psd=scipy_signal.welch(pre,fs=FS,nperseg=2400,axis=1)
    mains=(frequencies>=49)&(frequencies<=51)
    nearby=((frequencies>=44)&(frequencies<=48))|((frequencies>=52)&(frequencies<=56))
    ratios=psd[:,mains].mean(axis=1)/np.maximum(psd[:,nearby].mean(axis=1),np.finfo(float).tiny)
    scores=[]
    for onset in onsets:
        delta=np.diff(raw[:,onset:onset+2*FS],axis=1)
        scores.append(float(np.max(np.abs(delta-center[:,None])/scale[:,None])))
    return {"nonfinite_samples":int(np.sum(~np.isfinite(raw))),"flat_channels_1_based":(np.flatnonzero(std<=1e-12)+1).tolist(),
            "line_noise_ratio_50hz_per_channel":ratios.tolist(),"trial_transient_scores":scores,
            "transient_threshold":12.,"flagged_trials_1_based":(np.flatnonzero(np.array(scores)>12)+1).tolist(),
            "interpretation":"Large sample differences relative to pre-task median/MAD; flags do not diagnose pathology or alter inclusion."}


def evaluate_grid(features,labels,splits,bins,budget,repeats):
    predictions=np.zeros((repeats,len(labels)),dtype=int);coverage=np.zeros_like(predictions)
    selections=[];fold_scores=[];folds=len(splits)//repeats
    for i,(train,test) in enumerate(splits):
        model,selected,x=fit_model(features,labels,train,bins,budget)
        pred=model.predict(x[test]);r=i//folds
        predictions[r,test]=pred;coverage[r,test]+=1;selections.append(selected)
        fold_scores.append(float(np.mean(labels[test]==pred)))
    if not np.all(coverage==1):
        raise AssertionError("Incorrect out-of-fold coverage")
    metric=classification_summary(np.tile(labels,repeats),predictions.ravel())
    metric["unique_trials"]=len(labels);metric["fold_sd_percent"]=float(np.std(fold_scores)*100)
    return metric,predictions,np.asarray(fold_scores),selections


def run(data_path,output=DEFAULT_OUTPUT):
    data_path=Path(data_path).resolve();output=Path(output).resolve()
    protected_folders=[ROOT/"source",ROOT/"team_submission_whole_record",ROOT/"results/whitening",ROOT/"results/contribution_angles",ROOT/"ECoG_G25_code"]
    if output==ROOT or any(output==folder or folder in output.parents for folder in protected_folders):
        raise ValueError("Output cannot overwrite protected source, submission, or historical results")
    output.mkdir(parents=True,exist_ok=True)
    # Incomplete reruns cannot be mistaken for a valid cache.
    (output/"manifest.json").unlink(missing_ok=True)
    protected=[data_path,*sorted((ROOT/"source").glob("*.py")),ROOT/"whitening_compare.py",ROOT/"explore_ecog_angles.py",
               ROOT/"run_team_submission.py",ROOT/"team_submission_whole_record/classify_whitened.py",
               ROOT/"team_submission_whole_record/results/on/arrays.npz",
               ROOT/"team_submission_whole_record/results/off/arrays.npz",ROOT/"results/whitening/comparison_arrays.npz"]
    from .partner_reproduction import SOURCES
    protected.extend(SOURCES)
    original_hashes={str(p):sha256(p) for p in protected}
    started=time.perf_counter();data,onsets,offsets,labels=load_recording(data_path)
    raw,glove=data[1:61],data[62:67]
    random_splits=list(RepeatedStratifiedKFold(n_splits=10,n_repeats=10,random_state=0).split(np.zeros((90,1)),labels))
    blocks=list(blocked(np.arange(90)))
    metrics={"schema_version":1,"status":"exploratory","config":CONFIG,"established_accuracy_percent":96.8888888889,
             "participant_metadata":"Team documentation describes one epilepsy participant; organizer matrix has no diagnosis fields. No stroke-specific claim.",
             "grid":{},"folds":[],"forward":[],"acquisition":[],"channel_loss":[],"montage":{}}
    arrays={"labels":labels,"onsets":onsets,"offsets":offsets};records=[]
    print("1/8 Reproducing historical fold scores",flush=True)
    metrics["historical_reproduction"]={}
    for name,path,key,score_key in [
        ("original",ROOT/"team_submission_whole_record/results/off/arrays.npz","features","three_class_scores"),
        ("whole_record_ar",ROOT/"team_submission_whole_record/results/on/arrays.npz","features","three_class_scores"),
        ("initial_rest_ar",ROOT/"results/whitening/comparison_arrays.npz","whitening_features","whitening_scores")]:
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["labels"],labels)
            summary,_,scores,_=evaluate_grid(saved[key].reshape(90,60,8),labels,random_splits,8,60,10)
            np.testing.assert_allclose(scores,saved[score_key],atol=1e-12,rtol=0)
        metrics["historical_reproduction"][name]={"exact_fold_scores":True,"accuracy_percent":summary["accuracy_percent"]}
    print("2/8 Building causal signals, trial behavior and quality diagnostics",flush=True)
    metrics["quality"]=quality(raw,onsets)
    features,coefficients,timing,cleaned,hg=build_features(raw,onsets,return_signals=True)
    arrays.update(features=features,ar_coefficients=coefficients)
    metrics["processing_timing"]=timing
    # Future truncation AND batch/chunk equivalence are checked against the real recording prefix.
    cut=int(onsets[0]+FS)
    prefix_clean,_=clean(raw[:,:cut],chunk=cut)
    np.testing.assert_allclose(prefix_clean,cleaned[:,:cut],atol=1e-10,rtol=1e-10)
    prefix_hg=band_filter(whiten(prefix_clean,coefficients,chunk=cut),[50,300],chunk=cut)
    np.testing.assert_allclose(prefix_hg,hg[:,:cut],atol=1e-9,rtol=1e-9)
    metrics["causal_checks"]={"future_truncation_passed":True,"batch_chunk_passed":True}
    table,traces=neuroscience(cleaned,glove,onsets,offsets,labels,int(onsets[0]-2*FS))
    table["transient_score"]=metrics["quality"]["trial_transient_scores"]
    table["quality_flag"]=table.transient_score>12
    metrics["history"],history_arrays=history_analysis(table)
    np.savez_compressed(output/"history_arrays.npz",**history_arrays)
    np.savez_compressed(output/"display_traces.npz",**traces)
    table.to_csv(output/"trial_table.csv",index=False)
    del traces
    print("3/8 Joint speed/channel grid: 28 settings, identical repeated and temporal splits",flush=True)
    for deadline in DEADLINES:
        bins=int(round((deadline-.25)/.25))
        for budget in BUDGETS:
            key=f"{deadline:.2f}s_{budget}ch";item={"deadline":deadline,"budget":budget}
            for protocol,splits,repeats in [("random",random_splits,10),("blocked",blocks,1)]:
                summary,pred,scores,selections=evaluate_grid(features,labels,splits,bins,budget,repeats)
                item[protocol]=summary;arrays[key+"_"+protocol]=pred
                if protocol=="random":
                    arrays[key+"_selection_frequency"]=np.bincount(np.concatenate(selections),minlength=60)/len(selections)
                for fold,(train,test) in enumerate(splits):
                    repeat=fold//(len(splits)//repeats)
                    for trial in test:
                        records.append({"condition":key,"protocol":protocol,"repeat":repeat+1,"fold":fold%(len(splits)//repeats)+1,
                            "trial":int(trial+1),"cue":NAMES[int(labels[trial])],"cue_code":int(labels[trial]),
                            "prediction":NAMES[int(pred[repeat,trial])],"prediction_code":int(pred[repeat,trial]),
                            "correct":bool(pred[repeat,trial]==labels[trial]),"deadline":deadline,"feature_channels":budget})
            metrics["grid"][key]=item
        print(f"  deadline {deadline:.2f}s complete",flush=True)
    print("4/8 Nested joint selection and calibrated fixed-model confidence",flush=True)
    nested=np.zeros(90,dtype=int);confidence_pred=np.zeros((7,90),dtype=int);probabilities=np.zeros((7,90,3))
    contributions=np.zeros((90,60,3));offset_values=np.zeros(90)
    for fold,(train,test) in enumerate(blocks):
        inner=list(blocked(train,3));scores=[]
        for deadline in DEADLINES:
            bins=int(round((deadline-.25)/.25))
            for budget in BUDGETS:
                correct=0
                for itrain,itest in inner:
                    model,_,x=fit_model(features,labels,itrain,bins,budget)
                    correct+=int(np.sum(model.predict(x[itest])==labels[itest]))
                scores.append((correct,deadline,budget))
        # Higher correct count; deterministic speed/channel tie-break.
        correct,deadline,budget=sorted(scores,key=lambda t:(-t[0],t[1],t[2]))[0]
        model,selected,x=fit_model(features,labels,train,int(round((deadline-.25)/.25)),budget)
        nested[test]=model.predict(x[test])
        row={"fold":fold+1,"train_trials":(train+1).tolist(),"test_trials":(test+1).tolist(),
             "chosen_deadline":deadline,"chosen_budget":budget,"inner_correct":correct,"inner_total":len(train),
             "selected_channels":(selected+1).tolist(),"temperatures":{}}
        arrays[f"fold_{fold}_train"]=train;arrays[f"fold_{fold}_test"]=test
        for di,d in enumerate(DEADLINES):
            bins=int(round((d-.25)/.25))
            pred,p,t,selected,model,x=calibrated_fold(features,labels,train,test,bins)
            confidence_pred[di,test]=pred;probabilities[di,test]=p;row["temperatures"][str(d)]=t
            expected=arrays[f"{d:.2f}s_60ch_blocked"][0,test]
            np.testing.assert_array_equal(pred,expected)
            if d==1.:
                mean=x[train].mean(axis=0);w=model.coef_[1]-model.coef_[0]
                contributions[test]=((x[test]-mean)*w).reshape(len(test),60,3)
                offset_values[test]=model.intercept_[1]-model.intercept_[0]+mean@w
                np.testing.assert_allclose(contributions[test].sum(axis=(1,2))+offset_values[test],
                    model.decision_function(x[test])[:,1]-model.decision_function(x[test])[:,0],atol=1e-8)
        metrics["folds"].append(row)
        print(f"  outer block {fold+1}: nested selected {deadline}s / {budget} channels",flush=True)
    metrics["nested_joint"]=classification_summary(labels,nested)
    metrics["confidence"]={str(d):classification_summary(labels,confidence_pred[i],probabilities[i]) for i,d in enumerate(DEADLINES)}
    metrics["adaptive"],adaptive_pred,adaptive_times=adaptive(labels,confidence_pred,probabilities,DEADLINES)
    arrays.update(nested_predictions=nested,calibrated_predictions=confidence_pred,calibrated_probabilities=probabilities,
                  adaptive_predictions=adaptive_pred,adaptive_decision_seconds=adaptive_times,
                  default_contributions=contributions,contribution_offsets=offset_values)
    # Attach calibrated probabilities to matching chronological fixed-model rows only.
    for row in records:
        if row["protocol"]=="blocked" and row["feature_channels"]==60:
            p=probabilities[DEADLINES.index(row["deadline"]),row["trial"]-1]
            row.update(confidence=float(p.max()),prob_rock=float(p[0]),prob_scissors=float(p[1]),prob_paper=float(p[2]))
    for i in range(90):
        records.append({"condition":"nested_joint","protocol":"blocked","repeat":1,"fold":i//18+1,"trial":i+1,
            "cue":NAMES[int(labels[i])],"cue_code":int(labels[i]),"prediction":NAMES[int(nested[i])],
            "prediction_code":int(nested[i]),"correct":bool(nested[i]==labels[i]),
            "deadline":metrics["folds"][i//18]["chosen_deadline"],"feature_channels":metrics["folds"][i//18]["chosen_budget"]})
        records.append({"condition":"adaptive_0.90","protocol":"blocked","repeat":1,"fold":i//18+1,"trial":i+1,
            "cue":NAMES[int(labels[i])],"cue_code":int(labels[i]),"prediction":NAMES.get(int(adaptive_pred[i]),"Abstain"),
            "prediction_code":int(adaptive_pred[i]),"correct":bool(adaptive_pred[i]==labels[i]),
            "deadline":float(adaptive_times[i]),"feature_channels":60,"accepted":bool(adaptive_pred[i]!=0)})
    print("4b/8 Central whitening evidence and G25 follow-up reproduction",flush=True)
    from .whitening import evidence
    from .partner_reproduction import reproduce
    metrics["whitening"]=evidence(cleaned,onsets,labels,features,hg,random_splits,blocks,arrays,records)
    del hg,cleaned
    metrics["whitening"]["partner_reproduction"]=reproduce(data,arrays,records)
    source_result=metrics["whitening"]["partner_reproduction"]
    metrics["whitening"]["reported_followup"]["status"]=("Reproduced from the supplied G25 scripts" if all(source_result["matches_reported_followup"].values()) else "Source scripts reproduced; see explicit numerical differences")
    metrics["whitening"]["reported_followup"]["source"]="ECoG_G25_code/4_partner_checks/partner_checks.py with 3_oct5_analyses/common.py"
    metrics["history"]["status"]="Secondary diagnostic; no longer the central contribution"
    print("5/8 Past-only evaluation and chance sanity check",flush=True)
    for deadline in [1.,2.25]:
        all_y=[];all_pred=[];fold_rows=[]
        for start in [45,60,75]:
            train,test=np.arange(start-1),np.arange(start,start+15)
            model,_,x=fit_model(features,labels,train,int(round((deadline-.25)/.25)),60)
            pred=model.predict(x[test]);all_y.extend(labels[test]);all_pred.extend(pred)
            for trial,prediction in zip(test,pred):
                records.append({"condition":f"d{deadline:.2f}_c60","protocol":"forward","repeat":1,"fold":len(fold_rows)+1,
                    "trial":int(trial+1),"cue":NAMES[int(labels[trial])],"cue_code":int(labels[trial]),
                    "prediction":NAMES[int(prediction)],"prediction_code":int(prediction),
                    "correct":bool(prediction==labels[trial]),"deadline":deadline,"feature_channels":60})
            fold_rows.append({"train_trials":(train+1).tolist(),"test_trials":(test+1).tolist(),"correct":int(np.sum(pred==labels[test]))})
        metrics["forward"].append({"deadline":deadline,"summary":classification_summary(all_y,all_pred),"folds":fold_rows})
    null=[];rng=np.random.default_rng(0)
    for _ in range(20):
        yy=rng.permutation(labels);summary,_,_,_=evaluate_grid(features,yy,blocks,3,60,1)
        null.append(summary["accuracy_percent"])
    metrics["shuffled_blocked_mean_percent"]=float(np.mean(null));arrays["shuffled_percent"]=np.array(null)
    print("6/8 Reduced acquisition and persistent channel-loss preprocessing",flush=True)
    for budget in [10,20,40]:
        pred=np.zeros(90,dtype=int);selection=[];acquisition_prob=np.zeros((90,3));temperatures=[]
        for fold,(train,test) in enumerate(blocks):
            kept=np.sort(rank_channels(features,labels,train,3)[:budget])
            subset,_,_=build_features(raw[kept],onsets,channel_ids=kept)
            model,_,x=fit_model(subset,labels,train,3,budget);pred[test]=model.predict(x[test])
            positions={int(t):i for i,t in enumerate(train)};inner_logits=np.empty((len(train),3))
            for itrain,itest in blocked(train,3):
                inner_kept=np.sort(rank_channels(features,labels,itrain,3)[:budget])
                inner_subset,_,_=build_features(raw[inner_kept],onsets,channel_ids=inner_kept)
                imodel,_,ix=fit_model(inner_subset,labels,itrain,3,budget)
                inner_logits[[positions[int(t)] for t in itest]]=imodel.decision_function(ix[itest])
            temp=temperature(inner_logits,labels[train]);temperatures.append(temp)
            acquisition_prob[test]=softmax(model.decision_function(x[test])/temp,axis=1)
            selection.append((kept+1).tolist())
        metrics["acquisition"].append({"channels":budget,"summary":classification_summary(labels,pred,acquisition_prob),"selected_per_fold":selection,"temperatures":temperatures})
        arrays[f"acquisition_{budget}_predictions"]=pred
        arrays[f"acquisition_{budget}_probabilities"]=acquisition_prob
        print(f"  {budget}-channel acquisition complete",flush=True)
    for count in [1,3,6]:
        for seed in range(5):
            dropped=np.sort(np.random.default_rng(seed).choice(60,count,replace=False));kept=np.setdiff1d(np.arange(60),dropped)
            altered,_,_=build_features(raw[kept],onsets,channel_ids=kept)
            summary,pred,_,_=evaluate_grid(altered,labels,blocks,3,len(kept),1)
            metrics["channel_loss"].append({"removed":count,"seed":seed,"dropped_channels":(dropped+1).tolist(),"summary":summary})
            arrays[f"loss_{count}_{seed}_predictions"]=pred
        print(f"  loss of {count} channels complete (five patterns)",flush=True)
    print("7/8 Referencing comparison and signal redundancy",flush=True)
    local,_,_=build_features(raw,onsets,montage="local")
    for name,f in [("common_average",features),("local_neighbors",local)]:
        summary,pred,_,_=evaluate_grid(f,labels,blocks,3,60,1)
        metrics["montage"][name]=summary;arrays[name+"_predictions"]=pred
    # Correlations in training data only, averaged over outer folds; association, not connectivity.
    arrays["training_channel_correlations"]=np.mean([np.corrcoef(features[train,:,:3].mean(axis=2),rowvar=False) for train,_ in blocks],axis=0)
    default_model,_,default_x=fit_model(features,labels,blocks[0][0],3,60)
    infer=[]
    for _ in range(100):
        begin=time.perf_counter();default_model.predict(default_x[:1]);infer.append((time.perf_counter()-begin)*1000)
    metrics["processing_timing"]["one_trial_classifier_p50_ms"]=float(np.median(infer))
    print("8/8 Writing verified artifacts, scientific figures and presentation notes",flush=True)
    pd.DataFrame(records).to_csv(output/"predictions.csv",index=False)
    np.savez_compressed(output/"model_arrays.npz",**arrays)
    versions={p:importlib.metadata.version(p) for p in ["numpy","scipy","scikit-learn","pandas","matplotlib","streamlit","plotly"]}
    metrics["provenance"]={"created_at_utc":datetime.now(timezone.utc).isoformat(),"data_sha256":sha256(data_path),
        "protected_hashes":original_hashes,"python":platform.python_version(),"versions":versions,"blas_threads":1,
        "elapsed_seconds":time.perf_counter()-started,"code_sha256":code_hash()}
    metrics["limitations"]=metrics["whitening"]["limitations"]+["One previously explored recording and participant; all new comparisons are exploratory.",
        "Repeated CV predictions reuse 90 trials; they are not independent participants or new trials.",
        "Cue and visual stimulus identify the requested gesture; motor and stimulus contributions are not fully separable.",
        "Glove-defined movement onset is an estimate; no detectable displacement is not proof of no motor activity.",
        "Calibration and adaptive decisions are evaluated within one recording, not prospective deployment.",
        "Channel loss is persistent loss with retraining, not unexpected dropout handled by a frozen model.",
        "Local referencing and channel correlations do not locate exact sources or demonstrate causal connectivity.",
        "Lower high-gamma power does not establish reduced effort; failed decoding does not establish forgetting."]
    for path,digest in original_hashes.items():
        if sha256(path)!=digest:
            raise AssertionError("Protected source changed")
    metrics["protected_sources_unchanged"]=True
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    from .report import write_report
    write_report(output,metrics,table)
    manifest={"schema_version":1,"data_path":str(data_path),"data_sha256":sha256(data_path),"code_sha256":code_hash(),
              "protected_source_hashes":{str(Path(p).relative_to(ROOT)):digest for p,digest in original_hashes.items()
                                         if Path(p)!=data_path and Path(p).is_relative_to(ROOT)},
              "config_sha256":__import__("hashlib").sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest(),
              "files":{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="manifest.json"}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Completed suite: {output}\nWhitening: matched ablation and G25 source reproduction exported",flush=True)
    return metrics


def main_run(data_path,output):
    with threadpool_limits(limits=1):
        return run(data_path,output)
