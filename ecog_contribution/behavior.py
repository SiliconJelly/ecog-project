"""Brain-hand case study: glove is explanatory evidence, never a decoder feature."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import signal
from .signal import FS, band_filter
from .validation import blocked

NAMES={1:"Rock",2:"Scissors",3:"Paper"}


def sustained_onset(displacement, baseline_displacement, multiplier=3):
    median=np.median(baseline_displacement)
    mad=np.median(np.abs(baseline_displacement-median))*1.4826
    threshold=max(.02,median+multiplier*mad)
    above=displacement>threshold
    count=np.convolve(above.astype(int),np.ones(60,dtype=int),mode="valid")
    starts=np.flatnonzero(count==60)
    return (float(starts[0]/FS) if len(starts) else np.nan),float(threshold)


def log_power(x):
    return 10*np.log10(np.maximum(np.mean(x*x,axis=1),np.finfo(float).tiny))


def trial_table(glove, onsets, offsets, labels):
    # A trailing 50 ms mean makes peak-speed estimates less sensitive to quantization.
    smooth=signal.lfilter(np.ones(60)/60,[1.],glove,axis=1)
    speed=np.linalg.norm(np.diff(smooth,axis=1,prepend=smooth[:,:1])*FS,axis=0)/np.sqrt(5)
    rows=[]
    for i,o in enumerate(onsets):
        base=np.median(glove[:,o-FS:o],axis=1)
        rest=np.sqrt(np.mean((glove[:,o-FS:o]-base[:,None])**2,axis=0))
        disp=np.sqrt(np.mean((glove[:,o:o+2*FS]-base[:,None])**2,axis=0))
        onset,threshold=sustained_onset(disp,rest)
        row={"trial":i+1,"gesture":NAMES[int(labels[i])],"label":int(labels[i]),"cue_seconds":o/FS,
             "cue_duration_seconds":(offsets[i]-o)/FS,"starting_posture":float(base.mean()),
             "movement_amount":float(np.percentile(disp,95)),"movement_onset_seconds":onset,
             "movement_detected":bool(np.isfinite(onset)),"movement_threshold":threshold,
             "peak_finger_speed":float(speed[o:o+2*FS].max()),
             "rest_seconds":float((o-offsets[i-1])/FS) if i else np.nan}
        for mult in [2,4]:
            row[f"movement_onset_{mult}mad_seconds"]=sustained_onset(disp,rest,mult)[0]
        row.update({f"starting_finger_{j+1}":float(v) for j,v in enumerate(base)})
        rows.append(row)
    table=pd.DataFrame(rows)
    table["previous_gesture"]=table.gesture.shift(1)
    table["previous_movement_amount"]=table.movement_amount.shift(1)
    return table


def neuroscience(cleaned, glove, onsets, offsets, labels, ar_end):
    table=trial_table(glove,onsets,offsets,labels)
    bands={"high_gamma":band_filter(cleaned,[50,300]),"beta":band_filter(cleaned,[13,30]),
           "erp":band_filter(cleaned,[1,30])}
    relative_times=np.arange(-FS,3*FS,6)/FS
    traces={"times":relative_times}
    for name,x in bands.items():
        common=log_power(x[:,2*FS:ar_end]).mean() if name!="erp" else 0.
        pieces=[]
        for i,o in enumerate(onsets):
            # Science power is unwhitened; absolute physical scale is retained.
            pre=log_power(x[:,o-FS:o]).mean() if name!="erp" else 0.
            if name!="erp":
                absolute=log_power(x[:,o+int(.5*FS):o+2*FS]).mean()
                table.loc[i,f"{name}_absolute_db"]=absolute
                table.loc[i,f"{name}_common_db"]=absolute-common
                table.loc[i,f"{name}_trial_relative_db"]=absolute-pre
                table.loc[i,f"pre_cue_{name}_db"]=pre
                # Causal trailing 50 ms power display; no centered/future smoothing.
                segment=x[:,o-FS-60:o+3*FS]
                power=signal.lfilter(np.ones(60)/60,[1.],segment*segment,axis=1)[:,60:]
                piece=10*np.log10(np.maximum(power[:,::6],np.finfo(float).tiny))
            else:
                piece=x[:,o-FS:o+3*FS:6]
            pieces.append(piece.astype(np.float32))
        traces[name]=np.stack(pieces)
        if name!="erp":
            traces[name+"_baseline"]=table[f"pre_cue_{name}_db"].to_numpy()
            traces[name+"_common"]=np.array(common)
    traces["glove"]=np.stack([glove[:,o-FS:o+3*FS:6] for o in onsets]).astype(np.float32)
    # One extended scientific panel includes the next cue and the intervening rest.
    rock=table[(table.gesture=="Rock") & (table.index<len(table)-1)]
    representative=int((rock.movement_amount-rock.movement_amount.median()).abs().idxmin())
    o=onsets[representative];end=min(onsets[representative+1]+int(.3*FS),cleaned.shape[1])
    segment=bands["high_gamma"][:,o-FS-60:end]
    power=signal.lfilter(np.ones(60)/60,[1.],segment*segment,axis=1)[:,60:]
    traces["annotated_times"]=np.arange(o-FS,end,6)/FS-o/FS
    traces["annotated_high_gamma"]=10*np.log10(np.maximum(power[:,::6],np.finfo(float).tiny)).mean(axis=0)
    traces["annotated_glove"]=glove[:,o-FS:end:6]
    traces["annotated_trial_index"]=np.array(representative)
    traces["annotated_next_cue_seconds"]=np.array((onsets[representative+1]-o)/FS)
    onset=table.movement_onset_seconds.to_numpy()
    movement_times=np.arange(-.5,1.51,.005)
    for name in ["high_gamma","beta","erp"]:
        aligned=np.full((len(table),len(movement_times)),np.nan,dtype=np.float32)
        for i,t in enumerate(onset):
            if np.isfinite(t):
                aligned[i]=np.interp(movement_times+t,relative_times,traces[name][i].mean(axis=0),left=np.nan,right=np.nan)
        traces[name+"_movement_aligned"]=aligned
    traces["movement_times"]=movement_times
    # Flags are descriptive within-gesture disagreements, not disease or label corrections.
    table["brain_hand_disagreement_z"]=0.
    for gesture,group in table.groupby("gesture"):
        for col in ["movement_amount","high_gamma_absolute_db"]:
            values=group[col].to_numpy();scale=np.std(values) or 1.
            z=(values-values.mean())/scale
            if col=="movement_amount": zg=z
            else: zn=z
        table.loc[group.index,"brain_hand_disagreement_z"]=zn-zg
    return table,traces


def design(table, history=False, beta=False):
    # Six prespecified base covariates, plus intercept. Peak speed/onset remain diagnostics.
    x=np.column_stack([(table.label==2).astype(float),(table.label==3).astype(float),
                       table.starting_posture,table.movement_amount,table.trial,table.rest_seconds])
    names=["Scissors vs Rock","Paper vs Rock","Starting posture","Current movement amount","Trial number","Rest duration"]
    if history:
        col="pre_cue_beta_db" if beta else "previous_movement_amount"
        x=np.column_stack([x,table[col]])
        names.append(col)
    return x,names


def adjusted_effect(x,y,seed=0,bootstraps=1000,block_length=5):
    # Fixed-scale coefficient: dB per SD of preceding movement. Blocks retain serial structure.
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1.
    xx=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    coef=np.linalg.lstsq(xx,y,rcond=None)[0]
    rng=np.random.default_rng(seed);values=[];n=len(x)
    for _ in range(bootstraps):
        starts=rng.integers(0,n-block_length+1,size=int(np.ceil(n/block_length)))
        ids=np.concatenate([np.arange(s,s+block_length) for s in starts])[:n]
        if np.linalg.matrix_rank(xx[ids])==xx.shape[1]:
            values.append(np.linalg.lstsq(xx[ids],y[ids],rcond=None)[0][-1])
    if len(values)<bootstraps*.8:
        raise ValueError("Too few full-rank history bootstrap samples")
    residual=y-xx@coef
    return {"effect_db_per_history_sd":float(coef[-1]),"bootstrap_95_interval":np.percentile(values,[2.5,97.5]).tolist(),
            "bootstrap_valid_samples":len(values),"bootstrap_block_trials":block_length,
            "design_rank":int(np.linalg.matrix_rank(xx)),"design_columns":xx.shape[1],
            "design_condition_number":float(np.linalg.cond(xx)),"in_sample_r2":float(r2_score(y,xx@coef))}, residual,xx@coef


def history_analysis(table):
    # First trial has no observed predecessor. Retain all other trials, including unusual ones.
    transition=table.iloc[1:].copy().reset_index(drop=True)
    x,names=design(transition);xh,history_names=design(transition,history=True)
    splits=list(blocked(np.arange(len(transition)),5))
    results={"transition_count":len(transition),"primary_history":"previous_movement_amount",
             "base_covariates":names,"effect_units":"dB per one SD of preceding glove movement amount",
             "uncertainty":"95% moving-block bootstrap interval within this recording; not patient-population inference",
             "baseline_variants":{},"folds":[{"train_trials":(tr+2).tolist(),"test_trials":(te+2).tolist()} for tr,te in splits]}
    arrays={}
    for variant,column in [("absolute","high_gamma_absolute_db"),("common_reference","high_gamma_common_db"),
                           ("trial_baseline","high_gamma_trial_relative_db")]:
        y=transition[column].to_numpy();base=np.empty(len(y));extended=np.empty(len(y))
        for train,test in splits:
            for xx,pred in [(x,base),(xh,extended)]:
                model=make_pipeline(StandardScaler(),LinearRegression())
                model.fit(xx[train],y[train]);pred[test]=model.predict(xx[test])
        effect,residual,fitted=adjusted_effect(xh,y)
        effect.update(base_chronological_mse=float(mean_squared_error(y,base)),
                      history_chronological_mse=float(mean_squared_error(y,extended)),
                      mse_improvement=float(mean_squared_error(y,base)-mean_squared_error(y,extended)),
                      base_chronological_r2=float(r2_score(y,base)),history_chronological_r2=float(r2_score(y,extended)))
        # Unadjusted association conditions on gesture only, for comparison with controlled model.
        unadjusted=np.column_stack([x[:,:2],transition.previous_movement_amount])
        effect["gesture_only_history_effect"]=adjusted_effect(unadjusted,y)[0]
        lo,hi=effect["bootstrap_95_interval"]
        ulo,uhi=effect["gesture_only_history_effect"]["bootstrap_95_interval"]
        if lo*hi>0 and effect["mse_improvement"]>0:
            effect["interpretation"]="History adds predictive information in this within-recording comparison."
        elif ulo*uhi>0 and lo*hi<=0 and effect["mse_improvement"]<=0:
            effect["interpretation"]="The association is attenuated after hand-behavior adjustment; compatible with behavioral confounding."
        else:
            effect["interpretation"]="This recording does not distinguish the explanations reliably."
        results["baseline_variants"][variant]=effect
        arrays[variant+"_observed"]=y;arrays[variant+"_base_oof"]=base;arrays[variant+"_history_oof"]=extended
        arrays[variant+"_residual"]=residual
    # Beta association is separate from the small history model; both are secondary diagnostics.
    xb,_=design(transition,history=True,beta=True)
    results["secondary_pre_cue_beta_effect"]=adjusted_effect(xb,transition.high_gamma_absolute_db.to_numpy())[0]
    results["limitations"]=["One previously explored participant recording; associations are not causal.",
        "Current behavior is a potential mediator as well as a confound; adjustment changes the question.",
        "Bootstrap interval assumes short-range dependence captured by five-trial blocks.",
        "Shared reference subtracts a constant; with an intercept its adjusted effects equal absolute-power effects.",
        "Scientific outcome pools all 60 channels without selecting electrodes by this effect.",
        "Primary outcome is mean channel log power from 0.5 to 2 seconds after cue, before whitening."]
    return results,arrays
