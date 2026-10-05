"""Read-only local dashboard. Model fitting is deliberately absent from this app."""
from pathlib import Path
import sys
import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ecog_contribution.artifacts import DEFAULT_OUTPUT,validate_cache,code_hash,cache_fingerprint

COLORS={"Rock":"#60a5fa","Scissors":"#fb923c","Paper":"#34d399"}
DEADLINES=[.5,.75,1.,1.25,1.5,2.,2.25]
st.set_page_config(page_title="ECoG | Whitening Under Test",page_icon="🧠",layout="wide")
st.markdown("""<style>
.stApp{background:#0b1020;color:#e5ecf6}section[data-testid="stSidebar"]{background:#111a2d}
h1,h2,h3{letter-spacing:-.025em}div[data-testid="stMetric"]{background:#121e33;border:1px solid #263651;padding:16px;border-radius:12px}
div[data-testid="stMetricValue"]{color:#67e8f9}div[data-testid="stMetricLabel"] p{white-space:normal!important}a{color:#7dd3fc}button:focus-visible,input:focus-visible{outline:3px solid #67e8f9!important}
div[data-testid="stMetricValue"]>div{white-space:normal!important;overflow:visible!important;text-overflow:clip!important}
.study-note{border-left:3px solid #38bdf8;padding:12px 16px;background:#121e33;border-radius:4px;margin-bottom:16px}
</style>""",unsafe_allow_html=True)


@st.cache_data(show_spinner="Verifying local results…")
def load_cache(manifest_stamp,analysis_code):
    ok,message=validate_cache()
    if not ok:
        return None,message
    metrics=json.loads((DEFAULT_OUTPUT/"metrics.json").read_text())
    with np.load(DEFAULT_OUTPUT/"model_arrays.npz") as z:
        models={k:z[k] for k in z.files}
    with np.load(DEFAULT_OUTPUT/"display_traces.npz") as z:
        traces={k:z[k] for k in z.files}
    table=pd.read_csv(DEFAULT_OUTPUT/"trial_table.csv")
    with np.load(DEFAULT_OUTPUT/"history_arrays.npz") as z:
        history={k:z[k] for k in z.files}
    return (metrics,models,traces,table,history),message


def chart(fig,key=None):
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0b1020",plot_bgcolor="#101a2e",
                      font_color="#dce7f6",margin=dict(l=35,r=20,t=50,b=40),legend=dict(orientation="h",y=-.22))
    st.plotly_chart(fig,width="stretch",key=key,config={"displaylogo":False})


def note(text):
    st.markdown(f'<div class="study-note">{text}</div>',unsafe_allow_html=True)


def confusion(summary):
    cm=np.asarray(summary["confusion_matrix"])
    fig=go.Figure(go.Heatmap(z=cm,x=list(COLORS),y=list(COLORS),text=cm,texttemplate="%{text}",colorscale="Blues"))
    fig.update_layout(title="Requested cue → held-out prediction",xaxis_title="Predicted gesture",yaxis_title="Requested gesture")
    chart(fig)


stamp=(DEFAULT_OUTPUT/"manifest.json").stat().st_mtime_ns if (DEFAULT_OUTPUT/"manifest.json").exists() else 0
loaded,message=load_cache((stamp,cache_fingerprint()),code_hash())
st.sidebar.title("ECoG Signal Lab")
st.sidebar.caption("WHITEN · DECODE · VERIFY")
page=st.sidebar.radio("Perspective",["Overview","Whitening evidence","Reliability","Trial replay","Signal diagnostics","Spatial signals"])
st.sidebar.caption("90 trials · 60 electrodes · 1 recording")
st.sidebar.caption("Local research replay · exploratory")
if loaded is None:
    st.title("Prepare the contribution suite")
    st.warning(message)
    st.code("source .venv/bin/activate\npython -m ecog_contribution run\nstreamlit run dashboard/app.py",language="bash")
    st.stop()
metrics,models,traces,table,history=loaded
st.sidebar.success("Verified cached results")
with st.sidebar.expander("Evidence and provenance"):
    st.write(metrics["participant_metadata"])
    st.caption("One participant; whitening supports gesture decoding, not a clinical or intention claim.")
    st.caption("Dataset SHA-256: "+metrics["provenance"]["data_sha256"][:16]+"…")
    st.caption("Generated: "+metrics["provenance"]["created_at_utc"])
    st.caption("All interactions read results; no model training.")

if page=="Overview":
    st.title("Whitening Under Test")
    st.caption("Fast, reliable ECoG gesture decoding · Team G25 follow-up")
    note("Whitening is the central decoder path: compare it with the same pipeline without whitening, inspect paired mistakes, and test calibration using only earlier signal samples.")
    source=metrics["whitening"]["partner_reproduction"];causal=source["causal"]
    fixed=metrics["whitening"]["deadlines"]["2.25"]["random"]["paired"]
    cols=st.columns(2)+st.columns(2)
    cols[0].metric("Established submission", "96.9%",help="Original whole-record whitening; repeated 10-fold × 10 CV. Preserved separately.")
    cols[1].metric("G25 causal-filter follow-up",f"{causal['accuracy_without_percent']:.1f}% → {causal['accuracy_with_percent']:.1f}%")
    cols[2].metric("Pre-task fit · repeated CV",f"{fixed['accuracy_without_percent']:.1f}% → {fixed['accuracy_with_percent']:.1f}%")
    default=metrics["grid"]["1.00s_60ch"]
    cols[3].metric("1 s whitened · purged blocks",f"{default['blocked']['correct']}/90")
    st.subheader("The follow-up result, reproduced from the team scripts")
    st.write(f"Whitening increased mean accuracy by **{causal['gain_percentage_points']:.2f} percentage points**. Across 20 matching CV seeds, **{causal['trials_helped']} trials improved** their correctness fraction and **{causal['trials_hurt']} worsened**. The original one-sided sign-flip test gives **p = {causal['p_signflip_one_sided']:.4f}**.")
    st.caption("90 distinct trials, 1,800 repeated prediction exposures. This p-value is a one-sided Monte Carlo sign-flip on per-trial average correctness, not a test treating every CV prediction as independent.")
    st.info("The G25 headline uses causal filter application with whole-record AR fitting, so it is retrospective. Our pre-task-fit path and second-half holdout test calibration that ends before test samples arrive. All new results remain exploratory.")
    half=causal["first_half_holdout"]
    st.write(f"With the filter fitted through the first 45 trials, the causal held-out last 45 score is **{half['without']['correct']}/45 without whitening → {half['first_half']['correct']}/45 with whitening**. The Whitening evidence view gives the calibration cutoff and the stricter purged comparison.")
    st.subheader("Speed and reliability support the whitening path")
    protocol=st.selectbox("Evaluation protocol",["Purged chronological blocks","Repeated random trial CV"])
    key="blocked" if protocol.startswith("Purged") else "random"
    rows=[{"Window end (s)":v["deadline"],"Accuracy (%)":v[key]["accuracy_percent"],"Feature channels":str(v["budget"])} for v in metrics["grid"].values()]
    chart(px.line(pd.DataFrame(rows),x="Window end (s)",y="Accuracy (%)",color="Feature channels",markers=True,title="Pre-task-whitened speed × feature-channel comparison"))
    st.caption("Whitening-on/off ablations use fixed 60 channels and matching deadlines. This supporting grid changes the feature budget; acquisition reductions are evaluated separately.")
    st.subheader("The claim we withdrew")
    st.write("Reliable separation of a still hand from cued Paper was not established in the reported follow-up. Gesture-decoding accuracy does not establish movement-independent intention. The glove, movement history and unusual-trial plots remain secondary diagnostics.")
    with st.expander("Processing time versus waiting time"):
        st.json(metrics["processing_timing"])
        st.caption("Compute time, sample waiting time and glove reaction time are different measurements; recorded replay is not a deployed live BCI.")
    st.download_button("Download contribution report",(DEFAULT_OUTPUT/"CONTRIBUTION_REPORT.md").read_bytes(),file_name="ecog_whitening_contribution_report.md")

elif page=="Whitening evidence":
    st.title("The whitening comparison")
    note("Only the whitening step changes inside each comparison. Protocols and calibration periods stay visible so a better accuracy cannot hide future-signal fitting.")
    evidence=metrics["whitening"];source=evidence["partner_reproduction"]
    comparison=st.radio("Whitening comparison",["G25 reported follow-up","Past-only pre-task calibration","First-half calibration"],horizontal=True)
    if comparison=="G25 reported follow-up":
        v=source["causal"]
        cols=st.columns(3)
        cols[0].metric("Without whitening",f"{v['accuracy_without_percent']:.2f}%")
        cols[1].metric("Whole-record AR(10)",f"{v['accuracy_with_percent']:.2f}%")
        cols[2].metric("Gain",f"+{v['gain_percentage_points']:.2f} pp")
        st.write(f"**{v['trials_helped']} helped / {v['trials_hurt']} hurt**, based on each trial's fraction correct across the same 20 CV seeds. One-sided sign-flip p = **{v['p_signflip_one_sided']:.6f}**.")
        gain=models["partner_causal_trial_gain"]
        chart(go.Figure(go.Bar(x=np.arange(1,91),y=gain*100,marker_color=np.where(gain>=0,"#34d399","#fb923c"))).update_layout(title="Paired change in each trial's correctness fraction",xaxis_title="Original trial",yaxis_title="Percentage-point change across 20 seeds"))
        st.caption("Filtering is forward-only, but AR coefficients use the whole recording. This is within-recording retrospective evidence, not a prospective whole pipeline. The reproduced source uses a different CV scheme from the established 96.9%.")
        st.write("Matched source scripts:", "ECoG_G25_code/4_partner_checks/partner_checks.py and 3_oct5_analyses/common.py")
        with st.expander("Exact source reproduction and historical offline comparison"):
            st.json(source["matches_reported_followup"])
            st.json(source["zero_phase"])
    elif comparison=="Past-only pre-task calibration":
        d=st.select_slider("Ablation deadline (s)",options=DEADLINES,value=2.25)
        protocol=st.selectbox("Paired evaluation",["Purged chronological blocks","Repeated random CV"])
        key="blocked" if protocol.startswith("Purged") else "random"
        v=evidence["deadlines"][str(d)][key];pair=v["paired"];first=pair["first_repeat"]
        cols=st.columns(3)
        cols[0].metric("Without whitening",f"{pair['accuracy_without_percent']:.2f}%")
        cols[1].metric("Pre-task AR(10)",f"{pair['accuracy_with_percent']:.2f}%")
        cols[2].metric("Gain",f"{pair['gain_percentage_points']:+.2f} pp")
        lo,hi=pair["gain_block_bootstrap_95_interval"]
        st.write(f"Five-trial block-bootstrap gain interval: **[{lo:.2f}, {hi:.2f}] pp**. Calibration interval: **2.00–10.08 s**, before the first cue.")
        st.caption("For repeated CV, repeat outcomes are averaged within each original trial before block resampling. The interval is descriptive uncertainty within this one previously explored recording.")
        st.write(f"One prediction per trial in the first repeat: **{first['helped']} fixed**, **{first['hurt']} broken**, **{first['both_wrong']} wrong in both**.")
        st.caption(f"Descriptive exact two-sided McNemar p = {first['exact_mcnemar_two_sided_p']:.6f}. This is a different test and comparison from G25's one-sided sign-flip. Serial independence is not established.")
        off=models[f"whitening_off_{d:.2f}_{key}"][0];on=models[f"{d:.2f}s_60ch_{key}"][0];y=models["labels"]
        df=pd.DataFrame({"Trial":np.arange(1,91),"Cue":[list(COLORS)[x-1] for x in y],"Without":[list(COLORS)[x-1] for x in off],"Whitened":[list(COLORS)[x-1] for x in on],"Outcome":np.where((off!=y)&(on==y),"Fixed",np.where((off==y)&(on!=y),"Broken",np.where(on==y,"Both correct","Both wrong")))})
        st.dataframe(df,hide_index=True)
    else:
        causal=source["causal"];half=causal["first_half_holdout"]
        st.subheader("Reproduce the team's first-45 → last-45 check")
        st.write(f"Without whitening: **{half['without']['correct']}/45**. First-half whitening: **{half['first_half']['correct']}/45**.")
        st.caption(f"AR calibration ends at {source['half_cutoff_seconds']:.2f} s, before cue 46. Causal filtering is used. The original check has no adjacent-trial purge.")
        st.subheader("Stricter chronological acquisition-time check")
        rows=[]
        for item in evidence["second_half"]:
            for name,v in item["summaries"].items():
                rows.append({"Window end (s)":item["deadline"],"Method":name,"Correct":v["correct"],"Tested":v["total"],"Accuracy (%)":v["accuracy_percent"]})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.caption("This check divides the recording at its time midpoint (211.26 s), trains on complete earlier trials with one neighboring trial purged, and evaluates 46 later cues. Its cohort differs from the team's 45/45 split; compare only matched rows within each protocol.")
        st.write(evidence["first_half_scope"])
        with st.expander("Original expanding-window calibration caveat"):
            for text in source["limitations"]:st.write("• "+text)
    st.subheader("Does AR fitting explain the improvement uniquely?")
    sensitivity=[{"Variant":"AR(10) whole-record","Accuracy (%)":source["causal"]["accuracy_with_percent"]}]
    sensitivity.extend({"Variant":v["variant"],"Accuracy (%)":v["summary"]["accuracy_percent"]} for v in source["sensitivity"])
    st.dataframe(pd.DataFrame(sensitivity),hide_index=True)
    st.caption("AR-order and fixed first-difference checks use the same G25 causal 20-seed protocol. First difference ties AR(10) at 97.3%; fitted AR coefficients are not uniquely necessary for this improvement. These are retrospective mechanism and sensitivity controls.")
    channel=st.selectbox("Spectrum illustration electrode",list(range(1,61)),index=25)
    fig=go.Figure()
    for key,name,color in [("whitening_spectrum_without","Without whitening","#94a3b8"),("whitening_spectrum_first_half","First-half AR(10)","#67e8f9")]:
        fig.add_trace(go.Scatter(x=models["whitening_spectrum_hz"],y=10*np.log10(np.maximum(models[key][channel-1],np.finfo(float).tiny)),name=name,line_color=color))
    fig.update_layout(title=f"Electrode {channel}: spectrum on later signal samples",xaxis_title="Frequency (Hz)",yaxis_title="PSD (dB / Hz)");chart(fig)
    st.caption(evidence["spectrum_scope"]+" Frequency balance is a diagnostic; flatter spectra do not by themselves prove better decoding.")

elif page=="Reliability":
    st.title("Reliability across time and uncertainty")
    d=st.select_slider("Decision window end (s)",options=DEADLINES,value=1.)
    summary=metrics["confidence"][str(d)]
    cols=st.columns(4)
    cols[0].metric("Held-out accuracy",f"{summary['accuracy_percent']:.1f}%")
    cols[1].metric("Log loss",f"{summary['log_loss']:.3f}")
    cols[2].metric("Multiclass Brier",f"{summary['multiclass_brier']:.3f}")
    cols[3].metric("Calibration error",f"{summary['ece']:.3f}")
    note("Temperatures are fitted only to inner out-of-fold training scores. These probabilities are evaluated on outer chronological test blocks; they are not guaranteed deployment confidence.")
    left,right=st.columns(2)
    with left:
        bins=summary["calibration_bins"]
        fig=go.Figure(go.Scatter(x=[b["confidence"] for b in bins],y=[b["accuracy"] for b in bins],mode="lines+markers",text=[f"n={b['count']}" for b in bins],name="Observed"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash"),name="Ideal"))
        fig.update_layout(title="Confidence versus observed correctness",xaxis_title="Confidence",yaxis_title="Accuracy",xaxis_range=[0,1],yaxis_range=[0,1]);chart(fig)
    with right:
        curve=pd.DataFrame(summary["coverage_curve"])
        chart(px.line(curve,x="coverage_percent",y="accuracy_percent",markers=True,hover_data=["threshold","accepted"],title="Accuracy versus accepted-trial coverage"))
    a=metrics["adaptive"]
    st.subheader("Decide, wait, or abstain · fixed confidence ≥ 0.90")
    cols=st.columns(4)
    cols[0].metric("Accepted",f"{a['accepted']}/90");cols[1].metric("Abstained",str(a["abstained"]))
    cols[2].metric("Accepted accuracy",f"{a['accuracy_accepted_percent']:.1f}%" if a["accuracy_accepted_percent"] is not None else "No accepted trials")
    cols[3].metric("Mean accepted deadline",f"{a['mean_decision_seconds']:.2f} s" if a["mean_decision_seconds"] is not None else "—")
    st.caption("First eligible deadline wins. Predictions below threshold at 2.25 s abstain; abstentions are never counted as correct.")
    confusion(summary)
    st.subheader("Only earlier trials train the forward models")
    st.dataframe(pd.DataFrame([{"Window end":v["deadline"],"Correct":v["summary"]["correct"],"Tested":v["summary"]["total"],"Accuracy (%)":v["summary"]["accuracy_percent"]} for v in metrics["forward"]]),hide_index=True)
    st.caption("Expanding windows can train on previously tested trials once they become historical. This is not one permanently untouched test set.")
    st.subheader("Nested speed / channel selection")
    st.write(f"Outer held-out result: **{metrics['nested_joint']['correct']}/90**.")
    st.dataframe(pd.DataFrame([{k:r[k] for k in ["fold","chosen_deadline","chosen_budget","inner_correct","inner_total"]} for r in metrics["folds"]]),hide_index=True)
    st.subheader("Persistent channel loss, with retraining")
    loss=pd.DataFrame([{"Removed channels":str(v["removed"]),"Accuracy (%)":v["summary"]["accuracy_percent"],"Pattern":v["seed"]} for v in metrics["channel_loss"]])
    chart(px.strip(loss,x="Removed channels",y="Accuracy (%)",hover_data=["Pattern"],title="Five fixed channel-loss patterns at each severity"))
    st.caption("Removed channels do not enter CAR, AR fitting or the classifier. Models are retrained; this does not test unexpected dropout of a frozen model.")

elif page=="Trial replay":
    st.title("Recorded-data replay")
    note("Purged chronological held-out predictions · fixed 60-channel model · pre-task calibration. The requested cue provides timing and evaluation labels; the glove never enters the decoder.")
    c1,c2,c3=st.columns([2,1,2])
    decoder=st.radio("Replay decoder",["Pre-task AR-whitened","Without whitening"],horizontal=True)
    with c1: d=st.select_slider("Window end (s)",options=DEADLINES,value=1.,key="replay_deadline")
    di=DEADLINES.index(d)
    prefix="without_whitening_" if decoder=="Without whitening" else ""
    pred=models[prefix+"calibrated_predictions"][di];prob=models[prefix+"calibrated_probabilities"][di]
    with c2: errors=st.checkbox("Errors only")
    options=(np.flatnonzero(pred!=models["labels"])+1).tolist() if errors else list(range(1,91))
    if not options:
        st.info("No errors for this setting.");st.stop()
    with c3: trial=st.selectbox("Trial",options,format_func=lambda i:f"{i:02d} · {table.iloc[i-1].gesture}")
    selected=st.multiselect("Electrodes for displayed neural trace",list(range(1,61)),default=[26,37],max_selections=6)
    mode=st.radio("Neural display",["High-gamma power","1–30 Hz voltage"],horizontal=True)
    signature=(trial,d,decoder)
    if st.session_state.get("replay_signature")!=signature:
        st.session_state.replay_signature=signature;st.session_state.replay_clock=-1.;st.session_state.playing=False
    controls=st.columns(3)
    if controls[0].button("Play / pause",width="stretch"):
        st.session_state.playing=not st.session_state.get("playing",False);st.session_state.last_tick=time.monotonic()
    if controls[1].button("Restart",width="stretch"):
        st.session_state.replay_clock=-1.;st.session_state.last_tick=time.monotonic()
    speed=controls[2].selectbox("Replay speed",[.5,1.,2.],index=1)
    def seek_changed():
        st.session_state.replay_clock=st.session_state.seek_clock
        st.session_state.playing=False
    st.session_state.seek_clock=round(float(st.session_state.replay_clock)/.05)*.05
    st.slider("Seek time from cue (s)",-1.,3.,step=.05,key="seek_clock",on_change=seek_changed)
    @st.fragment(run_every=.1)
    def replay_frame():
        now=time.monotonic()
        if st.session_state.get("playing",False):
            st.session_state.replay_clock=min(3.,st.session_state.replay_clock+(now-st.session_state.get("last_tick",now))*speed)
            if st.session_state.replay_clock>=3.: st.session_state.playing=False
        st.session_state.last_tick=now
        t=st.session_state.replay_clock;i=trial-1
        columns=st.columns(4)
        columns[0].metric("Replay clock",f"{t:+.2f} s")
        columns[1].metric("Requested cue",table.iloc[i].gesture if 0<=t<2 else "Rest")
        ready=t>=d
        columns[2].metric("Held-out prediction",list(COLORS)[pred[i]-1] if ready else "Waiting for samples")
        columns[3].metric("Calibrated confidence",f"{prob[i].max():.1%}" if ready else "—")
        if ready:
            other_prefix="" if prefix else "without_whitening_"
            other=models[other_prefix+"calibrated_predictions"][di,i]
            st.caption(f"Other matched path at the same deadline: {list(COLORS)[other-1]}. Both use pre-task calibration and the same purged outer fold.")
            (st.success if pred[i]==models["labels"][i] else st.error)("Correct" if pred[i]==models["labels"][i] else "Cue / prediction mismatch retained in evaluation")
        mask=traces["times"]<=t;fig=go.Figure()
        key="high_gamma" if mode.startswith("High") else "erp"
        if not selected: st.info("Select an electrode to display neural samples.")
        for ch in selected:
            fig.add_trace(go.Scatter(x=traces["times"][mask],y=traces[key][i,ch-1,mask],name=f"Electrode {ch}"))
        fig.add_vrect(x0=0,x1=2,fillcolor="#64748b",opacity=.12,line_width=0)
        fig.add_vline(x=d,line_dash="dash",line_color="#38bdf8")
        fig.update_layout(title="Neural samples revealed up to replay time",xaxis_range=[-1,3],xaxis_title="Seconds from cue",yaxis_title="Unwhitened power (dB)" if key=="high_gamma" else "Recorded voltage units")
        chart(fig,key="replay_neural")
        fig=go.Figure()
        for j,name in enumerate(["Thumb","Index","Middle","Ring","Little"]):
            fig.add_trace(go.Scatter(x=traces["times"][mask],y=traces["glove"][i,j,mask],name=name))
        onset=table.iloc[i].movement_onset_seconds
        if np.isfinite(onset) and t>=onset: fig.add_vline(x=onset,line_dash="dot",line_color="#fb923c")
        fig.update_layout(title="Actual hand behavior",xaxis_range=[-1,3],xaxis_title="Seconds from cue",yaxis_title="Stored finger-bend values")
        chart(fig,key="replay_glove")
    replay_frame()
    st.caption("Neural display uses unwhitened signals for amplitude interpretation. Toggle the decoder to compare pre-task AR whitening with the unwhitened baseline. No playback timing is claimed as measured live BCI latency.")
    st.dataframe(table.iloc[[trial-1]],hide_index=True)

elif page=="Signal diagnostics":
    st.title("Secondary signal and hand diagnostics")
    st.info("Still hand versus cued Paper was not reliably established in the reported follow-up. These plots support inspection, not a movement-independent intention claim. Whitening is the central contribution.")
    sub=st.radio("Analysis",["Movement and rhythms","Unusual trials","Adjusted history"],horizontal=True)
    if sub=="Adjusted history":
        variant=st.selectbox("High-gamma reference",["absolute","common_reference","trial_baseline"])
        v=metrics["history"]["baseline_variants"][variant];lo,hi=v["bootstrap_95_interval"]
        cols=st.columns(3)
        cols[0].metric("Adjusted history effect",f"{v['effect_db_per_history_sd']:.3f} dB / SD")
        cols[1].metric("Within-recording interval",f"[{lo:.3f}, {hi:.3f}]")
        cols[2].metric("Held-out MSE improvement",f"{v['mse_improvement']:+.4f}")
        note(v["interpretation"])
        st.caption("Secondary history analysis: preceding movement amount. Six controls: gesture contrasts, starting posture, current movement amount, trial number, and rest duration. Peak speed and onset are recorded as diagnostics, keeping the explanatory model small.")
        fig=go.Figure()
        for j,(name,value) in enumerate(metrics["history"]["baseline_variants"].items()):
            a,b=value["bootstrap_95_interval"];fig.add_trace(go.Scatter(x=[a,b],y=[name,name],mode="lines",line_color="#67e8f9",showlegend=False))
            fig.add_trace(go.Scatter(x=[value["effect_db_per_history_sd"]],y=[name],mode="markers",marker_color="#67e8f9",showlegend=False))
        fig.add_vline(x=0,line_dash="dash");fig.update_layout(title="Baseline sensitivity: adjusted effect and block-bootstrap interval",xaxis_title="dB / preceding-movement SD");chart(fig)
        st.caption("Shared reference subtracts a constant. Its adjusted effects equal absolute power when the model includes an intercept. Trial baseline can change the association.")
        df=table.iloc[1:].copy()
        chart(px.scatter(df,x="movement_amount",y="high_gamma_absolute_db",color="gesture",color_discrete_map=COLORS,
                         hover_data=["trial","starting_posture","previous_movement_amount"],title="Within-gesture hand behavior and neural response"))
        st.dataframe(pd.DataFrame([{"Reference":name,"Base MSE":value["base_chronological_mse"],"History MSE":value["history_chronological_mse"],"Effect":value["effect_db_per_history_sd"]} for name,value in metrics["history"]["baseline_variants"].items()]),hide_index=True)
        with st.expander("Secondary beta association and model limits"):
            st.json(metrics["history"]["secondary_pre_cue_beta_effect"])
            st.write("This is a separate exploratory in-sample association, not a replacement primary hypothesis.")
            for limit in metrics["history"]["limitations"]: st.write("• "+limit)
    elif sub=="Movement and rhythms":
        alignment=st.radio("Alignment",["Cue","Estimated glove movement"],horizontal=True)
        band=st.selectbox("Signal",["high_gamma","beta","erp"])
        reference=st.selectbox("Power reference",["Absolute","Shared pre-task","Trial pre-cue"])
        fig=go.Figure()
        for name,color in COLORS.items():
            ids=np.flatnonzero((table.gesture==name).to_numpy())
            if alignment=="Cue":
                x=traces["times"];y=traces[band][ids].mean(axis=1)
            else:
                x=traces["movement_times"];y=traces[band+"_movement_aligned"][ids]
                y=y[np.isfinite(y).any(axis=1)]
                ids=ids[np.isfinite(table.iloc[ids].movement_onset_seconds.to_numpy())]
            if band!="erp":
                if reference=="Shared pre-task": y=y-float(traces[band+"_common"])
                if reference=="Trial pre-cue": y=y-traces[band+"_baseline"][ids,None]
            if len(y): fig.add_trace(go.Scatter(x=x,y=np.nanmean(y,axis=0),name=f"{name} (n={len(y)})",line_color=color))
        fig.add_vline(x=0,line_dash="dot");fig.update_layout(title="Descriptive all-electrode mean responses",xaxis_title="Seconds from alignment event",yaxis_title="Voltage units" if band=="erp" else "Power (dB)");chart(fig)
        st.caption("Movement-aligned plots omit only trials with undetected onset from that display. Classification and history analyses retain those trials. These curves do not establish motor versus visual-stimulus causality.")
        st.subheader("Onset sensitivity and starting state")
        st.dataframe(table[["trial","gesture","starting_posture","movement_amount","movement_detected","movement_onset_seconds","movement_onset_2mad_seconds","movement_onset_4mad_seconds","rest_seconds","pre_cue_beta_db"]],hide_index=True)
        chart(px.scatter(table.iloc[1:],x="rest_seconds",y="pre_cue_beta_db",color="previous_gesture",color_discrete_map=COLORS,title="Is the short rest interval a neutral baseline?",hover_data=["trial","gesture"]))
        st.caption("A 2025 MEG study motivates this overlap hypothesis; its 4–5 s duration is not transferred to this ECoG recording.")
    else:
        st.write("Previously noted Paper trials 1 and 22 remain visible alongside all other trials.")
        st.image(str(DEFAULT_OUTPUT/"figures/04_unusual_trials.png"))
        ranked=table.sort_values("brain_hand_disagreement_z",key=lambda s:s.abs(),ascending=False)
        st.dataframe(ranked[["trial","gesture","movement_amount","starting_posture","movement_onset_seconds","high_gamma_absolute_db","brain_hand_disagreement_z","quality_flag"]],hide_index=True)
        st.caption("Disagreement ranks use within-gesture standardized neural and glove measurements. They are case-inspection aids, not diagnoses or explanations of cause.")
        st.subheader("Signal-quality flags")
        st.write(metrics["quality"]["interpretation"])
        st.write("Flagged trials:",metrics["quality"]["flagged_trials_1_based"])

elif page=="Spatial signals":
    st.title("Spatial evidence and signal economy")
    note("Numbered electrode layout, not a source-localization map. Correlated electrodes and LDA contributions do not demonstrate causal brain connectivity.")
    budget=st.selectbox("Feature-selection budget",[10,20,40,60],index=1)
    values=models[f"1.00s_{budget}ch_selection_frequency"].reshape(6,10).T
    numbers=np.arange(1,61).reshape(6,10).T
    fig=go.Figure(go.Heatmap(z=values,text=numbers,colorscale="Blues",zmin=0,zmax=1,colorbar_title="Selection frequency"))
    fig.add_trace(go.Scatter(x=np.tile(np.arange(6),10),y=np.repeat(np.arange(10),6),mode="text",text=numbers.ravel(),textfont=dict(color=np.where(values.ravel()>.45,"#ffffff","#0b1020")),showlegend=False,hoverinfo="skip"))
    fig.update_layout(title="Training-only selection stability · one-second features",yaxis_autorange="reversed",xaxis_title="Layout column",yaxis_title="Layout row");chart(fig)
    st.subheader("Reduced acquisition actually changes preprocessing")
    acquisition=pd.DataFrame([{"Acquired channels":v["channels"],"Accuracy (%)":v["summary"]["accuracy_percent"]} for v in metrics["acquisition"]])
    st.dataframe(acquisition,hide_index=True)
    st.caption("Each outer fold selects channels using training labels only, then reruns CAR, pre-task AR and filtering on those raw channels. This differs from dropping classifier features.")
    st.dataframe(pd.DataFrame([{"Reference":name,"Accuracy (%)":v["accuracy_percent"]} for name,v in metrics["montage"].items()]),hide_index=True)
    trial=st.selectbox("Explain a held-out one-second decision",list(range(1,91)),index=64)
    values=models["default_contributions"][trial-1].sum(axis=1).reshape(6,10).T
    fig=go.Figure(go.Heatmap(z=values,text=numbers,colorscale="RdBu_r",zmid=0))
    fig.add_trace(go.Scatter(x=np.tile(np.arange(6),10),y=np.repeat(np.arange(10),6),mode="text",text=numbers.ravel(),textfont=dict(color=np.where(np.abs(values.ravel())>.6*np.max(np.abs(values)),"#ffffff","#0b1020")),showlegend=False,hoverinfo="skip"))
    fig.update_layout(title=f"Trial {trial}: contributions to Scissors minus Rock score",yaxis_autorange="reversed");chart(fig)
    st.caption(f"Sum of contributions + fixed offset {models['contribution_offsets'][trial-1]:.3f} reproduces the fitted score difference. The Paper score is also required for the final decision.")
    with st.expander("Training-channel redundancy"):
        fig=go.Figure(go.Heatmap(z=models["training_channel_correlations"],x=np.arange(1,61),y=np.arange(1,61),zmin=-1,zmax=1,colorscale="RdBu_r"))
        fig.update_layout(title="Feature correlations averaged over outer training folds");chart(fig)
        st.caption("Shared activity, reference effects and volume conduction can contribute to correlation.")

with st.expander("Download evidence and read limitations"):
    for limit in metrics["limitations"]: st.write("• "+limit)
    st.download_button("Per-trial prediction CSV",(DEFAULT_OUTPUT/"predictions.csv").read_bytes(),file_name="ecog_held_out_predictions.csv")
    st.download_button("Brain–hand trial table",(DEFAULT_OUTPUT/"trial_table.csv").read_bytes(),file_name="ecog_brain_hand_trials.csv")
    st.download_button("140-second presentation guide",(DEFAULT_OUTPUT/"PRESENTATION_140_SECONDS.md").read_bytes(),file_name="ecog_presentation_guide.md")
