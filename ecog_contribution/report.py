"""Scientific figure exports and evidence-grounded presentation documents."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS={"Rock":"#4c9ee8","Scissors":"#ef9751","Paper":"#51b89c"}
MEG_URL="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1547916/full"


def write_report(output,metrics,table):
    output=Path(output);figures=output/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    with np.load(output/"display_traces.npz") as z:
        times=z["times"];hg=z["high_gamma"].mean(axis=1);glove=z["glove"]
        at=z["annotated_times"];ah=z["annotated_high_gamma"];ag=z["annotated_glove"]
        representative=int(z["annotated_trial_index"]);next_cue=float(z["annotated_next_cue_seconds"])
    def save(fig,name):
        fig.savefig(figures/(name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
        fig.savefig(figures/(name+".svg"),bbox_inches="tight",facecolor="white");plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout="constrained")
    row=table.iloc[representative]
    axes[0].plot(at,(((at>=0)&(at<2))|(at>=next_cue)).astype(int),color="#627089");axes[0].set_ylabel("Cue on")
    axes[1].plot(at,ag.T);axes[1].set_ylabel("Finger bend")
    axes[2].plot(at,ah,color=COLORS["Rock"]);axes[2].set_ylabel("High-gamma (dB)")
    for a in axes:
        a.axvspan(0,2,color="#bbbbbb",alpha=.15);a.axvline(2,color="#666666",linestyle="--")
        a.axvline(next_cue,color="#627089",linestyle="--")
        if np.isfinite(row.movement_onset_seconds): a.axvline(row.movement_onset_seconds,color="#df6c36",linestyle=":")
    axes[2].set_xlabel("Seconds relative to cue; dotted line = estimated glove onset")
    axes[0].set_title(f"Annotated trial {representative+1}: cue → movement → hold → release/rest → next cue\nPhase boundaries are descriptive; cue offset does not measure release onset")
    save(fig,"01_annotated_trial")
    paper=table[table.gesture=="Paper"];cases=[int(paper.movement_amount.idxmin()),int(paper.movement_amount.idxmax())]
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True,layout="constrained")
    for j,i in enumerate(cases):
        axes[0,j].plot(times,glove[i].T);axes[0,j].set_title(f"Paper trial {i+1}: movement amount {table.iloc[i].movement_amount:.3f}")
        axes[1,j].plot(times,hg[i],color=COLORS["Paper"])
        for a in axes[:,j]: a.axvspan(0,2,color="#bbbbbb",alpha=.15)
        axes[1,j].set_xlabel("Seconds relative to cue")
    axes[0,0].set_ylabel("Finger bend");axes[1,0].set_ylabel("High-gamma (dB)")
    fig.suptitle("Secondary hand-behavior inspection; not a verified still-hand/Paper decoder\nMinimum/maximum displacement among Paper trials; no cases excluded")
    save(fig,"02_same_gesture_different_movement")
    history=metrics["history"];effect=history["baseline_variants"]["absolute"]
    with np.load(output/"history_arrays.npz") as z: residual=z["absolute_residual"]
    previous=table.previous_movement_amount.iloc[1:].to_numpy();xx=(previous-previous.mean())/previous.std()
    partial=residual+effect["effect_db_per_history_sd"]*xx
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for name,color in COLORS.items():
        mask=(table.gesture.iloc[1:]==name).to_numpy();axes[0].scatter(xx[mask],partial[mask],color=color,label=name,alpha=.8)
    order=np.argsort(xx);axes[0].plot(xx[order],effect["effect_db_per_history_sd"]*xx[order],color="#222222")
    axes[0].set(xlabel="Preceding movement amount (SD units)",ylabel="Adjusted partial response (dB)",title="History after accounting for hand behavior")
    axes[0].legend(frameon=False)
    variants=list(history["baseline_variants"])
    for j,name in enumerate(variants):
        value=history["baseline_variants"][name];b=value["effect_db_per_history_sd"];lo,hi=value["bootstrap_95_interval"]
        axes[1].errorbar(b,j,xerr=[[max(0,b-lo)],[max(0,hi-b)]],fmt="o",color="#4c7198",capsize=5)
    axes[1].axvline(0,color="#666666",linestyle="--");axes[1].set_yticks(range(3),["Absolute","Shared pre-task reference","Trial-specific baseline"])
    axes[1].set(xlabel="Adjusted effect (dB / preceding movement SD)",title="95% five-trial block-bootstrap intervals")
    fig.suptitle("Secondary history diagnostic: inconclusive after hand-behavior adjustment")
    save(fig,"03_history_effect")
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True,layout="constrained")
    for j,i in enumerate([0,21]):
        axes[0,j].plot(times,glove[i].T);axes[0,j].set_title(f"Previously noted unusual Paper trial {i+1}")
        axes[1,j].plot(times,hg[i],color=COLORS["Paper"])
        for a in axes[:,j]: a.axvspan(0,2,color="#bbbbbb",alpha=.15)
        axes[1,j].set_xlabel("Seconds relative to cue")
    axes[0,0].set_ylabel("Finger bend");axes[1,0].set_ylabel("High-gamma (dB)")
    fig.suptitle("Cue, hand behavior and neural activity can diverge\nCases fixed from prior team notes; these traces do not establish the cause of an error")
    save(fig,"04_unusual_trials")
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for channels in [10,20,40,60]:
        items=[v for v in metrics["grid"].values() if v["budget"]==channels]
        axes[0].plot([v["deadline"] for v in items],[v["blocked"]["accuracy_percent"] for v in items],"o-",label=f"{channels} features channels")
    axes[0].set(xlabel="Window ends after cue (s)",ylabel="Purged-block accuracy (%)",title="Joint speed / channel comparison",ylim=(60,101));axes[0].legend(fontsize=8)
    conf=metrics["confidence"]["1.0"]
    bins=conf["calibration_bins"];axes[1].plot([0,1],[0,1],"--",color="#777777")
    axes[1].plot([b["confidence"] for b in bins],[b["accuracy"] for b in bins],"o-",color="#4c7198")
    axes[1].set(xlabel="Calibrated confidence",ylabel="Observed accuracy",title="Held-out one-second confidence",xlim=(0,1),ylim=(0,1))
    save(fig,"05_speed_and_confidence")
    write_whitening_material(output,metrics,table,save)


def write_whitening_material(output,metrics,table,save):
    evidence=metrics["whitening"];source=evidence["partner_reproduction"];causal=source["causal"]
    local=evidence["deadlines"]["2.25"]
    half=causal["first_half_holdout"];a=metrics["adaptive"]
    with np.load(output/"model_arrays.npz") as z:
        gain=z["partner_causal_trial_gain"]
        hz=z["whitening_spectrum_hz"];psd=z["whitening_spectrum_without"];white_psd=z["whitening_spectrum_first_half"]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    comparisons=[("G25 causal filters / whole-record fit",causal["accuracy_without_percent"],causal["accuracy_with_percent"]),
        ("Pre-task fit / repeated CV",local["random"]["paired"]["accuracy_without_percent"],local["random"]["paired"]["accuracy_with_percent"]),
        ("Pre-task fit / purged blocks",local["blocked"]["paired"]["accuracy_without_percent"],local["blocked"]["paired"]["accuracy_with_percent"]),
        ("G25 first-half fit / last 45 trials",half["without"]["accuracy_percent"],half["first_half"]["accuracy_percent"])]
    for j,(name,before,after) in enumerate(comparisons):
        axes[0].plot([before,after],[j,j],color="#4c7198")
        axes[0].scatter(before,j,color="#929ba8",label="Without whitening" if j==0 else None)
        axes[0].scatter(after,j,color="#197f85",label="AR(10) whitening" if j==0 else None)
    axes[0].set_yticks(range(len(comparisons)),[v[0] for v in comparisons]);axes[0].invert_yaxis()
    axes[0].set(xlabel="Accuracy (%)",title="Matched comparisons; protocols differ across rows");axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(1,91),gain*100,color=np.where(gain>=0,"#197f85","#dd8650"))
    axes[1].set(xlabel="Original trial",ylabel="Change in correctness fraction (pp)",title=f"G25 paired 20-seed result: {causal['trials_helped']} helped, {causal['trials_hurt']} hurt")
    fig.suptitle("Whitening Under Test: one recording, controlled processing comparisons")
    save(fig,"06_whitening_evidence")
    fig,ax=plt.subplots(figsize=(11,4),layout="constrained")
    windows=[("Pre-task AR fit",2.,10.08),("G25 first-45 AR fit",0.,source["half_cutoff_seconds"]),("Whole-record AR fit",0.,507025/1200)]
    for j,(name,start,end) in enumerate(windows):ax.broken_barh([(start,end-start)],(j-.25,.5),facecolors="#7b9db8")
    test_start=float(table.iloc[45].cue_seconds);ax.axvspan(test_start,float(table.iloc[-1].cue_seconds+2.25),color="#64b5a0",alpha=.18,label="G25 last-45 test cues")
    ax.axvline(test_start,color="#197f85",linestyle="--");ax.set_yticks(range(3),[v[0] for v in windows]);ax.invert_yaxis()
    ax.set(xlabel="Recording time (s)",title="Causal filter application and past-only calibration are separate checks")
    ax.legend(loc="lower right",fontsize=8);save(fig,"07_calibration_boundary")
    fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    for x,name,color in [(psd,"Without whitening","#929ba8"),(white_psd,"First-half AR(10)","#197f85")]:
        ax.plot(hz,10*np.log10(np.maximum(x[25],np.finfo(float).tiny)),label=name,color=color)
    ax.set(xlabel="Frequency (Hz)",ylabel="PSD (dB / Hz)",title="Predetermined electrode 26: later-signal spectrum before high-gamma filtering")
    ax.legend();save(fig,"08_whitening_spectrum")
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    names=["No whitening","AR(10)"]+[v["variant"] for v in source["sensitivity"]]
    values=[causal["accuracy_without_percent"],causal["accuracy_with_percent"]]+[v["summary"]["accuracy_percent"] for v in source["sensitivity"]]
    ax.plot(values,np.arange(len(names)),"o",color="#197f85");ax.set_yticks(range(len(names)),names);ax.invert_yaxis()
    ax.set(xlabel="Mean accuracy (%)",title="G25 causal 20-seed sensitivity: first difference ties AR(10)")
    save(fig,"09_whitening_mechanism_control")
    report=["# Whitening Under Test", "", "Fast, reliable ECoG gesture decoding: Team G25 source reproduction and controlled follow-up. Updated 6 October 2026.", "",
        "## Central contribution", "Whitening is the strongest supported preprocessing improvement in this project. We test it with matched predictions, causal filtering, earlier-signal calibration, speed and reliability. It is not claimed to be the uniquely best possible decoder or transform.", "",
        "Established submission stays **96.9% mean repeated 10-fold × 10 CV**. The old 97.8% was one 88/90 run. New numbers below use distinct protocols and remain exploratory.", "",
        "## Reproduce the supplied G25 follow-up",f"The original causal-filter comparison gives **{causal['accuracy_without_percent']:.1f}% → {causal['accuracy_with_percent']:.1f}%**, a **{causal['gain_percentage_points']:.2f} percentage-point** improvement. **{causal['trials_helped']} trials improved** their average correctness fraction across matching CV seeds and **{causal['trials_hurt']} worsened**.", "",
        f"Original one-sided Monte Carlo sign-flip p = **{causal['p_signflip_one_sided']:.8f}** (rounded **{causal['p_signflip_one_sided']:.4f}**), from 20,000 draws with seed 41 after the zero-phase test consumes the RNG. The test uses 90 per-trial correctness-fraction differences across 20 ten-fold seeds, not 1,800 independent trials. Serial dependence and prior exploration limit inference.", "",
        "Source: `ECoG_G25_code/4_partner_checks/partner_checks.py`, shared `3_oct5_analyses/common.py`, and order-control definitions from `4_partner_checks/whitening_audit.py`. Source hashes accompany the exported predictions.", "",
        "The headline filter application is causal, but its AR coefficients are fitted to the whole recording. It is retrospective within-recording evidence. Causal filtering alone does not make its calibration prospective.", "",
        "## Calibration that ends before test samples",f"The team's causal first-half check trains the classifier on trials 1–45 and tests 46–90. AR fitting stops at **{source['half_cutoff_seconds']:.2f} s**, before cue 46 at **{table.iloc[45].cue_seconds:.2f} s**. The score is **{half['without']['correct']}/45 without whitening → {half['first_half']['correct']}/45 with whitening**, or **{half['without']['accuracy_percent']:.1f}% → {half['first_half']['accuracy_percent']:.1f}%**. This is a modest one-trial gain, not 97.3% prospective accuracy.", "",
        "Our stricter time-midpoint comparison fits AR only before 211.26 s, purges one adjacent classifier-training trial, and tests 46 later cues. The 45/45 and purged 43/46 cohorts are different; compare only paired methods within each protocol.", "",
        "| Time-midpoint holdout | Method | Correct / tested | Accuracy |", "|---|---|---:|---:|"]
    for item in evidence["second_half"]:
        for name,v in item["summaries"].items():report.append(f"| {item['deadline']:.2f} s | {name} | {v['correct']}/{v['total']} | {v['accuracy_percent']:.2f}% |")
    report += ["", "## Matched pre-task-fit whitening ablation", "Fit AR(10) using only cue-free 2.00–10.08 s. CAR, causal high-pass/notches/band filter, channel count, binning, LDA and split indices match across whitening on/off. The fixed primary deadline is 2.25 s; other deadlines are sensitivity/supporting comparisons.", "",
        "| Fixed 2.25 s / 60 channels | Without | With | Gain | Within-recording block interval |", "|---|---:|---:|---:|---|"]
    for name,v in local.items():
        pair=v["paired"];lo,hi=pair["gain_block_bootstrap_95_interval"]
        report.append(f"| {name} | {pair['accuracy_without_percent']:.2f}% | {pair['accuracy_with_percent']:.2f}% | {pair['gain_percentage_points']:+.2f} pp | [{lo:.2f}, {hi:.2f}] pp |")
    report += ["", "These 10-fold × 10-repeat splits differ from the team's 20-seed comparison. For uncertainty, repeat outcomes are averaged inside each original trial, then five-trial blocks are resampled. Exact two-sided McNemar summaries use only one prediction per trial from the first repeat; they are descriptive and do not replace G25's one-sided sign-flip.", "",
        "## Mechanism control and limits", "A fitted-free first-difference filter reaches the same 97.3% in the G25 causal CV scheme. AR(5) and AR(20) are similar. This supports frequency weighting / temporal decorrelation as useful processing, while weakening a claim that fitted AR coefficients are uniquely necessary. Whitening adds no recorded information and does not remove all artifacts.", "",
        "| Source causal sensitivity | Accuracy |", "|---|---:|",f"| AR(10) | {causal['accuracy_with_percent']:.2f}% |"]
    report += [f"| {v['variant']} | {v['summary']['accuracy_percent']:.2f}% |" for v in source["sensitivity"]]
    report += ["", "## Fast decisions and reliability",f"The pre-task-whitened one-second model gives **{metrics['grid']['1.00s_60ch']['blocked']['correct']}/90** under purged chronological evaluation. The fixed 0.90 confidence policy accepts **{a['accepted']}/90**, correctly classifies **{a['correct_accepted']}/{a['accepted']}**, and abstains on **{a['abstained']}**. Processing time, sample waiting time and glove reaction time remain separate.", "",
        "Training-only feature selection, referencing, acquired-channel subsets and persistent channel loss after retraining remain supporting checks. Results cannot be transferred to an unseen participant or recording.", "",
        "## Withdrawn hypothesis and retained diagnostics", "The user reports that still hand versus cued Paper did not hold up in follow-up. Reliable separation is not established and is withdrawn from the central story. The provided same-hand/paradox scripts describe controls but do not include a numerical replication output here; no score is invented.", "",
        "Our earlier movement-history analysis was independently inconclusive after accounting for actual hand behavior. It remains a secondary diagnostic appendix. Neither high gesture accuracy nor an inconclusive rest/Paper test establishes movement-independent intention or absence of a neural task signal.", "",
        "## Scientific figures", *[f"- [{name.replace('_',' ')}](figures/{name}.png)" for name in ["06_whitening_evidence","07_calibration_boundary","08_whitening_spectrum","09_whitening_mechanism_control","05_speed_and_confidence"]], "",
        "Figures 01–04 retain annotated brain/glove and history cases as secondary diagnostics. Prior output artifacts are archived separately, so the change in project direction is traceable.", "",
        "## Limits and reproduction", *["- "+v for v in metrics["limitations"]], *["- "+v for v in source["limitations"]], "",
        "Run `python -m ecog_contribution run`; launch `streamlit run dashboard/app.py`. The dashboard reads cached results and never fits models. CSV/NPZ predictions, splits, calibration coefficients, configuration and source hashes are exported. Original recording and submission files stay unchanged."]
    (output/"CONTRIBUTION_REPORT.md").write_text("\n".join(report)+"\n")
    script=f"""# 140-second presentation guide: Whitening Under Test

Updated 6 October 2026. Display Team G25 and the verified team identity; use the team's actual submission template. Rehearse the spoken timing.

| Time | Visual | Message |
|---|---|---|
| 0–15 s | Question and pipeline | Can the same recorded signal yield a more reliable gesture decision? |
| 15–40 s | Whitening evidence | Reproduced 94.4% → 97.3%; matching seeds and paired trial changes |
| 40–65 s | Calibration timeline | Causal filtering and earlier-signal calibration are separate |
| 65–90 s | Mechanism control | First difference ties AR; avoid claiming unique mechanism |
| 90–115 s | Speed / confidence / replay | Earlier decisions, visible mistakes and abstentions |
| 115–140 s | Limits and contribution | Withdraw unsupported still-hand/Paper claim; test new recordings next |

Spoken draft:

We are Team G25, decoding rock, scissors and paper from sixty ECoG channels across ninety trials. Our strongest supported processing direction is spectral whitening. We ask whether it improves decisions when the rest of the pipeline stays the same.

Whitening predicts a channel's next sample from its recent history, then subtracts that predictable component. This changes the frequency balance before we measure high-gamma power. It adds no new recorded information.

We reproduced the team's follow-up scripts. With forward-only filters, average gesture accuracy rises from {causal['accuracy_without_percent']:.1f} to {causal['accuracy_with_percent']:.1f} percent. On matching cross-validation seeds, {causal['trials_helped']} trials improve their fraction correct and none worsen. The original one-sided sign-flip test rounds to p equals {causal['p_signflip_one_sided']:.4f}.

That headline uses a filter fitted on the whole recording. We therefore distinguish causal application from past-only calibration. Fitting through the first forty-five trials and testing the last forty-five gives {half['without']['correct']} correct without whitening and {half['first_half']['correct']} with it. This is a modest gain, not prospective ninety-seven-percent accuracy.

We also test pre-task-only calibration, purged chronological blocks, earlier decision deadlines and lost channels. At one second our whitened model gets {metrics['grid']['1.00s_60ch']['blocked']['correct']} of ninety correct. A confidence policy accepts {a['accepted']} trials, gets {a['correct_accepted']} right, and abstains on {a['abstained']}. Recorded replay exposes the mistakes.

A fixed first-difference filter ties AR whitening here. We report that control because the fitted AR model is not uniquely necessary for the observed improvement.

Our still-hand-versus-paper hypothesis was not established, so we withdrew it from the headline. These are exploratory results from one participant. Our contribution is verified preprocessing evidence with clear calibration boundaries, measurable responsiveness and reproducible failures. The next test needs an independent recording.
"""
    (output/"PRESENTATION_140_SECONDS.md").write_text(script)
    update=f"""# Team update draft — not sent

The project now centers on whitening as our strongest supported preprocessing direction. We reproduced ECoG_G25_code's causal-filter result: {causal['accuracy_without_percent']:.1f}% → {causal['accuracy_with_percent']:.1f}%, {causal['trials_helped']} trials with improved 20-seed correctness fraction, {causal['trials_hurt']} worsened, one-sided sign-flip p={causal['p_signflip_one_sided']:.8f} (rounds to {causal['p_signflip_one_sided']:.4f}).

We distinguish the whole-record AR fit behind that headline from a filter fitted only before the held-out period. The original causal first-45 → last-45 check improves {half['without']['correct']}/45 → {half['first_half']['correct']}/45. Pre-task-only and purged time-midpoint checks are now exported beside the source reproduction. First difference also reaches 97.3%, so we avoid claiming AR fitting is uniquely responsible.

The still-hand-versus-Paper interpretation is withdrawn as a central claim after the reported failed follow-up. Movement history, glove behavior and unusual cases remain secondary diagnostics. The established 96.9% submission stays intact. Dashboard, figures, report, 140-second script and portable team bundles have been refocused. Work stays local; this draft has not been sent.

한국어: 핵심 기여를 whitening의 검증으로 바꿨습니다. causal filter 결과 {causal['accuracy_without_percent']:.1f}% → {causal['accuracy_with_percent']:.1f}%와 동일 CV 조건의 개선 trial {causal['trials_helped']}개, 악화 {causal['trials_hurt']}개를 재현했습니다. 전체 기록으로 AR을 학습한 결과와 시험 구간 이전에 학습을 끝낸 결과를 구분합니다. first-difference도 비슷한 성능이어서 AR만의 고유한 효과라고 주장하지 않습니다. still-hand/Paper 가설은 핵심 주장으로 사용하지 않으며, 기존 96.9% 제출본과 원본 데이터는 보존했습니다. 아직 팀에 전송하지 않았습니다.
"""
    (output/"TEAM_UPDATE_DRAFT.md").write_text(update)
