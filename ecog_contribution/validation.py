"""Nested evaluation and calibration. All fitted choices use training data only."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix, f1_score, log_loss


def blocked(indices, folds=5, purge=1):
    indices = np.sort(np.asarray(indices, dtype=int))
    if len(indices) < folds*3:
        raise ValueError("Too few trials for blocked evaluation")
    for test in np.array_split(indices, folds):
        removed = np.concatenate([test+d for d in range(-purge,purge+1)])
        train = indices[~np.isin(indices, removed)]
        yield train, test


def rank_channels(features, labels, train, bins):
    x = features[train,:,:bins].reshape(len(train),-1)
    f, _ = f_classif(x, labels[train])
    score = np.nan_to_num(f, nan=0., posinf=0.).reshape(features.shape[1],bins).mean(axis=1)
    return np.argsort(-score, kind="stable")


def fit_model(features, labels, train, bins, budget):
    if set(labels[train]) != {1,2,3}:
        raise ValueError("A training fold lacks one of the three gestures")
    n = features.shape[1]
    selected = rank_channels(features, labels, train, bins)[:budget] if budget<n else np.arange(n)
    x = features[:,selected,:bins].reshape(len(features),-1)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(x[train], labels[train])
    return model, selected, x


def temperature(scores, labels):
    if not np.isfinite(scores).all() or set(labels) != {1,2,3}:
        raise ValueError("Calibration requires finite out-of-fold scores and all three classes")
    def loss(log_t):
        logits = scores/np.exp(log_t)
        return np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(labels)),labels-1])
    result = minimize_scalar(loss, bounds=(np.log(.05),np.log(1000)), method="bounded")
    if not result.success:
        raise ValueError("Temperature calibration failed")
    return float(np.exp(result.x))


def calibrated_fold(features, labels, train, test, bins, budget=60):
    inner = list(blocked(train,3))
    logits = np.empty((len(train),3))
    positions = {int(t):i for i,t in enumerate(train)}
    for itrain, itest in inner:
        model, _, x = fit_model(features, labels, itrain, bins, budget)
        logits[[positions[int(t)] for t in itest]] = model.decision_function(x[itest])
    t = temperature(logits, labels[train])
    model, selected, x = fit_model(features, labels, train, bins, budget)
    scores = model.decision_function(x[test])
    return model.predict(x[test]), softmax(scores/t,axis=1), t, selected, model, x


def classification_summary(y, pred, probabilities=None):
    y, pred = np.asarray(y), np.asarray(pred)
    result = {"accuracy_percent":float(np.mean(y==pred)*100),
              "macro_f1_percent":float(f1_score(y,pred,labels=[1,2,3],average="macro")*100),
              "correct":int(np.sum(y==pred)), "total":len(y),
              "confusion_matrix":confusion_matrix(y,pred,labels=[1,2,3]).tolist()}
    if probabilities is not None:
        p=np.asarray(probabilities)
        result["log_loss"] = float(log_loss(y,p,labels=[1,2,3]))
        result["multiclass_brier"] = float(np.mean(np.sum((p-np.eye(3)[y-1])**2,axis=1)))
        confidence = p.max(axis=1)
        reliability=[]
        for lo in np.arange(0,1,.1):
            mask=(confidence>=lo)&(confidence<(lo+.1) if lo<.9 else confidence<=1)
            if mask.any():
                reliability.append({"confidence":float(confidence[mask].mean()),"accuracy":float(np.mean(y[mask]==pred[mask])),"count":int(mask.sum())})
        result["calibration_bins"]=reliability
        result["ece"] = float(sum(b["count"]*abs(b["confidence"]-b["accuracy"]) for b in reliability)/len(y))
        result["coverage_curve"] = []
        for threshold in [.5,.6,.7,.8,.9,.95,.99]:
            mask=confidence>=threshold
            result["coverage_curve"].append({"threshold":threshold,"accepted":int(mask.sum()),
                "coverage_percent":float(mask.mean()*100),"accuracy_percent":float(np.mean(y[mask]==pred[mask])*100) if mask.any() else None})
    return result


def adaptive(y, predictions, probabilities, deadlines, threshold=.9):
    # Dimensions: deadlines x trials. All per-deadline probabilities are outer held-out.
    confidence=probabilities.max(axis=2)
    n=len(y); chosen=np.full(n,-1,dtype=int)
    for i in range(len(deadlines)):
        mask=(chosen<0)&(confidence[i]>=threshold)
        chosen[mask]=i
    accepted=chosen>=0
    pred=np.full(n,0,dtype=int); times=np.full(n,np.nan)
    pred[accepted]=predictions[chosen[accepted],np.flatnonzero(accepted)]
    times[accepted]=np.asarray(deadlines)[chosen[accepted]]
    return {"threshold":threshold,"accepted":int(accepted.sum()),"abstained":int((~accepted).sum()),
            "correct_accepted":int(np.sum(pred[accepted]==y[accepted])),
            "coverage_percent":float(accepted.mean()*100),
            "accuracy_accepted_percent":float(np.mean(pred[accepted]==y[accepted])*100) if accepted.any() else None,
            "mean_decision_seconds":float(np.nanmean(times)) if accepted.any() else None}, pred,times
