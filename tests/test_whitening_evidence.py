import json
import numpy as np
import pandas as pd
import pytest
from ecog_contribution.artifacts import DEFAULT_OUTPUT,sha256
from ecog_contribution.signal import fit_ar,clean,FS
from ecog_contribution.whitening import paired_summary,first_half_split,fixed_pair
from ecog_contribution.validation import classification_summary
from threadpoolctl import threadpool_limits


def test_repeat_aggregation_and_paired_accounting():
    y=np.array([1,2,3,1,2,3]);off=np.tile(y,(10,1));on=off.copy()
    off[:,0]=2;on[:5,0]=2;on[:,1]=1
    pair=paired_summary(y,off,on,np.arange(1,7),bootstrap_samples=100)
    assert pair['unique_trials']==6 and pair['prediction_exposures']==60
    assert pair['gain_percentage_points']==pytest.approx(-100/12)
    first=pair['first_repeat']
    assert sum(first[k] for k in ['helped','hurt','both_correct','both_wrong'])==6
    assert len(first['hurt_trials'])==first['hurt']
    with pytest.raises(ValueError,match='matching'):paired_summary(y,off,on[:2],np.arange(6))


def test_first_half_boundaries_and_future_calibration_invariance():
    onsets=np.arange(12,420,4)*FS
    cutoff,train,test=first_half_split(onsets,422*FS)
    assert (onsets[test]>=cutoff).all()
    assert (onsets[train]+2.25*FS<=cutoff).all()
    assert train.max()<test.min()-1
    raw=np.random.default_rng(7).normal(size=(3,15000));boundary=9000
    changed=raw.copy();changed[:,boundary:]=1e8
    first,_=clean(raw);second,_=clean(changed)
    np.testing.assert_array_equal(fit_ar(first[:,2400:boundary]),fit_ar(second[:,2400:boundary]))
    rng=np.random.default_rng(8);features={'off':rng.normal(size=(90,3,8)),'first_half':rng.normal(size=(90,3,8))}
    labels=np.tile([1,2,3],30);tr=np.arange(43);te=np.arange(44,90)
    changed_labels=labels.copy();changed_labels[te]=4-labels[te]
    with threadpool_limits(limits=1):
        p0=fixed_pair(features,labels,tr,te,1.);p1=fixed_pair(features,changed_labels,tr,te,1.)
    for name in p0:np.testing.assert_array_equal(p0[name],p1[name])


@pytest.fixture(scope='module')
def exported():
    m=json.loads((DEFAULT_OUTPUT/'metrics.json').read_text())
    with np.load(DEFAULT_OUTPUT/'model_arrays.npz') as z:a={k:z[k] for k in z.files}
    p=pd.read_csv(DEFAULT_OUTPUT/'predictions.csv',low_memory=False)
    return m,a,p


def test_exact_source_followup_and_serial_dependence_labels(exported):
    m,a,p=exported;v=m['whitening']['partner_reproduction'];y=a['labels']
    assert all(v['matches_reported_followup'].values())
    for tag in ['causal','zero_phase']:
        off=a[f'partner_{tag}_off'];on=a[f'partner_{tag}_whole'];item=v[tag]
        assert off.shape==on.shape==(20,90)
        assert np.mean(off==y)*100==pytest.approx(item['accuracy_without_percent'])
        assert np.mean(on==y)*100==pytest.approx(item['accuracy_with_percent'])
        d=(on==y).mean(axis=0)-(off==y).mean(axis=0)
        assert (d>0).sum()==item['trials_helped'] and (d<0).sum()==item['trials_hurt']
        assert item['unique_trials']==90 and item['prediction_exposures']==1800
        rows=p[(p.protocol=='partner_20seed_cv')&(p.condition==f'partner_{tag}_whole_record')]
        assert len(rows)==1800
        assert rows.correct.mean()*100==pytest.approx(item['accuracy_with_percent'])
        for method,key in [('off','without'),('on','first_half')]:
            pred=a[f'partner_{tag}_half_{method}']
            score=classification_summary(y[45:],pred)
            assert score['confusion_matrix']==item['first_half_holdout'][key]['confusion_matrix']
    assert round(v['causal']['p_signflip_one_sided'],4)==.0004
    rng=np.random.default_rng(41)
    for tag in ['zero_phase','causal']:
        delta=a[f'partner_{tag}_trial_gain']
        signs=rng.choice([-1,1],size=(20000,90))
        p_value=(((signs*delta).mean(axis=1)>=delta.mean()).sum()+1)/20001
        assert p_value==v[tag]['p_signflip_one_sided']
    for item in v['sensitivity']:
        key={'first_difference':'first_difference','AR(5) whole-record':'ar5','AR(20) whole-record':'ar20'}[item['variant']]
        assert classification_summary(np.tile(y,20),a[f'partner_causal_{key}'].ravel())==item['summary']
    assert 'one-sided' in v['paired_test'].lower()
    assert v['half_cutoff_seconds']<a['onsets'][45]/FS
    for path,digest in v['source_hashes'].items():
        from ecog_contribution.artifacts import ROOT
        assert sha256(ROOT/path)==digest
    for seed in range(20):
        coverage=np.zeros(90,int)
        for fold in range(10):
            train=a[f'partner_seed_{seed}_fold_{fold}_train'];test=a[f'partner_seed_{seed}_fold_{fold}_test']
            assert not np.intersect1d(train,test).size;coverage[test]+=1
        assert np.all(coverage==1)


def test_ablation_metrics_all_deadlines_and_half_holdouts(exported):
    m,a,p=exported;y=a['labels'];e=m['whitening']
    for d,item in e['deadlines'].items():
        for protocol in ['random','blocked']:
            off=a[f'whitening_off_{float(d):.2f}_{protocol}'];on=a[f'{float(d):.2f}s_60ch_{protocol}']
            pair=paired_summary(y,off,on,np.arange(1,91))
            expected=item[protocol]['paired']
            assert pair==expected
            rows=p[(p.condition==f'whitening_off_{float(d):.2f}')&(p.protocol==protocol)]
            assert rows.correct.mean()*100==pytest.approx(expected['accuracy_without_percent'])
    tr,te=a['first_half_train'],a['first_half_test']
    assert tr.max()<te.min()-1
    assert (a['onsets'][te]>=507025//2).all()
    for item in e['second_half']:
        d=item['deadline']
        for name,v in item['summaries'].items():
            score=classification_summary(y[te],a[f'second_half_{name}_{d:.2f}'])
            assert score==v
    for item in e['forward']:
        ids=np.concatenate([np.array(f['test_trials'])-1 for f in item['folds']])
        for fold in item['folds']:assert max(fold['train_trials'])<min(fold['test_trials'])-1
        for name,v in item['summaries'].items():
            assert classification_summary(y[ids],a[f'whitening_forward_{name}_{item["deadline"]:.2f}'])==v
    assert 'Not established' in e['withdrawn_hypothesis']['status']
    assert m['history']['status'].startswith('Secondary')
