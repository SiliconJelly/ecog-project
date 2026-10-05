import json
import numpy as np
import pandas as pd
import pytest
from ecog_contribution.artifacts import DEFAULT_OUTPUT,validate_cache,sha256
from ecog_contribution.validation import classification_summary

@pytest.fixture(scope='module')
def results():
    assert validate_cache()[0], 'Run python -m ecog_contribution run before artifact acceptance tests'
    m=json.loads((DEFAULT_OUTPUT/'metrics.json').read_text());p=pd.read_csv(DEFAULT_OUTPUT/'predictions.csv',low_memory=False)
    with np.load(DEFAULT_OUTPUT/'model_arrays.npz') as z:a={k:z[k] for k in z.files}
    return m,p,a


def assert_metric(actual,expected):
    for key in ['correct','total','confusion_matrix']:assert actual[key]==expected[key]
    assert actual['accuracy_percent']==pytest.approx(expected['accuracy_percent'])
    if 'log_loss' in expected:
        for key in ['log_loss','multiclass_brier','ece']:assert actual[key]==pytest.approx(expected[key])


def test_historical_sources_and_coverage(results):
    m,p,a=results
    assert all(v['exact_fold_scores'] for v in m['historical_reproduction'].values())
    assert m['historical_reproduction']['whole_record_ar']['accuracy_percent']==pytest.approx(96.8888888889)
    for path,digest in m['provenance']['protected_hashes'].items():assert sha256(path)==digest
    for key,item in m['grid'].items():
        for protocol,repeats in [('blocked',1),('random',10)]:
            rows=p[(p.condition==key)&(p.protocol==protocol)]
            assert len(rows)==90*repeats
            assert (rows.groupby(['repeat','trial']).size()==1).all()
            assert_metric(classification_summary(rows.cue_code,rows.prediction_code),item[protocol])
    assert '1.00s_20ch' in m['grid']


def test_calibration_abstentions_and_remaining_metrics(results):
    m,p,a=results;y=a['labels']
    for i,d in enumerate(m['config']['deadlines']):
        assert_metric(classification_summary(y,a['calibrated_predictions'][i],a['calibrated_probabilities'][i]),m['confidence'][str(d)])
    pred=a['adaptive_predictions'];accepted=pred!=0
    assert m['adaptive']['accepted']==int(accepted.sum())
    assert m['adaptive']['abstained']==int((~accepted).sum())
    assert m['adaptive']['correct_accepted']==int((pred[accepted]==y[accepted]).sum())
    assert len(pred)==m['adaptive']['accepted']+m['adaptive']['abstained']==90
    rows=p[p.condition=='adaptive_0.90'];assert (rows.prediction_code==0).sum()==m['adaptive']['abstained']
    assert_metric(classification_summary(y,a['nested_predictions']),m['nested_joint'])
    for item in m['acquisition']:
        n=item['channels'];assert_metric(classification_summary(y,a[f'acquisition_{n}_predictions'],a[f'acquisition_{n}_probabilities']),item['summary'])
        assert all(len(ids)==n for ids in item['selected_per_fold'])
    for item in m['channel_loss']:
        n,s=item['removed'],item['seed'];assert len(item['dropped_channels'])==n
        assert_metric(classification_summary(y,a[f'loss_{n}_{s}_predictions'].ravel()),item['summary'])
    for name,metric in m['montage'].items():assert_metric(classification_summary(y,a[name+'_predictions'].ravel()),metric)


def test_forward_is_past_only_and_predictions_exported(results):
    m,p,a=results
    for item in m['forward']:
        for fold in item['folds']:
            assert max(fold['train_trials'])<min(fold['test_trials'])-1
        rows=p[(p.protocol=='forward')&(p.deadline==item['deadline'])]
        assert len(rows)==45 and rows.trial.nunique()==45
        assert_metric(classification_summary(rows.cue_code,rows.prediction_code),item['summary'])
    for fold in m['folds']:
        tr,te=np.array(fold['train_trials']),np.array(fold['test_trials'])
        assert np.min(np.abs(tr[:,None]-te))>1


def test_history_table_alignment_and_all_trials_retained(results):
    m,p,a=results;t=pd.read_csv(DEFAULT_OUTPUT/'trial_table.csv')
    assert len(t)==90 and m['history']['transition_count']==89
    assert t.previous_movement_amount.iloc[1:].to_numpy()==pytest.approx(t.movement_amount.iloc[:-1].to_numpy())
    with np.load(DEFAULT_OUTPUT/'history_arrays.npz') as z:
        for name,item in m['history']['baseline_variants'].items():
            y=z[name+'_observed'];assert len(y)==89
            assert np.mean((y-z[name+'_base_oof'])**2)==pytest.approx(item['base_chronological_mse'])
            assert np.mean((y-z[name+'_history_oof'])**2)==pytest.approx(item['history_chronological_mse'])
    with np.load(DEFAULT_OUTPUT/'display_traces.npz') as z:
        assert z['high_gamma'].shape==(90,60,800)
        assert z['annotated_times'][-1]>z['annotated_next_cue_seconds']
