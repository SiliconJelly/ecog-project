from pathlib import Path
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from ecog_contribution.signal import clean,band_filter,whiten,fit_ar,build_features,reference
from ecog_contribution.validation import blocked,calibrated_fold,fit_model,temperature,adaptive
from ecog_contribution.behavior import sustained_onset
from ecog_contribution.artifacts import validate_cache,sha256,code_hash,cache_fingerprint


def test_chunk_equivalence_and_future_invariance():
    raw=np.random.default_rng(6).normal(size=(4,18000))
    batch,_=clean(raw,chunk=18000);chunks,_=clean(raw,chunk=137)
    np.testing.assert_allclose(batch,chunks,rtol=1e-12,atol=1e-12)
    a=fit_ar(batch[:,2400:6000]);b=band_filter(whiten(batch,a,chunk=18000),[50,300],chunk=18000)
    c=band_filter(whiten(chunks,a,chunk=137),[50,300],chunk=137)
    np.testing.assert_allclose(b,c,rtol=1e-12,atol=1e-12)
    cut=12000;changed=raw.copy();changed[:,cut:]=1e8
    prefix,_=clean(raw[:,:cut]);modified,_=clean(changed)
    np.testing.assert_array_equal(prefix,modified[:,:cut])
    np.testing.assert_allclose(b[:,:cut],band_filter(whiten(prefix,a),[50,300]),rtol=1e-12,atol=1e-12)


def test_selection_calibration_ignore_outer_labels():
    rng=np.random.default_rng(5);features=rng.normal(size=(90,6,3));labels=np.tile([1,2,3],30)
    features[:,0,0]+=labels*.7
    train,test=next(blocked(np.arange(90)))
    changed=labels.copy();changed[test]=4-labels[test]
    with threadpool_limits(limits=1):
        first=calibrated_fold(features,labels,train,test,3,3)
        second=calibrated_fold(features,changed,train,test,3,3)
    for i in [0,1,3]:np.testing.assert_array_equal(first[i],second[i])
    assert first[2]==second[2]
    np.testing.assert_array_equal(first[4].coef_,second[4].coef_)


def test_purging_and_complete_coverage():
    coverage=np.zeros(90,int)
    for train,test in blocked(np.arange(90)):
        coverage[test]+=1
        assert np.min(np.abs(train[:,None]-test))>1
        for itrain,itest in blocked(train,3):
            assert not np.intersect1d(test,itrain).size
            assert np.min(np.abs(itrain[:,None]-itest))>1
    assert np.all(coverage==1)


def test_missing_onset_absent_classes_and_invalid_signal():
    onset,threshold=sustained_onset(np.zeros(2400),np.zeros(1200))
    assert np.isnan(onset) and threshold==.02
    onset,_=sustained_onset(np.r_[np.zeros(120),np.full(60,.03),np.zeros(2220)],np.zeros(1200))
    assert onset==.1
    with pytest.raises(ValueError,match='lacks'):fit_model(np.zeros((20,3,2)),np.ones(20,int),np.arange(15),2,3)
    with pytest.raises(ValueError,match='three classes'):temperature(np.ones((10,3)),np.ones(10,int))
    with pytest.raises(ValueError,match='Flat'):fit_ar(np.zeros((2,100)))
    with pytest.raises(ValueError,match='finite'):clean(np.full((3,100),np.nan))
    with pytest.raises(ValueError,match='neighbor'):reference(np.ones((2,100)),montage='local',channel_ids=[0,59])


def test_no_confident_trials_counts_abstentions():
    summary,pred,times=adaptive(np.array([1,2,3]),np.ones((2,3),int),np.full((2,3,3),1/3),[.5,1.])
    assert summary['accepted']==0 and summary['abstained']==3
    assert summary['accuracy_accepted_percent'] is None
    assert not pred.any() and np.isnan(times).all()


def test_acquisition_omits_excluded_channels_and_glove():
    rng=np.random.default_rng(4);raw=rng.normal(size=(5,18000));onsets=np.array([12000])
    kept=np.array([0,2,4]);changed=raw.copy();changed[[1,3]]=1e9
    first=build_features(raw[kept],onsets,channel_ids=kept)[0]
    second=build_features(changed[kept],onsets,channel_ids=kept)[0]
    np.testing.assert_array_equal(first,second)
    assert first.shape==(1,3,8)
    assert not np.allclose(reference(raw)[kept],reference(raw[kept]))
    # Only 60 ECoG rows cross the predictor interface; altering cue/glove rows cannot change it.
    data=np.vstack([np.zeros((1,18000)),raw,np.zeros((6,18000))]);mutated=data.copy();mutated[6:]=rng.normal(size=(6,18000))
    np.testing.assert_array_equal(build_features(data[1:6],onsets)[0],build_features(mutated[1:6],onsets)[0])


def test_missing_and_stale_cache(tmp_path):
    assert not validate_cache(tmp_path)[0]
    data=tmp_path/'recording';data.write_bytes(b'data');artifact=tmp_path/'metrics.json';artifact.write_text('{}')
    manifest={'schema_version':1,'code_sha256':code_hash(),'data_path':str(data),'data_sha256':sha256(data),'files':{'metrics.json':sha256(artifact)}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));assert validate_cache(tmp_path)[0]
    stamp=cache_fingerprint(tmp_path);artifact.write_text('{"changed":true}')
    assert cache_fingerprint(tmp_path)!=stamp and not validate_cache(tmp_path)[0]
    artifact.write_text('{}');data.write_bytes(b'changed');assert not validate_cache(tmp_path)[0]
    data.write_bytes(b'data');manifest['code_sha256']='obsolete';(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert not validate_cache(tmp_path)[0]


def test_protected_reproduction_inputs_invalidate_cache(tmp_path,monkeypatch):
    from ecog_contribution import artifacts
    original_code_hash=code_hash()
    monkeypatch.setattr(artifacts,'ROOT',tmp_path)
    monkeypatch.setattr(artifacts,'code_hash',lambda:original_code_hash)
    data=tmp_path/'recording';data.write_bytes(b'data')
    source=tmp_path/'source.py';source.write_text('unchanged')
    manifest={'schema_version':1,'code_sha256':original_code_hash,'data_path':str(data),
              'data_sha256':sha256(data),'files':{},'protected_source_hashes':{'source.py':sha256(source)}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert validate_cache(tmp_path)[0]
    stamp=cache_fingerprint(tmp_path);source.write_text('changed')
    assert cache_fingerprint(tmp_path)!=stamp
    assert 'Protected reproduction input' in validate_cache(tmp_path)[1]
    source.unlink();assert not validate_cache(tmp_path)[0]
