from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from ecog_contribution import validation

APP=Path(__file__).resolve().parents[1]/'dashboard/app.py'

def control(elements,label):
    return next(x for x in elements if x.label==label)

@pytest.fixture
def app(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Dashboard attempted model fitting')
    monkeypatch.setattr(validation,'fit_model',forbidden)
    at=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not at.exception
    assert 'Whitening' in at.title[0].value
    return at


def test_all_views_and_neuroscience_controls(app):
    for page in ['Whitening evidence','Reliability','Signal diagnostics','Spatial signals','Overview']:
        control(app.radio,'Perspective').set_value(page).run()
        assert not app.exception
        if page=='Whitening evidence':
            for choice in ['Past-only pre-task calibration','First-half calibration','G25 reported follow-up']:
                control(app.radio,'Whitening comparison').set_value(choice).run();assert not app.exception
        if page=='Signal diagnostics':
            for sub in ['Movement and rhythms','Unusual trials','Adjusted history']:
                control(app.radio,'Analysis').set_value(sub).run();assert not app.exception
                if sub=='Movement and rhythms':
                    control(app.radio,'Alignment').set_value('Estimated glove movement').run()
                    for band in ['beta','erp','high_gamma']:
                        control(app.selectbox,'Signal').set_value(band).run();assert not app.exception
                if sub=='Adjusted history':
                    control(app.selectbox,'High-gamma reference').set_value('trial_baseline').run();assert not app.exception


def test_replay_gating_errors_empty_channels(app):
    control(app.radio,'Perspective').set_value('Trial replay').run();assert not app.exception
    assert control(app.metric,'Held-out prediction').value=='Waiting for samples'
    control(app.slider,'Seek time from cue (s)').set_value(.95).run()
    assert control(app.metric,'Held-out prediction').value=='Waiting for samples'
    control(app.slider,'Seek time from cue (s)').set_value(1.).run()
    assert control(app.metric,'Held-out prediction').value in ['Rock','Scissors','Paper']
    control(app.multiselect,'Electrodes for displayed neural trace').set_value([]).run();assert not app.exception
    control(app.checkbox,'Errors only').set_value(True).run();assert not app.exception
    assert len(control(app.selectbox,'Trial').options)==2
    control(app.button,'Restart').click().run()
    assert control(app.metric,'Held-out prediction').value=='Waiting for samples'
    control(app.select_slider,'Window end (s)').set_value(.5).run();assert not app.exception
    control(app.radio,'Replay decoder').set_value('Without whitening').run();assert not app.exception
    assert control(app.metric,'Held-out prediction').value=='Waiting for samples'


def test_missing_results_shows_regeneration(monkeypatch):
    import streamlit as st
    from ecog_contribution import artifacts
    st.cache_data.clear()
    monkeypatch.setattr(artifacts,'validate_cache',lambda: (False,'Missing or changed artifact: predictions.csv'))
    at=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value=='Prepare the contribution suite'
    assert 'predictions.csv' in at.warning[0].value
    assert 'python -m ecog_contribution run' in at.code[0].value
    st.cache_data.clear()
