"""Unit tests for CometsProcess and DynamicFBAProcess.

Tests that require the COMETS Java engine (``COMETS_HOME`` env var) are
skipped when the environment is not set up.
"""

import os
import pytest
from process_bigraph import allocate_core

from pbg_comets import (
    CometsProcess,
    DynamicFBAProcess,
    SpatialDynamicFBAProcess,
)


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('DynamicFBAProcess', DynamicFBAProcess)
    c.register_link('SpatialDynamicFBAProcess', SpatialDynamicFBAProcess)
    c.register_link('CometsProcess', CometsProcess)
    return c


@pytest.fixture
def default_media():
    return {
        'glc__D': 10.0,
        'o2': 40.0,
        'nh4': 1000.0,
        'pi': 1000.0,
        'h2o': 1000.0,
        'co2': 1000.0,
        'h': 1000.0,
        'ac': 0.0,
    }


# ---------------------------------------------------------------------------
# DynamicFBAProcess
# ---------------------------------------------------------------------------

def test_dfba_instantiates_with_textbook_model(core):
    proc = DynamicFBAProcess(config={
        'models': ['textbook'],
        'initial_biomass': [1e-3],
        'initial_media': {'glc__D': 10.0, 'o2': 20.0, 'nh4': 100.0, 'pi': 100.0,
                          'h2o': 100.0, 'co2': 100.0, 'h': 100.0},
    }, core=core)
    assert proc is not None


def test_dfba_initial_state_reports_initial_biomass(core, default_media):
    proc = DynamicFBAProcess(config={
        'models': ['textbook'],
        'model_ids': ['Ecoli'],
        'initial_biomass': [0.02],
        'initial_media': default_media,
    }, core=core)
    s0 = proc.initial_state()
    assert set(s0.keys()) == {
        'biomass', 'media', 'growth_rates', 'total_biomass'}
    assert s0['biomass']['Ecoli'] == pytest.approx(0.02)
    assert s0['total_biomass'] == pytest.approx(0.02)
    assert s0['media']['glc__D'] == pytest.approx(10.0)


def test_dfba_biomass_grows_under_FBA(core, default_media):
    """Ecoli core on glucose should increase biomass and deplete glucose."""
    proc = DynamicFBAProcess(config={
        'models': ['textbook'],
        'model_ids': ['Ecoli'],
        'initial_biomass': [1e-3],
        'initial_media': default_media,
        'substep': 0.1,
    }, core=core)
    proc.initial_state()
    s = proc.update({}, interval=1.0)
    assert s['biomass']['Ecoli'] > 1e-3
    assert s['growth_rates']['Ecoli'] > 0.1
    assert s['media']['glc__D'] < 10.0


def test_dfba_no_substrate_no_growth(core):
    """With no glucose, biomass should be stationary."""
    media = {'glc__D': 0.0, 'o2': 50.0, 'nh4': 100.0, 'pi': 100.0,
             'h2o': 100.0, 'co2': 100.0, 'h': 100.0}
    proc = DynamicFBAProcess(config={
        'models': ['textbook'],
        'initial_biomass': [1e-3],
        'initial_media': media,
        'substep': 0.1,
    }, core=core)
    proc.initial_state()
    s = proc.update({}, interval=2.0)
    assert s['growth_rates'][list(s['growth_rates'])[0]] == pytest.approx(
        0.0, abs=1e-6)
    assert s['biomass'][list(s['biomass'])[0]] == pytest.approx(1e-3, abs=1e-9)


def test_dfba_zero_interval_is_noop(core, default_media):
    proc = DynamicFBAProcess(config={
        'models': ['textbook'],
        'initial_biomass': [1e-3],
        'initial_media': default_media,
    }, core=core)
    s0 = proc.initial_state()
    s1 = proc.update({}, interval=0.0)
    assert s0['biomass'] == s1['biomass']
    assert s0['media'] == s1['media']


def test_dfba_two_species_share_glucose(core, default_media):
    """Two competing E. coli populations should both grow and share glucose."""
    proc = DynamicFBAProcess(config={
        'models': ['textbook', 'textbook'],
        'model_ids': ['A', 'B'],
        'initial_biomass': [1e-3, 1e-3],
        'initial_media': default_media,
        'substep': 0.1,
    }, core=core)
    proc.initial_state()
    s = proc.update({}, interval=1.0)
    assert s['biomass']['A'] > 1e-3
    assert s['biomass']['B'] > 1e-3
    # Total biomass consistent
    total = sum(s['biomass'].values())
    assert s['total_biomass'] == pytest.approx(total, rel=1e-9)


def test_dfba_bound_override_blocks_glucose_uptake(core, default_media):
    """A species with glucose uptake forbidden should not grow on glucose-only media."""
    proc = DynamicFBAProcess(config={
        'models': ['textbook'],
        'model_ids': ['E_noglc'],
        'initial_biomass': [1e-3],
        'initial_media': default_media,
        'bound_overrides': {
            'E_noglc': {'glc__D': [0.0, 1000.0]},
        },
        'substep': 0.1,
    }, core=core)
    proc.initial_state()
    s = proc.update({}, interval=2.0)
    assert s['growth_rates']['E_noglc'] == pytest.approx(0.0, abs=1e-6)
    # Biomass unchanged
    assert s['biomass']['E_noglc'] == pytest.approx(1e-3, abs=1e-9)


def test_dfba_ids_dedupe_when_repeated_models(core, default_media):
    proc = DynamicFBAProcess(config={
        'models': ['textbook', 'textbook'],
        'initial_biomass': [1e-3, 1e-3],
        'initial_media': default_media,
    }, core=core)
    s0 = proc.initial_state()
    ids = list(s0['biomass'].keys())
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_dfba_death_rate_reduces_biomass(core, default_media):
    """Without growth substrate, death_rate should reduce biomass exponentially."""
    media = {'glc__D': 0.0, 'o2': 50.0, 'nh4': 100.0, 'pi': 100.0,
            'h2o': 100.0, 'co2': 100.0, 'h': 100.0}
    proc = DynamicFBAProcess(config={
        'models': ['textbook'],
        'initial_biomass': [1e-2],
        'initial_media': media,
        'death_rate': 0.1,
        'substep': 0.1,
    }, core=core)
    proc.initial_state()
    s = proc.update({}, interval=2.0)
    # Biomass should decline
    assert list(s['biomass'].values())[0] < 1e-2


# ---------------------------------------------------------------------------
# SpatialDynamicFBAProcess
# ---------------------------------------------------------------------------

def test_spatial_initial_state_has_correct_grid_shape(core, default_media):
    proc = SpatialDynamicFBAProcess(config={
        'models': ['textbook'],
        'model_ids': ['E_coli'],
        'initial_placement': {'E_coli': [[3, 3, 1e-4]]},
        'initial_media': default_media,
        'grid': [6, 8],
    }, core=core)
    s0 = proc.initial_state()
    bg = s0['biomass_grid']['E_coli']
    assert len(bg) == 6 and len(bg[0]) == 8
    mg = s0['media_grid']['glc__D']
    assert len(mg) == 6 and len(mg[0]) == 8
    # Biomass is concentrated at the seed cell
    assert bg[3][3] == pytest.approx(1e-4)
    assert bg[0][0] == 0.0


def test_spatial_biomass_grows_and_diffuses(core, default_media):
    import numpy as np
    proc = SpatialDynamicFBAProcess(config={
        'models': ['textbook'],
        'model_ids': ['E_coli'],
        'initial_placement': {'E_coli': [[4, 4, 1e-3]]},
        'initial_media': default_media,
        'grid': [9, 9],
        'biomass_diffusion': 5e-6,
        'media_diffusion': 1e-5,
        'substep': 0.2,
    }, core=core)
    s0 = proc.initial_state()
    initial_total = s0['total_biomass']
    s = proc.update({}, interval=1.0)
    # Biomass should grow overall and spread beyond the seed cell
    assert s['total_biomass'] > initial_total
    bg = np.array(s['biomass_grid']['E_coli'])
    assert bg[4, 4] < 1e-3 + 1e-12  # some biomass diffused out OR grew
    assert int((bg > 1e-10).sum()) > 1


def test_spatial_diffusion_conserves_media_without_biomass(core):
    """Without any biomass, diffusion should conserve the total metabolite amount."""
    import numpy as np
    media = {'glc__D': 0.0, 'o2': 0.0}  # don't start uniform
    proc = SpatialDynamicFBAProcess(config={
        'models': ['textbook'],
        'initial_placement': {},  # no biomass anywhere
        'initial_media': media,
        'uniform_media': False,
        'initial_media_placement': {'glc__D': [[4, 4, 10.0]]},
        'grid': [9, 9],
        'media_diffusion': 1e-5,
        'substep': 0.2,
    }, core=core)
    s0 = proc.initial_state()
    total_0 = sum(sum(row) for row in s0['media_grid']['glc__D'])
    assert total_0 == pytest.approx(10.0, rel=1e-6)
    s = proc.update({}, interval=2.0)
    total_f = sum(sum(row) for row in s['media_grid']['glc__D'])
    # Reflective BCs -> total mass conserved
    assert total_f == pytest.approx(10.0, rel=1e-6)
    # And the mass actually spread
    assert s['media_grid']['glc__D'][4][4] < 10.0


def test_spatial_per_species_vmax_is_applied(core):
    """Different species can have different O2 kinetics at the same cell."""
    import numpy as np
    media = {'glc__D': 5.0, 'o2': 20.0, 'nh4': 100.0, 'pi': 100.0,
             'h2o': 100.0, 'co2': 100.0, 'h': 100.0, 'ac': 0.0}
    proc = SpatialDynamicFBAProcess(config={
        'models': ['textbook', 'textbook'],
        'model_ids': ['fast', 'slow'],
        'initial_placement': {
            'fast': [[1, 1, 1e-4]],
            'slow': [[2, 2, 1e-4]],
        },
        'initial_media': media,
        'grid': [4, 4],
        'per_species_vmax': {
            'fast': {'o2': 20.0, 'glc__D': 10.0},
            'slow': {'o2': 1.0,  'glc__D': 1.0},  # starved of both carbon & O2
        },
        'substep': 0.25,
    }, core=core)
    proc.initial_state()
    s = proc.update({}, interval=1.0)
    fast_total = sum(sum(row) for row in s['biomass_grid']['fast'])
    slow_total = sum(sum(row) for row in s['biomass_grid']['slow'])
    assert fast_total > slow_total * 2  # fast clearly ahead


def test_spatial_degenerates_to_single_cell_for_1x1_grid(core, default_media):
    proc = SpatialDynamicFBAProcess(config={
        'models': ['textbook'],
        'model_ids': ['E_coli'],
        'initial_biomass': [1e-3],
        'initial_media': default_media,
        'grid': [1, 1],
        'substep': 0.2,
    }, core=core)
    s0 = proc.initial_state()
    # Center cell (0,0) seeded
    assert s0['biomass_grid']['E_coli'] == [[1e-3]]
    s = proc.update({}, 1.0)
    # Growth should happen
    assert s['biomass_grid']['E_coli'][0][0] > 1e-3


# ---------------------------------------------------------------------------
# CometsProcess — most tests require COMETS_HOME
# ---------------------------------------------------------------------------

def _has_comets():
    return bool(os.environ.get('COMETS_HOME')) and os.path.isdir(
        os.environ.get('COMETS_HOME', ''))


def test_comets_initial_state_does_not_require_COMETS_HOME(core):
    """initial_state() reports config values and doesn't invoke Java."""
    proc = CometsProcess(config={
        'models': ['textbook'],
        'model_ids': ['Ecoli'],
        'initial_biomass': [1e-6],
        'initial_media': {'glc__D': 10.0, 'o2': 20.0},
    }, core=core)
    s = proc.initial_state()
    assert s['biomass']['Ecoli'] == pytest.approx(1e-6)
    assert s['media']['glc__D'] == pytest.approx(10.0)


def test_comets_update_without_COMETS_HOME_raises_clear_error(core, monkeypatch):
    monkeypatch.delenv('COMETS_HOME', raising=False)
    proc = CometsProcess(config={
        'models': ['textbook'],
        'initial_biomass': [1e-6],
        'initial_media': {'glc__D': 10.0},
    }, core=core)
    with pytest.raises(RuntimeError, match='COMETS_HOME'):
        proc.update({}, interval=1.0)


@pytest.mark.skipif(not _has_comets(), reason='COMETS_HOME not set')
def test_comets_runs_end_to_end(core):
    """Live integration test — only runs when COMETS is installed."""
    proc = CometsProcess(config={
        'models': ['textbook'],
        'model_ids': ['Ecoli'],
        'initial_biomass': [1e-6],
        'initial_media': {'glc__D': 10.0, 'o2': 20.0,
                          'nh4': 100.0, 'pi': 100.0},
        'time_step': 0.1,
    }, core=core)
    proc.initial_state()
    s = proc.update({}, interval=2.0)
    assert s['biomass']['Ecoli'] > 1e-6
