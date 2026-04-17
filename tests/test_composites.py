"""Integration tests — build composites and run them via process-bigraph."""

import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter, gather_emitter_results

from pbg_comets import (
    DynamicFBAProcess,
    SpatialDynamicFBAProcess,
    CometsProcess,
    make_dfba_document,
    make_spatial_dfba_document,
    make_comets_document,
)


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('DynamicFBAProcess', DynamicFBAProcess)
    c.register_link('SpatialDynamicFBAProcess', SpatialDynamicFBAProcess)
    c.register_link('CometsProcess', CometsProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


@pytest.fixture
def default_media():
    return {
        'glc__D': 10.0, 'o2': 40.0, 'nh4': 1000.0, 'pi': 1000.0,
        'h2o': 1000.0, 'co2': 1000.0, 'h': 1000.0, 'ac': 0.0,
    }


def test_make_dfba_document_has_expected_shape(default_media):
    doc = make_dfba_document(
        models=['textbook'],
        model_ids=['Ecoli'],
        initial_biomass=[1e-3],
        initial_media=default_media,
        interval=0.5,
    )
    assert doc['community']['_type'] == 'process'
    assert doc['community']['address'] == 'local:DynamicFBAProcess'
    assert doc['community']['interval'] == 0.5
    assert doc['emitter']['_type'] == 'step'
    # Outputs are wired into stores.*
    out = doc['community']['outputs']
    assert out['biomass'] == ['stores', 'biomass']


def test_dfba_composite_runs(core, default_media):
    doc = make_dfba_document(
        models=['textbook'],
        model_ids=['Ecoli'],
        initial_biomass=[1e-3],
        initial_media=default_media,
        interval=1.0,
        substep=0.2,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(3.0)
    final = sim.state['stores']
    assert final['biomass']['Ecoli'] > 1e-3
    assert final['total_biomass'] > 1e-3


def test_dfba_emitter_collects_timeseries(core, default_media):
    doc = make_dfba_document(
        models=['textbook'],
        model_ids=['Ecoli'],
        initial_biomass=[1e-3],
        initial_media=default_media,
        interval=1.0,
        substep=0.2,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(3.0)
    results = gather_emitter_results(sim)
    # There is exactly one emitter keyed by its path
    assert len(results) == 1
    frames = list(results.values())[0]
    assert len(frames) >= 2
    # Times should be monotonic
    times = [f.get('time', 0.0) for f in frames]
    assert all(t1 <= t2 for t1, t2 in zip(times, times[1:]))
    # Biomass should be growing
    biomasses = [f['total_biomass'] for f in frames]
    assert biomasses[-1] > biomasses[0]


def test_make_spatial_dfba_document_has_expected_shape(default_media):
    doc = make_spatial_dfba_document(
        models=['textbook'],
        model_ids=['E_coli'],
        grid=[5, 5],
        initial_placement={'E_coli': [[2, 2, 1e-4]]},
        initial_media=default_media,
        interval=0.5,
    )
    assert doc['community']['address'] == 'local:SpatialDynamicFBAProcess'
    assert doc['community']['config']['grid'] == [5, 5]
    # Grid outputs wired
    out = doc['community']['outputs']
    assert out['biomass_grid'] == ['stores', 'biomass_grid']
    assert out['media_grid'] == ['stores', 'media_grid']


def test_spatial_dfba_composite_runs(core, default_media):
    doc = make_spatial_dfba_document(
        models=['textbook'],
        model_ids=['E_coli'],
        grid=[5, 5],
        initial_placement={'E_coli': [[2, 2, 5e-4]]},
        initial_media=default_media,
        biomass_diffusion=5e-6,
        media_diffusion=1e-5,
        substep=0.25,
        interval=0.5,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(1.5)
    final = sim.state['stores']
    assert final['total_biomass'] > 5e-4
    # Grid output present
    assert 'biomass_grid' in final
    bg = final['biomass_grid']['E_coli']
    assert len(bg) == 5 and len(bg[0]) == 5


def test_make_comets_document_has_expected_shape(default_media):
    doc = make_comets_document(
        models=['textbook'],
        model_ids=['Ecoli'],
        initial_biomass=[1e-6],
        initial_media=default_media,
        grid=[3, 3],
        interval=0.5,
    )
    assert doc['community']['address'] == 'local:CometsProcess'
    assert doc['community']['config']['grid'] == [3, 3]
