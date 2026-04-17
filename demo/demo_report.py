"""Demo: COMETS-style multi-configuration dynamic FBA report.

Runs three distinct community simulations using :class:`DynamicFBAProcess`:
a monoculture, a two-species competition, and a two-species cross-feeding
interaction. Generates a self-contained interactive HTML report with
Plotly time-series charts (biomass, media, growth rates, community
composition), a colored bigraph-viz architecture diagram, and a
navigatable PBG document tree.

Kept self-contained: uses only the ``e_coli_core`` textbook model bundled
with ``cobra``, no internet or COMETS Java backend required.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time as _time
from typing import Any, Dict, List

from process_bigraph import allocate_core

from pbg_comets import DynamicFBAProcess, SpatialDynamicFBAProcess
from pbg_comets.composites import (
    make_dfba_document,
    make_spatial_dfba_document,
)


# ---------------------------------------------------------------------------
# Simulation configurations
# ---------------------------------------------------------------------------

_BASE_MEDIA = {
    'o2': 40.0,
    'nh4': 1000.0,
    'pi': 1000.0,
    'h2o': 1000.0,
    'co2': 1000.0,
    'h': 1000.0,
    'so4': 1000.0,
    'k': 1000.0,
    'mg2': 1000.0,
    'ca2': 1000.0,
    'fe2': 1000.0,
    'fe3': 1000.0,
    'cu2': 1000.0,
    'cl': 1000.0,
    'mn2': 1000.0,
    'zn2': 1000.0,
    'cobalt2': 1000.0,
    'mobd': 1000.0,
    'ni2': 1000.0,
}


CONFIGS: List[Dict[str, Any]] = [
    {
        'id': 'monoculture',
        'title': 'E. coli Monoculture on Glucose',
        'subtitle': 'Single-species batch growth with overflow metabolism',
        'description': (
            'A single population of E. coli (core model) is grown in batch '
            'on 15 mmol glucose under oxygen-limited conditions. Overflow '
            'metabolism drives acetate secretion while glucose is being '
            'consumed. After glucose depletion the culture enters a '
            'stationary phase with residual acetate in the medium.'
        ),
        'dfba_config': {
            'models': ['textbook'],
            'model_ids': ['E_coli'],
            'initial_biomass': [5e-4],
            'initial_media': {**_BASE_MEDIA, 'glc__D': 15.0, 'o2': 10.0, 'ac': 0.0},
            'volume': 1.0,
            'default_vmax': 10.0,
            'default_km': 0.01,
            'vmax_overrides': {'o2': 10.0},
            'substep': 0.1,
        },
        'total_time': 12.0,
        'n_snapshots': 60,
        'color_scheme': 'indigo',
        'tracked_media': ['glc__D', 'ac', 'etoh', 'for', 'o2', 'lac__D'],
    },
    {
        'id': 'competition',
        'title': 'Two-Species Competition for Glucose',
        'subtitle': 'Identical E. coli strains with different initial biomass',
        'description': (
            'Two E. coli populations compete for the same glucose pool. '
            'The "minor" strain starts at 10% of the biomass of the "major" '
            'strain. Because the models are identical and growth is '
            'exponential, the minor strain never catches up — a Lotka-'
            'Volterra-style competitive exclusion dynamic where both '
            'strains grow until the shared substrate is exhausted.'
        ),
        'dfba_config': {
            'models': ['textbook', 'textbook'],
            'model_ids': ['E_major', 'E_minor'],
            'initial_biomass': [1e-3, 1e-4],
            'initial_media': {**_BASE_MEDIA, 'glc__D': 10.0, 'o2': 20.0, 'ac': 0.0},
            'volume': 1.0,
            'default_vmax': 10.0,
            'default_km': 0.02,
            'vmax_overrides': {'o2': 20.0},
            'substep': 0.1,
        },
        'total_time': 8.0,
        'n_snapshots': 60,
        'color_scheme': 'emerald',
        'tracked_media': ['glc__D', 'ac', 'etoh', 'o2', 'for', 'lac__D'],
    },
    {
        'id': 'crossfeeding',
        'title': 'Cross-Feeding: Glucose Producer + Acetate Consumer',
        'subtitle': 'Commensal two-species interaction on a shared pool',
        'description': (
            'Two E. coli populations with engineered exchange bounds: the '
            'glucose consumer (E_glc) grows on glucose and secretes '
            'acetate under oxygen-limited conditions; the acetate consumer '
            '(E_ac) cannot eat glucose but has its acetate uptake opened '
            'and grows on the acetate waste. The result is sequential '
            'resource use — classic cross-feeding — producing a double '
            'inflection in the total biomass curve.'
        ),
        'dfba_config': {
            'models': ['textbook', 'textbook'],
            'model_ids': ['E_glc', 'E_ac'],
            'initial_biomass': [5e-4, 5e-4],
            'initial_media': {**_BASE_MEDIA, 'glc__D': 15.0, 'o2': 8.0, 'ac': 0.0},
            'volume': 1.0,
            'default_vmax': 10.0,
            'default_km': 0.01,
            'vmax_overrides': {'o2': 8.0, 'ac': 5.0},
            'bound_overrides': {
                'E_ac': {
                    'glc__D': [0.0, 1000.0],   # block glucose uptake
                    'ac': [-1000.0, 1000.0],   # open acetate uptake
                },
            },
            'substep': 0.1,
        },
        'total_time': 18.0,
        'n_snapshots': 90,
        'color_scheme': 'rose',
        'tracked_media': ['glc__D', 'ac', 'etoh', 'o2', 'for', 'lac__D'],
    },
]


# Spatial (COMETS-style) simulations — 2D grid with local FBA + diffusion.
# These drive the ``SpatialDynamicFBAProcess`` and are rendered as
# time-animated 2D heatmap viewers in the report.

def _unit_salts(overrides=None):
    """Standard trace-metabolite background per cell (mmol)."""
    base = {
        'nh4': 100.0, 'pi': 100.0, 'h2o': 100.0, 'co2': 100.0, 'h': 100.0,
        'so4': 100.0, 'k': 100.0, 'mg2': 100.0, 'ca2': 100.0,
        'fe2': 100.0, 'fe3': 100.0, 'cu2': 100.0, 'cl': 100.0,
        'mn2': 100.0, 'zn2': 100.0, 'cobalt2': 100.0, 'mobd': 100.0,
        'ni2': 100.0,
    }
    if overrides:
        base.update(overrides)
    return base


SPATIAL_CONFIGS: List[Dict[str, Any]] = [
    {
        'id': 'colony',
        'title': 'Single-Species Colony Spreading',
        'subtitle': 'A bacterial colony expanding over a uniform nutrient field',
        'description': (
            'A single E. coli inoculum is placed at the center of a 25×25 '
            'lattice of 0.04 cm cells (1 cm² total). Each cell holds an '
            'initial pool of glucose and oxygen. Biomass diffuses very '
            'slowly (keeping the colony spatially compact) while glucose '
            'and acetate diffuse about 400× faster. A growing colony '
            'depletes its local substrate, and the nutrient gradient drives '
            'the classic COMETS radially-expanding growth front.'
        ),
        'spatial_config': {
            'models': ['textbook'],
            'model_ids': ['E_coli'],
            'initial_placement': {'E_coli': [[12, 12, 8e-5]]},
            'initial_media': _unit_salts({'glc__D': 0.8, 'o2': 8.0, 'ac': 0.0}),
            'grid': [25, 25],
            'space_width': 0.04,
            'biomass_diffusion': 3e-8,
            'media_diffusion': 1.5e-5,
            'vmax_overrides': {'o2': 20.0},
            'substep': 0.25,
        },
        'total_time': 14.0,
        'n_snapshots': 40,
        'color_scheme': 'indigo',
        'tracked_media': ['glc__D', 'ac'],
    },
    {
        'id': 'sp_competition',
        'title': 'Spatial Two-Colony Competition',
        'subtitle': 'Two colonies contacting across a shared nutrient field',
        'description': (
            'Two identical E. coli populations are inoculated at opposite '
            'edges of a 25×25 lattice and grow toward each other through a '
            'uniform glucose-and-oxygen field. Each colony depletes '
            'glucose locally; once the advancing fronts meet, the '
            'competing populations form a spatial "no-grow zone" where '
            'substrate is gone. Total biomass and the spatial niche '
            'partitioning are visible in the 2D viewer below.'
        ),
        'spatial_config': {
            'models': ['textbook', 'textbook'],
            'model_ids': ['E_left', 'E_right'],
            'initial_placement': {
                'E_left':  [[5,  12, 8e-5]],
                'E_right': [[19, 12, 8e-5]],
            },
            'initial_media': _unit_salts({'glc__D': 0.6, 'o2': 8.0, 'ac': 0.0}),
            'grid': [25, 25],
            'space_width': 0.04,
            'biomass_diffusion': 4e-8,
            'media_diffusion': 1.2e-5,
            'vmax_overrides': {'o2': 20.0},
            'substep': 0.25,
        },
        'total_time': 14.0,
        'n_snapshots': 40,
        'color_scheme': 'emerald',
        'tracked_media': ['glc__D', 'ac'],
    },
    {
        'id': 'sp_crossfeed',
        'title': 'Spatial Cross-Feeding',
        'subtitle': 'Acetate secreted by a central colony feeds a peripheral ring',
        'description': (
            'A glucose consumer (E_glc) is placed at the center of a 25×25 '
            'lattice. Oxygen is limited, so E_glc secretes acetate as a '
            'fermentation byproduct. Four acetate-specialist colonies '
            '(E_ac) are seeded at the lattice corners: they cannot eat '
            'glucose but have acetate uptake opened. As acetate diffuses '
            'from the center, the corner colonies light up — a classic '
            'spatial commensalism pattern driven entirely by local FBA and '
            'cross-feeding.'
        ),
        'spatial_config': {
            'models': ['textbook', 'textbook'],
            'model_ids': ['E_glc', 'E_ac'],
            'initial_placement': {
                'E_glc': [[12, 12, 2e-3]],
                'E_ac':  [[6, 12, 1e-4], [18, 12, 1e-4], [12, 6, 1e-4], [12, 18, 1e-4]],
            },
            'initial_media': _unit_salts({'glc__D': 0.5, 'o2': 3.0, 'ac': 0.0}),
            'grid': [25, 25],
            'space_width': 0.04,
            'biomass_diffusion': 3e-8,
            'media_diffusion': 3e-5,
            'diffusion_overrides': {'ac': 5e-5},
            # Globally, default vmax=10; per-species kinetic overrides let
            # the glucose consumer be O2-limited (forcing overflow / acetate
            # secretion) while the acetate consumer has abundant O2 uptake.
            'per_species_vmax': {
                'E_glc': {'o2': 4.0, 'glc__D': 10.0},
                'E_ac':  {'o2': 20.0, 'ac': 8.0},
            },
            'bound_overrides': {
                'E_ac': {
                    'glc__D': [0.0, 1000.0],
                    'ac': [-1000.0, 1000.0],
                },
            },
            'substep': 0.25,
        },
        'total_time': 20.0,
        'n_snapshots': 45,
        'color_scheme': 'rose',
        'tracked_media': ['glc__D', 'ac', 'o2'],
    },
]


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def run_simulation(cfg_entry: Dict[str, Any]):
    """Run one config, return (snapshots, runtime_s) where each snapshot is a dict."""
    core = allocate_core()
    core.register_link('DynamicFBAProcess', DynamicFBAProcess)

    interval = cfg_entry['total_time'] / cfg_entry['n_snapshots']
    proc = DynamicFBAProcess(config=cfg_entry['dfba_config'], core=core)

    t0 = _time.perf_counter()
    state0 = proc.initial_state()
    snapshots = [_snap(0.0, state0)]
    t = 0.0
    for _ in range(cfg_entry['n_snapshots']):
        result = proc.update({}, interval=interval)
        t += interval
        snapshots.append(_snap(round(t, 4), result))
    runtime = _time.perf_counter() - t0
    return snapshots, runtime


def _snap(t: float, state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'time': float(t),
        'biomass': dict(state['biomass']),
        'media': dict(state['media']),
        'growth_rates': dict(state['growth_rates']),
        'total_biomass': float(state['total_biomass']),
    }


def run_spatial_simulation(cfg_entry: Dict[str, Any]):
    """Run a spatial config, returning (snapshots, runtime_s).

    Snapshots include both the scalar aggregates and the full 2D grid
    fields for biomass (per species) and a tracked subset of media.
    """
    core = allocate_core()
    core.register_link('SpatialDynamicFBAProcess', SpatialDynamicFBAProcess)

    interval = cfg_entry['total_time'] / cfg_entry['n_snapshots']
    proc = SpatialDynamicFBAProcess(config=cfg_entry['spatial_config'], core=core)

    t0 = _time.perf_counter()
    state0 = proc.initial_state()
    snapshots = [_spatial_snap(0.0, state0, cfg_entry['tracked_media'])]
    t = 0.0
    for _ in range(cfg_entry['n_snapshots']):
        result = proc.update({}, interval=interval)
        t += interval
        snapshots.append(_spatial_snap(round(t, 4), result, cfg_entry['tracked_media']))
    runtime = _time.perf_counter() - t0
    return snapshots, runtime


def _spatial_snap(t: float, state: Dict[str, Any],
                  tracked_media: List[str]) -> Dict[str, Any]:
    """Package a spatial state into a JSON-serializable snapshot."""
    media_grid = {
        k: state['media_grid'][k]
        for k in tracked_media if k in state['media_grid']
    }
    return {
        'time': float(t),
        'biomass': dict(state['biomass']),
        'media': {k: state['media'][k] for k in tracked_media
                  if k in state['media']},
        'growth_rates': dict(state['growth_rates']),
        'total_biomass': float(state['total_biomass']),
        'biomass_grid': dict(state['biomass_grid']),
        'media_grid': media_grid,
    }


def generate_spatial_bigraph_image(cfg_entry: Dict[str, Any]) -> str:
    """Colored bigraph-viz PNG for a spatial config (as base64 data URI)."""
    from bigraph_viz import plot_bigraph
    species_ids = list(cfg_entry['spatial_config'].get('model_ids') or ['species'])
    species_str = ' + '.join(species_ids)
    grid = cfg_entry['spatial_config'].get('grid', [1, 1])

    doc = {
        'community': {
            '_type': 'process',
            'address': 'local:SpatialDynamicFBAProcess',
            'config': {'species': species_str, 'grid': f'{grid[0]}×{grid[1]}'},
            'outputs': {
                'biomass_grid': ['stores', 'biomass_grid'],
                'media_grid': ['stores', 'media_grid'],
                'total_biomass': ['stores', 'total_biomass'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                'biomass_grid': ['stores', 'biomass_grid'],
                'media_grid': ['stores', 'media_grid'],
                'total_biomass': ['stores', 'total_biomass'],
                'time': ['global_time'],
            },
        },
    }
    node_colors = {
        ('community',): '#0ea5e9',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0f2fe',
    }
    outdir = tempfile.mkdtemp(prefix='pbg_comets_sp_bg_')
    plot_bigraph(
        state=doc,
        out_dir=outdir, filename='bigraph',
        file_format='png', remove_process_place_edges=True, rankdir='LR',
        node_fill_colors=node_colors, node_label_size='16pt',
        port_labels=False, dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{data}'


def build_spatial_pbg_document(cfg_entry: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg_entry['spatial_config'])
    interval = cfg_entry['total_time'] / cfg_entry['n_snapshots']
    return make_spatial_dfba_document(
        models=cfg.get('models', []),
        grid=cfg.get('grid', [1, 1]),
        model_ids=cfg.get('model_ids'),
        initial_biomass=cfg.get('initial_biomass'),
        initial_placement=cfg.get('initial_placement'),
        initial_media=cfg.get('initial_media'),
        uniform_media=cfg.get('uniform_media', True),
        initial_media_placement=cfg.get('initial_media_placement'),
        space_width=cfg.get('space_width', 0.05),
        biomass_diffusion=cfg.get('biomass_diffusion', 1e-6),
        media_diffusion=cfg.get('media_diffusion', 5e-6),
        diffusion_overrides=cfg.get('diffusion_overrides'),
        biomass_diffusion_overrides=cfg.get('biomass_diffusion_overrides'),
        default_vmax=cfg.get('default_vmax', 10.0),
        default_km=cfg.get('default_km', 0.01),
        vmax_overrides=cfg.get('vmax_overrides'),
        km_overrides=cfg.get('km_overrides'),
        uptake_caps=cfg.get('uptake_caps'),
        bound_overrides=cfg.get('bound_overrides'),
        substep=cfg.get('substep', 0.25),
        death_rate=cfg.get('death_rate', 0.0),
        interval=interval,
    )


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------

def generate_bigraph_image(cfg_entry: Dict[str, Any]) -> str:
    """Generate a colored bigraph PNG and return a base64 data URI."""
    from bigraph_viz import plot_bigraph

    species_ids = list(cfg_entry['dfba_config'].get('model_ids') or ['species'])
    species_str = ' + '.join(species_ids)

    # Simplified doc for a clean diagram — only the key ports
    doc = {
        'community': {
            '_type': 'process',
            'address': 'local:DynamicFBAProcess',
            'config': {'species': species_str},
            'outputs': {
                'biomass': ['stores', 'biomass'],
                'media': ['stores', 'media'],
                'total_biomass': ['stores', 'total_biomass'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                'biomass': ['stores', 'biomass'],
                'media': ['stores', 'media'],
                'total_biomass': ['stores', 'total_biomass'],
                'time': ['global_time'],
            },
        },
    }

    node_colors = {
        ('community',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp(prefix='pbg_comets_bg_')
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{data}'


def build_pbg_document(cfg_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Build the full composite document dict used for the JSON tree viewer."""
    cfg = dict(cfg_entry['dfba_config'])
    interval = cfg_entry['total_time'] / cfg_entry['n_snapshots']
    doc = make_dfba_document(
        models=cfg.get('models', []),
        model_ids=cfg.get('model_ids'),
        initial_biomass=cfg.get('initial_biomass'),
        initial_media=cfg.get('initial_media'),
        volume=cfg.get('volume', 1.0),
        default_vmax=cfg.get('default_vmax', 10.0),
        default_km=cfg.get('default_km', 0.01),
        vmax_overrides=cfg.get('vmax_overrides'),
        km_overrides=cfg.get('km_overrides'),
        uptake_caps=cfg.get('uptake_caps'),
        bound_overrides=cfg.get('bound_overrides'),
        substep=cfg.get('substep', 0.1),
        death_rate=cfg.get('death_rate', 0.0),
        interval=interval,
    )
    # The config has `'models': ['textbook', ...]` — replace the string specs
    # with their human-readable form for display.
    return doc


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

COLOR_SCHEMES = {
    'indigo':  {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
                'accent': '#818cf8'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'accent': '#34d399'},
    'rose':    {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
                'accent': '#fb7185'},
}

SPECIES_PALETTE = ['#6366f1', '#10b981', '#f43f5e', '#f59e0b', '#0ea5e9', '#a855f7']
MEDIA_PALETTE   = ['#6366f1', '#10b981', '#f43f5e', '#f59e0b', '#0ea5e9',
                   '#a855f7', '#14b8a6', '#eab308']


def _spatial_field_stats(snapshots, field_key, subkey):
    """Return [vmin, vmax] (2-98 percentile) for a spatial field across snapshots."""
    import numpy as np
    arrs = []
    for s in snapshots:
        if field_key in s and subkey in s[field_key]:
            arrs.append(np.asarray(s[field_key][subkey], dtype=float))
    if not arrs:
        return [0.0, 1.0]
    flat = np.concatenate([a.ravel() for a in arrs])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return [0.0, 1.0]
    vmax = float(np.percentile(flat, 99))
    vmin = float(max(0.0, np.percentile(flat, 1)))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return [vmin, vmax]


def _spatial_section_html(idx, cfg, snapshots, runtime, all_js_data):
    """Build the HTML + JS payload for one spatial configuration."""
    sid = cfg['id']
    cs = COLOR_SCHEMES[cfg['color_scheme']]
    species_ids = list(cfg['spatial_config'].get('model_ids') or [])
    if not species_ids:
        species_ids = list(snapshots[0]['biomass'].keys())
    tracked = cfg['tracked_media']
    grid = cfg['spatial_config'].get('grid', [1, 1])
    nx, ny = grid[0], grid[1]

    times = [s['time'] for s in snapshots]
    biomass_totals = {
        sp: [s['biomass'].get(sp, 0.0) for s in snapshots]
        for sp in species_ids
    }
    growth = {
        sp: [s['growth_rates'].get(sp, 0.0) for s in snapshots]
        for sp in species_ids
    }
    media_totals = {
        m: [s['media'].get(m, 0.0) for s in snapshots]
        for m in tracked
    }
    total_biomass = [s['total_biomass'] for s in snapshots]

    # Per-field stats for colormap ranges
    biomass_ranges = {sp: _spatial_field_stats(snapshots, 'biomass_grid', sp)
                      for sp in species_ids}
    media_ranges = {m: _spatial_field_stats(snapshots, 'media_grid', m)
                    for m in tracked}

    # Per-snapshot flat grids (ny*nx for row-major order)
    def _flat_series(field_key, sub):
        return [s[field_key].get(sub, []) for s in snapshots]

    all_js_data[sid] = {
        'kind': 'spatial',
        'grid': [nx, ny],
        'species_ids': species_ids,
        'media_ids': tracked,
        'species_palette': SPECIES_PALETTE[:len(species_ids)],
        'media_palette':   MEDIA_PALETTE[:len(tracked)],
        'times': times,
        'biomass_totals': biomass_totals,
        'media_totals': media_totals,
        'growth': growth,
        'total_biomass': total_biomass,
        'biomass_grid': {sp: _flat_series('biomass_grid', sp)
                         for sp in species_ids},
        'media_grid':   {m: _flat_series('media_grid', m) for m in tracked},
        'biomass_ranges': biomass_ranges,
        'media_ranges': media_ranges,
    }

    b_init = total_biomass[0]
    b_final = total_biomass[-1]
    b_pct = f'{b_final/b_init:.1f}×' if b_init > 0 else 'N/A'

    print(f'  Generating spatial bigraph diagram for {sid}...')
    bigraph_img = generate_spatial_bigraph_image(cfg)

    species_chips = ''.join(
        f'<span class="species-chip" '
        f'style="background:{SPECIES_PALETTE[i % len(SPECIES_PALETTE)]};">'
        f'{sp}</span>'
        for i, sp in enumerate(species_ids)
    )

    # Buttons to toggle which field is displayed in the 2D viewer
    field_buttons = []
    for i, sp in enumerate(species_ids):
        field_buttons.append(
            f'<button class="field-btn" data-field="biomass:{sp}" '
            f'style="border-color:{SPECIES_PALETTE[i % len(SPECIES_PALETTE)]};'
            f'color:{SPECIES_PALETTE[i % len(SPECIES_PALETTE)]};">'
            f'{sp} biomass</button>')
    for i, m in enumerate(tracked):
        field_buttons.append(
            f'<button class="field-btn" data-field="media:{m}" '
            f'style="border-color:{MEDIA_PALETTE[i % len(MEDIA_PALETTE)]};'
            f'color:{MEDIA_PALETTE[i % len(MEDIA_PALETTE)]};">'
            f'{m}</button>')
    field_buttons_html = '\n        '.join(field_buttons)

    section = f"""
<div class="sim-section" id="sim-{sid}">
  <div class="sim-header" style="border-left:4px solid {cs['primary']};">
    <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">S{idx+1}</div>
    <div>
      <h2 class="sim-title">{cfg['title']}</h2>
      <p class="sim-subtitle">{cfg['subtitle']}</p>
    </div>
  </div>
  <p class="sim-description">{cfg['description']}</p>
  <div class="species-row">{species_chips}</div>

  <div class="metrics-row">
    <div class="metric"><span class="metric-label">Grid</span>
      <span class="metric-value">{nx}×{ny}</span>
      <span class="metric-sub">{nx*ny} cells</span></div>
    <div class="metric"><span class="metric-label">Cell size</span>
      <span class="metric-value">{cfg['spatial_config'].get('space_width', 0.05)*10:.1f} mm</span></div>
    <div class="metric"><span class="metric-label">Snapshots</span>
      <span class="metric-value">{len(snapshots)}</span>
      <span class="metric-sub">Δt {cfg['total_time']/cfg['n_snapshots']:.2f} hr</span></div>
    <div class="metric"><span class="metric-label">Biomass gain</span>
      <span class="metric-value">{b_pct}</span>
      <span class="metric-sub">{b_init:.2e} → {b_final:.2e}</span></div>
    <div class="metric"><span class="metric-label">Species</span>
      <span class="metric-value">{len(species_ids)}</span></div>
    <div class="metric"><span class="metric-label">Runtime</span>
      <span class="metric-value">{runtime:.1f}s</span></div>
  </div>

  <h3 class="subsection-title">2D Field Viewer</h3>
  <div class="field-toggle">
    {field_buttons_html}
  </div>
  <div class="heat-viewer-wrap">
    <canvas id="heat-{sid}" class="heat-canvas"></canvas>
    <div class="heat-colorbar">
      <div class="cb-title"><span id="cb-title-{sid}">biomass</span></div>
      <div class="cb-val" id="cb-max-{sid}">1.00</div>
      <div class="cb-gradient" id="cb-grad-{sid}"></div>
      <div class="cb-val" id="cb-min-{sid}">0.00</div>
    </div>
    <div class="slider-controls">
      <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};"
              onclick="toggleSpatialPlay('{sid}')" id="play-{sid}">Play</button>
      <label>Time</label>
      <input type="range" class="time-slider" id="slider-{sid}"
             min="0" max="{len(snapshots)-1}" value="0" step="1"
             style="accent-color:{cs['primary']};">
      <span class="time-val" id="tval-{sid}">t = 0.00 hr</span>
    </div>
  </div>

  <h3 class="subsection-title">Aggregate Dynamics</h3>
  <div class="charts-row">
    <div class="chart-box"><div id="chart-sbio-{sid}" class="chart"></div></div>
    <div class="chart-box"><div id="chart-smedia-{sid}" class="chart"></div></div>
  </div>

  <div class="pbg-row">
    <div class="pbg-col">
      <h3 class="subsection-title">Bigraph Architecture</h3>
      <div class="bigraph-img-wrap">
        <img src="{bigraph_img}" alt="Bigraph architecture diagram">
      </div>
    </div>
    <div class="pbg-col">
      <h3 class="subsection-title">Composite Document</h3>
      <div class="json-tree" id="json-{sid}"></div>
    </div>
  </div>
</div>
"""
    return section


def generate_html(sim_results, spatial_results, output_path):
    sections_html = []
    spatial_sections_html = []
    all_js_data: Dict[str, Any] = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        species_ids = list(snapshots[0]['biomass'].keys())
        tracked = [m for m in cfg['tracked_media']
                   if m in snapshots[0]['media']]

        times = [s['time'] for s in snapshots]
        biomass_series = {
            sp: [s['biomass'].get(sp, 0.0) for s in snapshots]
            for sp in species_ids
        }
        media_series = {
            m: [s['media'].get(m, 0.0) for s in snapshots]
            for m in tracked
        }
        growth_series = {
            sp: [s['growth_rates'].get(sp, 0.0) for s in snapshots]
            for sp in species_ids
        }
        total_biomass = [s['total_biomass'] for s in snapshots]

        all_js_data[sid] = {
            'times': times,
            'biomass': biomass_series,
            'media': media_series,
            'growth': growth_series,
            'total_biomass': total_biomass,
            'species_ids': species_ids,
            'media_ids': tracked,
            'species_palette': SPECIES_PALETTE[:len(species_ids)],
            'media_palette': MEDIA_PALETTE[:len(tracked)],
        }

        # Metrics
        b_init = total_biomass[0]
        b_final = total_biomass[-1]
        b_pct = f'{b_final/b_init:.1f}×' if b_init > 0 else 'N/A'
        glc_init = snapshots[0]['media'].get('glc__D', 0.0)
        glc_final = snapshots[-1]['media'].get('glc__D', 0.0)
        ac_final = snapshots[-1]['media'].get('ac', 0.0)
        mu_max = max(
            max(growth_series[sp]) if growth_series[sp] else 0.0
            for sp in species_ids
        )

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        species_chips = ''.join(
            f'<span class="species-chip" '
            f'style="background:{SPECIES_PALETTE[i % len(SPECIES_PALETTE)]};">'
            f'{sp}</span>'
            for i, sp in enumerate(species_ids)
        )

        section = f"""
<div class="sim-section" id="sim-{sid}">
  <div class="sim-header" style="border-left:4px solid {cs['primary']};">
    <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
    <div>
      <h2 class="sim-title">{cfg['title']}</h2>
      <p class="sim-subtitle">{cfg['subtitle']}</p>
    </div>
  </div>
  <p class="sim-description">{cfg['description']}</p>
  <div class="species-row">{species_chips}</div>

  <div class="metrics-row">
    <div class="metric"><span class="metric-label">Species</span>
      <span class="metric-value">{len(species_ids)}</span></div>
    <div class="metric"><span class="metric-label">Snapshots</span>
      <span class="metric-value">{len(snapshots)}</span></div>
    <div class="metric"><span class="metric-label">Biomass gain</span>
      <span class="metric-value">{b_pct}</span>
      <span class="metric-sub">{b_init:.2e} → {b_final:.2e} gDW</span></div>
    <div class="metric"><span class="metric-label">Glucose</span>
      <span class="metric-value">{glc_final:.2f}</span>
      <span class="metric-sub">{glc_init:.1f} → {glc_final:.2f} mmol</span></div>
    <div class="metric"><span class="metric-label">Acetate (final)</span>
      <span class="metric-value">{ac_final:.2f}</span>
      <span class="metric-sub">mmol</span></div>
    <div class="metric"><span class="metric-label">Max μ</span>
      <span class="metric-value">{mu_max:.2f}</span>
      <span class="metric-sub">1/hr</span></div>
    <div class="metric"><span class="metric-label">Runtime</span>
      <span class="metric-value">{runtime:.2f}s</span></div>
  </div>

  <h3 class="subsection-title">Community Dynamics</h3>
  <div class="charts-row">
    <div class="chart-box"><div id="chart-biomass-{sid}" class="chart"></div></div>
    <div class="chart-box"><div id="chart-stack-{sid}" class="chart"></div></div>
    <div class="chart-box"><div id="chart-growth-{sid}" class="chart"></div></div>
    <div class="chart-box"><div id="chart-media-{sid}" class="chart"></div></div>
  </div>

  <div class="pbg-row">
    <div class="pbg-col">
      <h3 class="subsection-title">Bigraph Architecture</h3>
      <div class="bigraph-img-wrap">
        <img src="{bigraph_img}" alt="Bigraph architecture diagram">
      </div>
    </div>
    <div class="pbg-col">
      <h3 class="subsection-title">Composite Document</h3>
      <div class="json-tree" id="json-{sid}"></div>
    </div>
  </div>
</div>
"""
        sections_html.append(section)

    # Spatial sections
    for idx, (cfg, (snapshots, runtime)) in enumerate(spatial_results):
        spatial_sections_html.append(
            _spatial_section_html(idx, cfg, snapshots, runtime, all_js_data))

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results + spatial_results]
    )

    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}
    pbg_docs.update({
        r[0]['id']: build_spatial_pbg_document(r[0]) for r in spatial_results
    })
    # Replace any cobra.Model or non-JSON-serializable config entries with
    # placeholder strings so the JSON tree can render them.
    pbg_docs = json.loads(json.dumps(pbg_docs, default=lambda o: str(o)))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pbg-comets — Dynamic FBA Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 45%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.35rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:820px; }}
.page-header .tag {{ display:inline-block; background:#fff; border:1px solid #e2e8f0;
                     border-radius:999px; padding:.2rem .7rem; font-size:.7rem;
                     margin-right:.4rem; color:#6366f1; font-weight:600;
                     letter-spacing:.04em; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100;
        flex-wrap:wrap; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             color:#334155; transition:all .15s; background:#fff; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1rem; max-width:880px; }}
.species-row {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:1.5rem; }}
.species-chip {{ color:white; font-family:'SF Mono',Menlo,Monaco,monospace;
                 font-size:.72rem; font-weight:600; padding:.22rem .6rem;
                 border-radius:6px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.68rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.25rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.68rem; color:#94a3b8; margin-top:.1rem; }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:300px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto;
              font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
              font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
.footer strong {{ color:#64748b; }}
.section-banner {{
  background:linear-gradient(135deg, #f0f9ff 0%, #eff6ff 100%);
  padding:2rem 3rem; border-top:1px solid #dbeafe; border-bottom:1px solid #dbeafe;
}}
.section-banner h2 {{ font-size:1.4rem; font-weight:700; color:#0c4a6e;
                      margin-bottom:.4rem; }}
.section-banner p {{ color:#475569; font-size:.9rem; max-width:860px; }}
.section-banner code {{ background:#fff; padding:.1rem .35rem; border-radius:4px;
                        color:#0ea5e9; font-size:.85em; }}
/* 2D heatmap viewer */
.field-toggle {{ display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.7rem; }}
.field-btn {{ background:#fff; border:1.5px solid; border-radius:7px;
              padding:.3rem .75rem; font-size:.78rem; font-weight:600;
              cursor:pointer; transition:all .15s; }}
.field-btn:hover {{ transform:translateY(-1px); }}
.field-btn.active {{ background:currentColor; }}
.field-btn.active {{ color:#fff !important; }}
.heat-viewer-wrap {{ position:relative; background:#0f172a; border:1px solid #e2e8f0;
                     border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.heat-canvas {{ width:100%; height:460px; display:block;
                image-rendering:pixelated; image-rendering:crisp-edges; }}
.heat-colorbar {{ position:absolute; top:.8rem; right:.8rem;
                  background:rgba(255,255,255,.94); border:1px solid #e2e8f0;
                  border-radius:8px; padding:.6rem;
                  display:flex; flex-direction:column; align-items:center; gap:.2rem;
                  backdrop-filter:blur(4px); }}
@media(max-width:900px) {{
  .charts-row, .pbg-row {{ grid-template-columns:1fr; }}
  .sim-section, .page-header, .section-banner {{ padding:1.5rem; }}
}}
/* Slider controls reuse */
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(15,23,42,.86));
                    padding:1.5rem 1.5rem 1rem; display:flex;
                    align-items:center; gap:.8rem; color:#e2e8f0; }}
.slider-controls label {{ font-size:.8rem; color:#cbd5e1; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.95rem; font-weight:600; color:#f8fafc; min-width:110px;
             text-align:right; }}
.play-btn {{ background:#fff; border:1.5px solid; padding:.3rem .8rem;
             border-radius:7px; cursor:pointer; font-size:.8rem; font-weight:600;
             transition:all .15s; }}
.play-btn:hover {{ transform:scale(1.05); }}
.cb-title {{ font-size:.62rem; text-transform:uppercase; letter-spacing:.04em;
             color:#475569; font-weight:600; }}
.cb-val {{ font-size:.7rem; color:#334155; }}
.cb-gradient {{ width:18px; height:90px; border-radius:3px;
  background:linear-gradient(to bottom,
    #fde047, #f97316, #dc2626, #7c3aed, #1e40af, #0f172a); }}
</style>
</head>
<body>

<div class="page-header">
  <span class="tag">pbg-comets</span>
  <span class="tag">Dynamic FBA</span>
  <span class="tag">process-bigraph</span>
  <h1>COMETS-Style Community Simulation Report</h1>
  <p>Three microbial community simulations wrapped as <strong>process-bigraph</strong>
  Processes via <code>DynamicFBAProcess</code>, a pure-Python dFBA
  implementation built on <code>cobra</code>. Each configuration illustrates
  a distinct ecological scenario (monoculture, competition, cross-feeding)
  with interactive time-series visualizations.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="section-banner">
  <h2>Spatial Dynamics — 2D Grid with Local FBA + Diffusion</h2>
  <p>The next configurations run on a spatial lattice using
  <code>SpatialDynamicFBAProcess</code>, which mirrors the per-cell-FBA +
  Fickian-diffusion model used by COMETS. Each cell solves its own FBA
  against local media concentrations; biomass and metabolites spread via
  an explicit Laplacian stencil between cells.</p>
</div>

{''.join(spatial_sections_html)}

<div class="footer">
  Generated by <strong>pbg-comets</strong> &mdash;
  cobra + process-bigraph &mdash;
  dynamic flux balance analysis in the style of COMETS
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') {{
    const s = Math.abs(obj) < 1e-3 && obj !== 0 ? obj.toExponential(3) :
              Number.isInteger(obj) ? obj.toString() : obj.toPrecision(6);
    return '<span class="jt-num">' + s + '</span>';
  }}
  if (typeof obj === 'string') {{
    const safe = obj.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return '<span class="jt-str">"' + safe + '"</span>';
  }}
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  const tog = el.parentElement.querySelector('.jt-toggle');
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    if (tog) tog.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    if (tog) tog.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Plotly Charts ───
const baseLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:45 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time (hr)', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
  legend:{{ font:{{ size:9 }}, bgcolor:'rgba(255,255,255,0.6)',
            bordercolor:'#e2e8f0', borderwidth:1 }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];

  // Biomass (log-ish linear lines, one per species)
  const biomassTraces = d.species_ids.map((sp, i) => ({{
    x:d.times, y:d.biomass[sp], type:'scatter', mode:'lines',
    name:sp, line:{{ color:d.species_palette[i], width:2.2 }},
  }}));
  Plotly.newPlot('chart-biomass-' + sid, biomassTraces, {{
    ...baseLayout,
    title:{{ text:'Biomass per species', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...baseLayout.yaxis, title:{{ text:'Biomass (gDW)', font:{{ size:10 }} }},
            type:'log', exponentformat:'power'}},
    showlegend:true,
  }}, pCfg);

  // Stacked area community composition (fraction of total biomass)
  const totals = d.total_biomass;
  const stackTraces = d.species_ids.map((sp, i) => ({{
    x:d.times,
    y:d.biomass[sp].map((b, k) => totals[k] > 0 ? b / totals[k] : 0),
    type:'scatter', mode:'lines', name:sp,
    stackgroup:'comp', groupnorm:'',
    line:{{ color:d.species_palette[i], width:0 }},
    fillcolor:d.species_palette[i],
  }}));
  Plotly.newPlot('chart-stack-' + sid, stackTraces, {{
    ...baseLayout,
    title:{{ text:'Community composition', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...baseLayout.yaxis, title:{{ text:'Fraction of biomass', font:{{ size:10 }} }},
            range:[0, 1.001]}},
    showlegend:true,
  }}, pCfg);

  // Growth rates
  const growthTraces = d.species_ids.map((sp, i) => ({{
    x:d.times, y:d.growth[sp], type:'scatter', mode:'lines',
    name:sp, line:{{ color:d.species_palette[i], width:2, dash:'solid' }},
  }}));
  Plotly.newPlot('chart-growth-' + sid, growthTraces, {{
    ...baseLayout,
    title:{{ text:'Specific growth rate μ(t)', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...baseLayout.yaxis, title:{{ text:'μ (1/hr)', font:{{ size:10 }} }} }},
    showlegend:true,
  }}, pCfg);

  // Media time series
  const mediaTraces = d.media_ids.map((m, i) => ({{
    x:d.times, y:d.media[m], type:'scatter', mode:'lines',
    name:m, line:{{ color:d.media_palette[i], width:2 }},
  }}));
  Plotly.newPlot('chart-media-' + sid, mediaTraces, {{
    ...baseLayout,
    title:{{ text:'Media dynamics', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...baseLayout.yaxis, title:{{ text:'Amount (mmol)', font:{{ size:10 }} }} }},
    showlegend:true,
  }}, pCfg);
}});

// ─── Spatial 2D Heatmap Viewers ───
const spatialState = {{}};  // sid -> {{ field, step, playing, intervalId }}

// Simple turbo-like colormap for biomass (dark -> yellow) and blue-to-yellow for media.
function colormapBiomass(t) {{
  t = Math.max(0, Math.min(1, t));
  // viridis-ish stops: #0f172a -> #312e81 -> #7c3aed -> #dc2626 -> #f97316 -> #fde047
  const stops = [
    [0.06,0.09,0.16], [0.19,0.18,0.51], [0.49,0.23,0.93],
    [0.86,0.15,0.15], [0.98,0.45,0.09], [0.99,0.88,0.28],
  ];
  const seg = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = stops[i], b = stops[i+1];
  return [
    (a[0] + (b[0]-a[0])*f) * 255,
    (a[1] + (b[1]-a[1])*f) * 255,
    (a[2] + (b[2]-a[2])*f) * 255,
  ];
}}
function colormapMedia(t) {{
  t = Math.max(0, Math.min(1, t));
  // dark -> teal -> white
  const stops = [
    [0.06,0.09,0.16], [0.03,0.15,0.38], [0.02,0.28,0.55],
    [0.08,0.50,0.70], [0.35,0.78,0.82], [0.92,0.97,0.99],
  ];
  const seg = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = stops[i], b = stops[i+1];
  return [
    (a[0] + (b[0]-a[0])*f) * 255,
    (a[1] + (b[1]-a[1])*f) * 255,
    (a[2] + (b[2]-a[2])*f) * 255,
  ];
}}

function drawHeatmap(sid) {{
  const d = DATA[sid];
  if (!d || d.kind !== 'spatial') return;
  const st = spatialState[sid];
  const canvas = document.getElementById('heat-' + sid);
  if (!canvas) return;
  const [nx, ny] = d.grid;
  // Pixel-scale: fit nicely into 460px height
  const cellPx = Math.floor(460 / ny);
  const W = nx * cellPx;
  const H = ny * cellPx;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);

  const [kind, name] = st.field.split(':');
  let grid, range, cmap;
  if (kind === 'biomass') {{
    grid = d.biomass_grid[name][st.step];
    range = d.biomass_ranges[name];
    cmap = colormapBiomass;
  }} else {{
    grid = d.media_grid[name][st.step];
    range = d.media_ranges[name];
    cmap = colormapMedia;
  }}
  const vmin = range[0], vmax = range[1];

  const img = ctx.createImageData(W, H);
  for (let iy = 0; iy < ny; iy++) {{
    for (let ix = 0; ix < nx; ix++) {{
      const v = grid[ix][iy];
      const t = (v - vmin) / (vmax - vmin + 1e-18);
      const [r, g, b] = cmap(t);
      // Paint a cellPx x cellPx block
      for (let py = 0; py < cellPx; py++) {{
        for (let px = 0; px < cellPx; px++) {{
          const X = ix * cellPx + px;
          const Y = iy * cellPx + py;
          const p = (Y * W + X) * 4;
          img.data[p] = r;
          img.data[p+1] = g;
          img.data[p+2] = b;
          img.data[p+3] = 255;
        }}
      }}
    }}
  }}
  ctx.putImageData(img, 0, 0);

  // Update colorbar + time text
  const cbMin = document.getElementById('cb-min-' + sid);
  const cbMax = document.getElementById('cb-max-' + sid);
  const cbGrad = document.getElementById('cb-grad-' + sid);
  const cbTitle = document.getElementById('cb-title-' + sid);
  const fmt = v => Math.abs(v) >= 0.01 || v === 0
    ? v.toFixed(2) : v.toExponential(1);
  if (cbMin) cbMin.textContent = fmt(vmin);
  if (cbMax) cbMax.textContent = fmt(vmax);
  if (cbTitle) cbTitle.textContent = kind + ':' + name;
  // Paint the gradient bar with the chosen colormap
  if (cbGrad) {{
    const stops = [];
    for (let k = 0; k <= 10; k++) {{
      const t = k / 10;
      const [r, g, b] = cmap(1 - t);  // top = max
      stops.push('rgb(' + r.toFixed(0) + ',' + g.toFixed(0) + ',' + b.toFixed(0) + ') ' + (t*100).toFixed(0) + '%');
    }}
    cbGrad.style.background = 'linear-gradient(to bottom, ' + stops.join(', ') + ')';
  }}
  // Time label
  const tv = document.getElementById('tval-' + sid);
  if (tv) tv.textContent = 't = ' + d.times[st.step].toFixed(2) + ' hr';
}}

function toggleSpatialPlay(sid) {{
  const st = spatialState[sid];
  const btn = document.getElementById('play-' + sid);
  const d = DATA[sid];
  const slider = document.getElementById('slider-' + sid);
  st.playing = !st.playing;
  if (st.playing) {{
    btn.textContent = 'Pause';
    st.intervalId = setInterval(() => {{
      st.step = (st.step + 1) % d.times.length;
      slider.value = st.step;
      drawHeatmap(sid);
    }}, 140);
  }} else {{
    btn.textContent = 'Play';
    clearInterval(st.intervalId);
  }}
}}

// Initialize each spatial viewer
Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];
  if (d.kind !== 'spatial') return;
  const firstField = d.species_ids.length > 0
    ? 'biomass:' + d.species_ids[0]
    : 'media:' + d.media_ids[0];
  spatialState[sid] = {{ field: firstField, step: 0, playing: false,
                         intervalId: null }};

  // Hook up field toggle buttons
  document.querySelectorAll('#sim-' + sid + ' .field-btn').forEach(btn => {{
    const f = btn.dataset.field;
    if (f === firstField) btn.classList.add('active');
    btn.addEventListener('click', () => {{
      document.querySelectorAll('#sim-' + sid + ' .field-btn').forEach(b =>
        b.classList.remove('active'));
      btn.classList.add('active');
      spatialState[sid].field = f;
      drawHeatmap(sid);
    }});
  }});

  // Hook up slider
  const slider = document.getElementById('slider-' + sid);
  slider.addEventListener('input', () => {{
    spatialState[sid].step = parseInt(slider.value);
    drawHeatmap(sid);
  }});

  // Aggregate charts
  const bioTraces = d.species_ids.map((sp, i) => ({{
    x:d.times, y:d.biomass_totals[sp], type:'scatter', mode:'lines',
    name:sp, line:{{ color:d.species_palette[i], width:2.2 }},
  }}));
  Plotly.newPlot('chart-sbio-' + sid, bioTraces, {{
    ...baseLayout,
    title:{{ text:'Total biomass per species', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...baseLayout.yaxis, title:{{ text:'Biomass (gDW)', font:{{ size:10 }} }} }},
    showlegend:true,
  }}, pCfg);

  const mediaTraces = d.media_ids.map((m, i) => ({{
    x:d.times, y:d.media_totals[m], type:'scatter', mode:'lines',
    name:m, line:{{ color:d.media_palette[i], width:2 }},
  }}));
  Plotly.newPlot('chart-smedia-' + sid, mediaTraces, {{
    ...baseLayout,
    title:{{ text:'Total media (across grid)', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...baseLayout.yaxis, title:{{ text:'Amount (mmol)', font:{{ size:10 }} }} }},
    showlegend:true,
  }}, pCfg);

  drawHeatmap(sid);
}});
</script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running well-mixed: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        print(f'  Runtime: {runtime:.2f}s ({len(snapshots)} snapshots)')
        sim_results.append((cfg, (snapshots, runtime)))

    spatial_results = []
    for cfg in SPATIAL_CONFIGS:
        print(f'Running spatial:  {cfg["title"]}...')
        snapshots, runtime = run_spatial_simulation(cfg)
        print(f'  Runtime: {runtime:.2f}s ({len(snapshots)} snapshots)')
        spatial_results.append((cfg, (snapshots, runtime)))

    print('Generating HTML report...')
    generate_html(sim_results, spatial_results, output_path)

    import subprocess
    subprocess.run(['open', '-a', 'Safari', output_path], check=False)


if __name__ == '__main__':
    run_demo()
