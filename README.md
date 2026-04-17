# pbg-comets

Process-bigraph wrapper for [COMETS](https://github.com/segrelab/comets) (Computation Of Microbial Ecosystems in Time and Space) and its Python interface [cometspy](https://github.com/segrelab/cometspy). Also ships a pure-Python dynamic-FBA process that works without the COMETS Java backend, for lightweight simulations and self-contained demos.

## What it does

Exposes three `process-bigraph` Processes that run microbial community simulations at the level of **constraint-based metabolism** — genome-scale FBA coupled to time-dependent biomass and media dynamics:

- **`CometsProcess`** — bridge around `cometspy`. Each `update(interval)` packages the current biomass and media into a fresh `cometspy.layout`, sets `maxCycles = ceil(interval / time_step)`, runs the COMETS Java engine via `cometspy.comets(...).run()`, and reads the final state back out. Supports COMETS's full spatial grid, diffusion, signaling, and metabolite kinetics. Requires a local COMETS install and `COMETS_HOME` set in the environment.

- **`DynamicFBAProcess`** — pure-Python well-mixed dynamic FBA, built on top of `cobra`. Each `update(interval)` sub-steps an explicit-Euler loop that, at each step, sets Michaelis-Menten uptake bounds from the current media, solves FBA per species, and updates biomass (exponential) and media (linear) using the resulting fluxes. No Java dependency.

- **`SpatialDynamicFBAProcess`** — pure-Python 2D-grid dFBA with Fickian diffusion, mirroring the spatial model used by COMETS. Each cell solves its own FBA against local media; biomass and metabolites diffuse between neighbouring cells via an explicit 5-point Laplacian stencil. Produces the same time-series outputs as `DynamicFBAProcess` plus 2D `biomass_grid` / `media_grid` fields that the demo renders as interactive heatmap animations.

The three processes share the same scalar port signature so they can be swapped; the spatial variant adds three additional grid ports.

## Installation

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# Demo extras (Plotly + bigraph-viz)
uv pip install -e '.[dev]'
```

**Optional — to use `CometsProcess`:** install the COMETS Java engine from <https://www.runcomets.org/get-started> and set `COMETS_HOME` to the install directory.

## Quick Start

### Self-contained dynamic FBA (no COMETS install needed)

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter, gather_emitter_results
from pbg_comets import DynamicFBAProcess, make_dfba_document

core = allocate_core()
core.register_link('DynamicFBAProcess', DynamicFBAProcess)
core.register_link('ram-emitter', RAMEmitter)

doc = make_dfba_document(
    models=['textbook'],             # cobra's bundled e_coli_core
    model_ids=['E_coli'],
    initial_biomass=[5e-4],          # gDW
    initial_media={
        'glc__D': 15.0, 'o2': 10.0, 'nh4': 1000.0, 'pi': 1000.0,
        'h2o': 1000.0, 'co2': 1000.0, 'h': 1000.0, 'ac': 0.0,
    },
    volume=1.0,                      # L
    substep=0.1,                     # integration sub-step in hr
    interval=0.2,                    # PBG update interval in hr
)

sim = Composite({'state': doc}, core=core)
sim.run(12.0)                        # hr

results = gather_emitter_results(sim)
frames = list(results.values())[0]
for f in frames[::10]:
    print(f"t={f['time']:5.2f}  B={f['total_biomass']:.3e}  "
          f"glc={f['media']['glc__D']:5.2f}  ac={f['media']['ac']:5.2f}")
```

### Using the real COMETS engine

```python
import os
os.environ.setdefault('COMETS_HOME', '/path/to/comets_2.12.5')

from pbg_comets import CometsProcess, make_comets_document
# same shape as make_dfba_document, plus grid / space_width / time_step
doc = make_comets_document(
    models=['textbook'],
    initial_biomass=[1e-6],
    initial_media={'glc__D': 10.0, 'o2': 20.0},
    grid=[5, 5],
    space_width=0.02,
    time_step=0.1,
    interval=1.0,
)
```

## API Reference

### `DynamicFBAProcess` — pure-Python dFBA

| Config | Type | Default | Description |
|--------|------|---------|-------------|
| `models` | list | `[]` | cobra.Model objects, SBML paths, or builtin names (`'textbook'`, `'ecoli'`, ...) |
| `model_ids` | list | `[]` | Optional species ids (defaults to cobra model id, de-duped if repeated) |
| `initial_biomass` | list[float] | `[1e-3]×n` | Starting biomass per species (gDW) |
| `initial_media` | map[float] | `{}` | Starting amounts keyed by metabolite id without compartment (e.g. `glc__D`) |
| `volume` | float | `1.0` | Compartment volume (L) — converts mmol ↔ mM for kinetics |
| `default_vmax` | float | `10.0` | MM Vmax applied to any metabolite without an override (mmol/gDW/hr) |
| `default_km` | float | `0.01` | MM Km applied to any metabolite without an override (mM) |
| `vmax_overrides`, `km_overrides` | map[float] | `{}` | Per-metabolite kinetic overrides |
| `uptake_caps` | map[float] | `{}` | Hard caps on metabolite uptake rate |
| `bound_overrides` | map[map[list]] | `{}` | Per-species exchange bound overrides `{species: {met: [lb, ub]}}` |
| `substep` | float | `0.1` | Internal Euler sub-step (hr) |
| `death_rate` | float | `0.0` | First-order biomass decay (1/hr) |
| `min_biomass` | float | `1e-18` | Biomass floor (below this, snap to zero) |

### `SpatialDynamicFBAProcess` — 2D grid dFBA with diffusion

Same config keys as `DynamicFBAProcess` plus:

| Config | Type | Default | Description |
|--------|------|---------|-------------|
| `grid` | `(int, int)` | `(1, 1)` | Lattice dimensions (nx, ny) |
| `space_width` | float | `0.05` | Cell side length (cm) |
| `biomass_diffusion` | float | `1e-6` | Default biomass diffusion constant (cm²/s) |
| `media_diffusion` | float | `5e-6` | Default metabolite diffusion constant (cm²/s) |
| `diffusion_overrides` | map[float] | `{}` | Per-metabolite diffusion overrides |
| `biomass_diffusion_overrides` | map[float] | `{}` | Per-species biomass diffusion overrides |
| `initial_placement` | `{species: [[x, y, gDW], …]}` | `{}` | Seed biomass at specific lattice cells |
| `uniform_media` | bool | `True` | Apply `initial_media` uniformly to every cell |
| `initial_media_placement` | `{met: [[x, y, mmol], …]}` | `{}` | Add media to specific cells on top of uniform |
| `per_species_vmax`, `per_species_km` | map[map[float]] | `{}` | Per-species kinetic overrides (e.g. `{'E_ac': {'o2': 20.0}}`) |

**Additional output ports** (on top of the scalar ports):

| Port | Type | Description |
|------|------|-------------|
| `biomass_grid` | `overwrite[map[list]]` | `{species: 2D nested list of cell biomass}` |
| `media_grid` | `overwrite[map[list]]` | `{metabolite: 2D nested list of cell amounts}` |
| `growth_rate_grid` | `overwrite[map[list]]` | `{species: 2D nested list of μ per cell}` |

### `CometsProcess` — bridge to COMETS Java

Same config keys as `DynamicFBAProcess` (except the dFBA-only knobs) plus:

| Config | Type | Default | Description |
|--------|------|---------|-------------|
| `grid` | list[int,int] | `[1,1]` | Lattice dimensions (nx, ny) |
| `space_width` | float | `0.02` | Lattice box side length (cm) |
| `time_step` | float | `0.1` | COMETS internal step (hr) |
| `default_diff_c` | float | `5e-6` | Default metabolite diffusion constant (cm²/s) |
| `max_cycles_per_call` | int | `10000` | Safety cap on cycles per `update()` |
| `extra_params` | map[float] | `{}` | Passthrough entries into `cometspy.params` |

### Shared output ports (both processes)

| Port | Type | Description |
|------|------|-------------|
| `biomass` | `overwrite[map[float]]` | Biomass per species (gDW) |
| `media` | `overwrite[map[float]]` | Amount per metabolite (mmol) |
| `growth_rates` | `overwrite[map[float]]` | Per-species specific growth rate μ (1/hr) |
| `total_biomass` | `overwrite[float]` | Sum across species (gDW) |

## Architecture

```
Composite
├── community (DynamicFBAProcess | CometsProcess)
│   └── internally manages:
│       ├── list of cobra.Model instances (one per species)
│       ├── current biomass dict
│       └── current media dict
├── stores/
│   ├── biomass        (species → gDW)
│   ├── media          (metabolite → mmol)
│   ├── growth_rates   (species → 1/hr)
│   └── total_biomass  (float)
└── emitter (RAMEmitter)
    └── records biomass, media, μ, and total_biomass vs time
```

The bridge pattern: on each `update(state, interval)`, the Process solves FBA per species using the current media, advances biomass and media forward in time, and writes the new values to its output ports as `overwrite` (absolute replacement, not deltas).

## Demo

```bash
python demo/demo_report.py
```

Runs **six** configurations and opens an interactive HTML report (`demo/report.html`) in Safari:

- Three **well-mixed** dFBA examples — monoculture on glucose, two-species competition, and cross-feeding — shown with Plotly time-series charts for biomass, community composition, growth rates, and media.
- Three **spatial** dFBA examples on a 25×25 lattice — single-colony spreading, spatial two-colony competition, and spatial cross-feeding — each rendered as a **2D heatmap animation** (toggle between species biomass and metabolite fields, scrub with a time slider, play/pause) plus Plotly aggregate charts.

Every section also includes a colored bigraph architecture diagram and a collapsible view of the full PBG composite document.

## Tests

```bash
pytest -v
```

Tests that require the COMETS Java engine (`test_comets_runs_end_to_end`) are automatically skipped when `COMETS_HOME` is unset. The rest run offline using `cobra`'s bundled `e_coli_core` model.

## References

- **COMETS paper**: Dukovski I. *et al.*, "A metabolic modeling platform for the Computation Of Microbial Ecosystems in Time and Space (COMETS)", *Nature Protocols* 16, 5030–5082 (2021). <https://doi.org/10.1038/s41596-021-00593-3>
- **COMETS**: <https://www.runcomets.org/> — Segrè Lab, Boston University
- **cometspy**: <https://github.com/segrelab/cometspy>
- **cobra / cobrapy**: <https://opencobra.github.io/cobrapy/>
- **process-bigraph**: <https://github.com/vivarium-collective/process-bigraph>
