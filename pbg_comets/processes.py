"""Processes wrapping COMETS-style dynamic FBA for process-bigraph.

Two processes are exposed:

* :class:`CometsProcess` — bridge around the ``cometspy`` package, which
  shells out to the COMETS Java engine. Requires ``COMETS_HOME`` to be set
  and the COMETS binary to be installed. Each ``update(interval)`` call
  runs COMETS for ``ceil(interval / time_step)`` cycles using the current
  biomass/media as initial conditions, then reads back the final state.

* :class:`DynamicFBAProcess` — a self-contained well-mixed dFBA
  implementation in pure Python. Uses ``cobra`` for FBA, Michaelis-Menten
  uptake kinetics, and explicit Euler biomass/media updates. Does not
  require Java or ``COMETS_HOME``. Intended for demos, smoke tests, and
  situations where a lightweight alternative is appropriate.

Both processes share the same port signature so they can be swapped:

    inputs  -> (none; initial state provided by ``initial_state()``)
    outputs -> {
        'biomass':       map[species_id -> gDW],
        'media':         map[metabolite_id -> mmol],
        'growth_rates':  map[species_id -> 1/hr],
        'total_biomass': float (sum across species),
    }
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from process_bigraph import Process


# ---------------------------------------------------------------------------
# Helpers — model loading
# ---------------------------------------------------------------------------

def _load_cobra_model(spec):
    """Load a cobra model from one of several possible specifications.

    ``spec`` may be:
      * a cobra.Model instance (returned as-is)
      * a string path ending in .xml / .xml.gz / .sbml / .json — loaded
        via the matching cobra.io reader
      * one of cobra's bundled identifiers (``'textbook'``, ``'ecoli'``,
        ``'salmonella'``) — loaded via ``cobra.io.load_model``.
    """
    import cobra
    if isinstance(spec, cobra.Model):
        return spec.copy()
    if not isinstance(spec, str):
        raise TypeError(
            f'model spec must be a cobra.Model or string path/name; got {type(spec)}'
        )
    lower = spec.lower()
    if lower.endswith('.json'):
        return cobra.io.load_json_model(spec)
    if lower.endswith(('.xml', '.sbml', '.xml.gz')):
        return cobra.io.read_sbml_model(spec)
    return cobra.io.load_model(spec)


def _exchange_map(model) -> Dict[str, Tuple[str, float, float]]:
    """Return ``{metabolite_id: (rxn_id, default_lb, default_ub)}``.

    Only single-metabolite exchange reactions are considered. The
    metabolite id has the compartment suffix stripped (e.g. ``glc__D_e``
    -> ``glc__D``) so that species with different compartment conventions
    can still share a common media dict.
    """
    out = {}
    for rxn in model.exchanges:
        mets = list(rxn.metabolites.keys())
        if len(mets) != 1:
            continue
        met = mets[0]
        out[met.id] = (rxn.id, float(rxn.lower_bound), float(rxn.upper_bound))
    return out


# ---------------------------------------------------------------------------
# DynamicFBAProcess — pure-Python dFBA
# ---------------------------------------------------------------------------

class DynamicFBAProcess(Process):
    """Well-mixed dynamic flux balance analysis Process.

    Implements the same integration loop as COMETS for a single well-mixed
    compartment: each sub-step, Michaelis-Menten uptake bounds are set from
    the current media concentrations, FBA is solved per species, and
    biomass and media are updated by explicit Euler:

        B_i(t + dt) = B_i(t) * exp(mu_i * dt)
        S_j(t + dt) = max(0, S_j(t) + sum_i v_ij * B_i(t) * dt)

    This lets the demo and tests run without requiring the COMETS Java
    backend. For spatial simulations, diffusion, or signaling, use
    :class:`CometsProcess` instead.

    Config
    ------
    models: list of cobra.Model objects or strings (paths or builtin names).
    model_ids: optional list of names for each species (defaults to the
        cobra model id, disambiguated with a suffix if repeated).
    initial_biomass: list of floats (gDW) — one per species.
    initial_media: dict of ``{metabolite_id_without_compartment: mmol}``.
    volume: float (L), used to convert between mmol and mM for kinetics.
    default_vmax, default_km: Michaelis-Menten defaults (mmol/gDW/hr, mM).
    substep: internal integration step (hr). The externally requested
        ``interval`` is broken into ``ceil(interval / substep)`` sub-steps.
    death_rate: per-hour first-order biomass decay rate (default 0).
    min_biomass: biomass below this is snapped to zero (default 1e-18).
    """

    config_schema = {
        # Models and species
        'models': {'_type': 'list', '_default': []},
        'model_ids': {'_type': 'list', '_default': []},
        'initial_biomass': {'_type': 'list', '_default': []},
        # Media
        'initial_media': {'_type': 'map[float]', '_default': {}},
        'volume': {'_type': 'float', '_default': 1.0},
        # Kinetics
        'default_vmax': {'_type': 'float', '_default': 10.0},
        'default_km': {'_type': 'float', '_default': 0.01},
        'vmax_overrides': {'_type': 'map[float]', '_default': {}},
        'km_overrides': {'_type': 'map[float]', '_default': {}},
        # Extra uptake caps: metabolite_id -> abs(lb) upper limit
        'uptake_caps': {'_type': 'map[float]', '_default': {}},
        # Per-species exchange bound overrides:
        # {species_id: {metabolite_id: [lb, ub]}}
        'bound_overrides': {'_type': 'quote', '_default': {}},
        # Integration
        'substep': {'_type': 'float', '_default': 0.1},
        'death_rate': {'_type': 'float', '_default': 0.0},
        'min_biomass': {'_type': 'float', '_default': 1e-18},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._models: List = []
        self._model_ids: List[str] = []
        self._exchanges: List[Dict[str, Tuple[str, float, float]]] = []
        self._biomass: Dict[str, float] = {}
        self._media: Dict[str, float] = {}
        self._growth_rates: Dict[str, float] = {}
        self._initialized = False

    # ---- ports ---------------------------------------------------------

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'biomass': 'overwrite[map[float]]',
            'media': 'overwrite[map[float]]',
            'growth_rates': 'overwrite[map[float]]',
            'total_biomass': 'overwrite[float]',
        }

    # ---- lifecycle -----------------------------------------------------

    def _build(self):
        if self._initialized:
            return

        cfg = self.config
        specs = cfg['models']
        if not specs:
            raise ValueError('DynamicFBAProcess requires at least one model')

        loaded = [_load_cobra_model(s) for s in specs]

        # Resolve species ids — dedupe by suffix if repeated
        ids_from_cfg = list(cfg.get('model_ids') or [])
        ids: List[str] = []
        seen: Dict[str, int] = {}
        for i, m in enumerate(loaded):
            base = ids_from_cfg[i] if i < len(ids_from_cfg) and ids_from_cfg[i] else m.id
            count = seen.get(base, 0)
            seen[base] = count + 1
            ids.append(base if count == 0 else f'{base}_{count + 1}')

        self._models = loaded
        self._model_ids = ids
        self._exchanges = [_exchange_map(m) for m in loaded]

        # Initial biomass
        bio_in = list(cfg.get('initial_biomass') or [])
        if len(bio_in) != len(ids):
            if len(bio_in) == 0:
                bio_in = [1e-3] * len(ids)
            else:
                raise ValueError(
                    f'initial_biomass length ({len(bio_in)}) must match '
                    f'number of models ({len(ids)})'
                )
        self._biomass = dict(zip(ids, (float(b) for b in bio_in)))
        self._growth_rates = {sid: 0.0 for sid in ids}

        # Initial media — start with requested plus a default pool of any
        # metabolite that a species can exchange (at 0 mmol) so later
        # secretion products appear automatically in the media dict.
        media = dict(cfg.get('initial_media') or {})
        for ex in self._exchanges:
            for met_id in ex:
                key = self._strip_compartment(met_id)
                media.setdefault(key, 0.0)
        self._media = {k: float(v) for k, v in media.items()}

        # Apply per-species static bound overrides once (e.g. force a
        # model to only secrete acetate, not consume it).
        overrides = cfg.get('bound_overrides') or {}
        for sid, model, ex_map in zip(ids, loaded, self._exchanges):
            species_overrides = overrides.get(sid, {})
            for met_key, bounds in species_overrides.items():
                rxn_id = self._find_rxn_id(met_key, ex_map)
                if rxn_id is None:
                    continue
                lb, ub = bounds
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = float(lb)
                rxn.upper_bound = float(ub)
                # persist the new lb as the new default lb
                ex_map[self._resolve_key_to_ex_id(met_key, ex_map)] = (
                    rxn_id, float(lb), float(ub))

        self._initialized = True

    # ---- kinetics helpers ---------------------------------------------

    @staticmethod
    def _strip_compartment(met_id: str) -> str:
        """Remove trailing ``_e`` (extracellular) or ``_c`` compartment tag."""
        if len(met_id) > 2 and met_id[-2] == '_' and met_id[-1] in 'ecp':
            return met_id[:-2]
        return met_id

    def _find_rxn_id(self, met_key: str, ex_map) -> Optional[str]:
        """Given a media-dict key, return the corresponding exchange rxn id."""
        ex_id = self._resolve_key_to_ex_id(met_key, ex_map)
        if ex_id is None:
            return None
        return ex_map[ex_id][0]

    def _resolve_key_to_ex_id(self, met_key: str, ex_map) -> Optional[str]:
        """Return the exchange-map key (metabolite id) matching ``met_key``."""
        if met_key in ex_map:
            return met_key
        # media uses stripped ids; exchange map uses compartmented ids
        for full_id in ex_map:
            if self._strip_compartment(full_id) == met_key:
                return full_id
        return None

    def _concentration(self, amount_mmol: float) -> float:
        """Convert mmol in volume (L) to mM."""
        return amount_mmol / max(self.config['volume'], 1e-18)

    def _uptake_bound(self, met_key: str) -> float:
        """Michaelis-Menten uptake bound v = -vmax * S / (Km + S)."""
        amount = self._media.get(met_key, 0.0)
        if amount <= 0:
            return 0.0
        conc = self._concentration(amount)
        vmax = self.config['vmax_overrides'].get(
            met_key, self.config['default_vmax'])
        km = self.config['km_overrides'].get(
            met_key, self.config['default_km'])
        flux = vmax * conc / (km + conc)
        cap = self.config['uptake_caps'].get(met_key)
        if cap is not None:
            flux = min(flux, cap)
        return -float(flux)  # negative == uptake for exchange rxns

    # ---- integration ---------------------------------------------------

    def initial_state(self):
        self._build()
        return self._read_state()

    def _read_state(self):
        total = float(sum(self._biomass.values()))
        return {
            'biomass': dict(self._biomass),
            'media': dict(self._media),
            'growth_rates': dict(self._growth_rates),
            'total_biomass': total,
        }

    def _step(self, dt: float):
        """Advance one integration step of length ``dt`` (hours)."""
        death = self.config['death_rate']
        min_b = self.config['min_biomass']
        flux_records: List[Tuple[str, Dict[str, float], float]] = []

        # 1. Solve FBA for each species with current media-derived bounds
        for sid, model, ex_map in zip(
                self._model_ids, self._models, self._exchanges):
            biomass = self._biomass[sid]
            if biomass <= min_b:
                self._growth_rates[sid] = 0.0
                continue

            # Set exchange lower bounds from MM uptake
            for full_id, (rxn_id, default_lb, _) in ex_map.items():
                key = self._strip_compartment(full_id)
                # Only tighten bounds for mets that start below zero
                # (i.e. were consumable by default in the FBA model).
                if default_lb >= 0:
                    continue
                target_lb = self._uptake_bound(key)
                # Respect the model's original lb floor
                target_lb = max(default_lb, target_lb)
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = float(target_lb)

            # Solve FBA
            try:
                sol = model.optimize()
                mu = float(sol.objective_value or 0.0)
                fluxes = sol.fluxes if sol.status == 'optimal' else None
            except Exception:
                mu = 0.0
                fluxes = None
            if not math.isfinite(mu) or mu < 0:
                mu = 0.0
            self._growth_rates[sid] = mu

            # Collect exchange fluxes keyed by media-dict key
            ex_fluxes: Dict[str, float] = {}
            if fluxes is not None:
                for full_id, (rxn_id, _, _) in ex_map.items():
                    key = self._strip_compartment(full_id)
                    ex_fluxes[key] = float(fluxes.get(rxn_id, 0.0))
            flux_records.append((sid, ex_fluxes, mu))

        # 2. Apply biomass and media updates (explicit Euler, exponential
        # growth for biomass to stay positive under larger dt)
        for sid, ex_fluxes, mu in flux_records:
            biomass = self._biomass[sid]
            # Biomass: exponential growth minus linear death
            new_biomass = biomass * math.exp((mu - death) * dt)
            if new_biomass < min_b:
                new_biomass = 0.0
            self._biomass[sid] = new_biomass

            # Media: v (mmol/gDW/hr) * B (gDW) * dt (hr) = delta mmol
            # (positive flux = secretion, negative = uptake)
            avg_biomass = 0.5 * (biomass + new_biomass)
            for key, v in ex_fluxes.items():
                delta = v * avg_biomass * dt
                self._media[key] = max(0.0, self._media.get(key, 0.0) + delta)

    def update(self, state, interval):
        self._build()
        if interval <= 0:
            return self._read_state()

        substep = float(self.config['substep'])
        n = max(1, int(math.ceil(interval / substep)))
        dt = interval / n
        for _ in range(n):
            self._step(dt)
        return self._read_state()


# ---------------------------------------------------------------------------
# SpatialDynamicFBAProcess — pure-Python 2D grid dFBA with diffusion
# ---------------------------------------------------------------------------

class SpatialDynamicFBAProcess(Process):
    """2D spatial dynamic FBA Process.

    Mirrors the spatial model used by COMETS on a single-process backend:

    * The world is a rectangular ``[nx, ny]`` grid of cuboidal cells, each
      of side length ``space_width`` (cm). Each cell holds a local biomass
      per species (gDW) and a local amount per metabolite (mmol).
    * On each sub-step, every cell with biomass above ``min_biomass`` for a
      given species solves FBA locally using Michaelis-Menten uptake bounds
      derived from its own media, then updates local biomass (exponential)
      and local media (linear) using the resulting fluxes.
    * Between reactive sub-steps, biomass and media are diffused by an
      explicit 5-point stencil diffusion step (Fickian, reflective
      boundaries) with per-species and per-metabolite diffusion constants.

    The result reproduces COMETS spatial dynamics (colony spreading,
    cross-feeding rings, competitive exclusion zones) without requiring
    the COMETS Java backend.

    Outputs include both scalar aggregates (total biomass per species,
    total media per metabolite) and the full 2D fields as nested lists so
    downstream tools / the demo report can render heatmap animations.

    Config
    ------
    Same as :class:`DynamicFBAProcess`, plus:

    grid: ``[nx, ny]`` lattice dimensions.
    space_width: cell side length (cm). Cell volume is ``space_width**3``.
    biomass_diffusion: default biomass diffusion constant (cm^2 / s).
    media_diffusion: default metabolite diffusion constant (cm^2 / s).
    diffusion_overrides: per-metabolite overrides of ``media_diffusion``.
    biomass_diffusion_overrides: per-species overrides of ``biomass_diffusion``.
    initial_placement: ``{species_id: [[x, y, biomass_gDW], ...]}`` — cells
        not listed start at zero biomass. Coordinates are integer lattice
        indices. ``initial_biomass`` is ignored when this is non-empty.
    uniform_media: if True (default), ``initial_media`` is applied to every
        cell. When False, cells start empty and the user must place media
        via ``initial_media_placement``.
    initial_media_placement: ``{metabolite_id: [[x, y, mmol], ...]}`` —
        applied on top of the uniform media layer.
    """

    config_schema = {
        # Models and species
        'models': {'_type': 'list', '_default': []},
        'model_ids': {'_type': 'list', '_default': []},
        'initial_biomass': {'_type': 'list', '_default': []},
        # Media
        'initial_media': {'_type': 'map[float]', '_default': {}},
        'uniform_media': {'_type': 'boolean', '_default': True},
        'initial_media_placement': {'_type': 'quote', '_default': {}},
        # Kinetics
        'default_vmax': {'_type': 'float', '_default': 10.0},
        'default_km': {'_type': 'float', '_default': 0.01},
        'vmax_overrides': {'_type': 'map[float]', '_default': {}},
        'km_overrides': {'_type': 'map[float]', '_default': {}},
        'per_species_vmax': {'_type': 'quote', '_default': {}},
        'per_species_km': {'_type': 'quote', '_default': {}},
        'uptake_caps': {'_type': 'map[float]', '_default': {}},
        'bound_overrides': {'_type': 'quote', '_default': {}},
        # Integration
        'substep': {'_type': 'float', '_default': 0.1},
        'death_rate': {'_type': 'float', '_default': 0.0},
        'min_biomass': {'_type': 'float', '_default': 1e-12},
        # Spatial
        'grid': {'_type': 'tuple[integer,integer]', '_default': (1, 1)},
        'space_width': {'_type': 'float', '_default': 0.05},
        'biomass_diffusion': {'_type': 'float', '_default': 1.0e-6},
        'media_diffusion': {'_type': 'float', '_default': 5.0e-6},
        'diffusion_overrides': {'_type': 'map[float]', '_default': {}},
        'biomass_diffusion_overrides': {'_type': 'map[float]', '_default': {}},
        'initial_placement': {'_type': 'quote', '_default': {}},
        # Numerics
        'diffusion_substeps': {'_type': 'integer', '_default': 1},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._models: List = []
        self._model_ids: List[str] = []
        self._exchanges: List[Dict[str, Tuple[str, float, float]]] = []
        self._media_keys: List[str] = []
        self._biomass_fields: Dict[str, np.ndarray] = {}
        self._media_fields: Dict[str, np.ndarray] = {}
        self._growth_rate_fields: Dict[str, np.ndarray] = {}
        self._initialized = False

    # ---- ports ---------------------------------------------------------

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'biomass_grid': 'overwrite[map[list]]',
            'media_grid': 'overwrite[map[list]]',
            'growth_rate_grid': 'overwrite[map[list]]',
            'biomass': 'overwrite[map[float]]',
            'media': 'overwrite[map[float]]',
            'growth_rates': 'overwrite[map[float]]',
            'total_biomass': 'overwrite[float]',
        }

    # ---- lifecycle -----------------------------------------------------

    def _build(self):
        if self._initialized:
            return

        cfg = self.config
        specs = cfg['models']
        if not specs:
            raise ValueError('SpatialDynamicFBAProcess requires at least one model')

        loaded = [_load_cobra_model(s) for s in specs]

        ids_from_cfg = list(cfg.get('model_ids') or [])
        ids: List[str] = []
        seen: Dict[str, int] = {}
        for i, m in enumerate(loaded):
            base = ids_from_cfg[i] if i < len(ids_from_cfg) and ids_from_cfg[i] else m.id
            count = seen.get(base, 0)
            seen[base] = count + 1
            ids.append(base if count == 0 else f'{base}_{count + 1}')
        self._models = loaded
        self._model_ids = ids
        self._exchanges = [_exchange_map(m) for m in loaded]

        nx, ny = int(cfg['grid'][0]), int(cfg['grid'][1])
        if nx < 1 or ny < 1:
            raise ValueError(f'grid must have positive dimensions, got {cfg["grid"]}')

        # Initial biomass fields
        placement = cfg.get('initial_placement') or {}
        bio_in = list(cfg.get('initial_biomass') or [])
        self._biomass_fields = {}
        for i, sid in enumerate(ids):
            field = np.zeros((nx, ny), dtype=float)
            if sid in placement:
                for entry in placement[sid]:
                    x, y, b = int(entry[0]), int(entry[1]), float(entry[2])
                    if 0 <= x < nx and 0 <= y < ny:
                        field[x, y] += b
            elif i < len(bio_in):
                b = float(bio_in[i])
                # Default: deposit at the center cell
                cx, cy = nx // 2, ny // 2
                field[cx, cy] = b
            self._biomass_fields[sid] = field

        # Media fields — union of requested keys and exchangeable mets
        media_seed = dict(cfg.get('initial_media') or {})
        for ex in self._exchanges:
            for met_id in ex:
                key = self._strip_compartment(met_id)
                media_seed.setdefault(key, 0.0)
        self._media_keys = list(media_seed.keys())

        self._media_fields = {}
        uniform = bool(cfg.get('uniform_media', True))
        media_placement = cfg.get('initial_media_placement') or {}
        for key in self._media_keys:
            field = np.full((nx, ny), float(media_seed[key]), dtype=float) \
                if uniform else np.zeros((nx, ny), dtype=float)
            if key in media_placement:
                for entry in media_placement[key]:
                    x, y, v = int(entry[0]), int(entry[1]), float(entry[2])
                    if 0 <= x < nx and 0 <= y < ny:
                        field[x, y] += v
            self._media_fields[key] = field

        self._growth_rate_fields = {
            sid: np.zeros((nx, ny), dtype=float) for sid in ids
        }

        # Apply static per-species bound overrides once
        overrides = cfg.get('bound_overrides') or {}
        for sid, model, ex_map in zip(ids, loaded, self._exchanges):
            species_overrides = overrides.get(sid, {})
            for met_key, bounds in species_overrides.items():
                ex_id = self._resolve_key_to_ex_id(met_key, ex_map)
                if ex_id is None:
                    continue
                rxn_id, _, _ = ex_map[ex_id]
                lb, ub = bounds
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = float(lb)
                rxn.upper_bound = float(ub)
                ex_map[ex_id] = (rxn_id, float(lb), float(ub))

        self._initialized = True

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _strip_compartment(met_id: str) -> str:
        if len(met_id) > 2 and met_id[-2] == '_' and met_id[-1] in 'ecp':
            return met_id[:-2]
        return met_id

    def _resolve_key_to_ex_id(self, met_key: str, ex_map) -> Optional[str]:
        if met_key in ex_map:
            return met_key
        for full_id in ex_map:
            if self._strip_compartment(full_id) == met_key:
                return full_id
        return None

    def _cell_volume(self) -> float:
        return float(self.config['space_width']) ** 3 * 1000.0  # cm^3 -> L (1 cm^3 = 1 mL = 1e-3 L, so *1e-3); actually wait

    # ---- integration ---------------------------------------------------

    def initial_state(self):
        self._build()
        return self._read_state()

    def _read_state(self):
        biomass_totals = {
            sid: float(np.sum(field))
            for sid, field in self._biomass_fields.items()
        }
        media_totals = {
            key: float(np.sum(field))
            for key, field in self._media_fields.items()
        }
        # Mean growth rate weighted by local biomass
        growth_means: Dict[str, float] = {}
        for sid, field in self._growth_rate_fields.items():
            b = self._biomass_fields[sid]
            btot = np.sum(b)
            growth_means[sid] = float(
                np.sum(field * b) / btot) if btot > 0 else 0.0

        return {
            'biomass_grid': {sid: f.tolist()
                             for sid, f in self._biomass_fields.items()},
            'media_grid':   {key: f.tolist()
                             for key, f in self._media_fields.items()},
            'growth_rate_grid': {sid: f.tolist()
                                 for sid, f in self._growth_rate_fields.items()},
            'biomass': biomass_totals,
            'media': media_totals,
            'growth_rates': growth_means,
            'total_biomass': float(sum(biomass_totals.values())),
        }

    def _cell_L(self) -> float:
        """Local cell volume in liters (space_width^3 cm^3 = *1e-3 L)."""
        w = float(self.config['space_width'])
        return (w ** 3) * 1e-3

    def _local_uptake_bound(
        self, sid: str, met_key: str, local_amount: float,
    ) -> float:
        if local_amount <= 0:
            return 0.0
        volume_L = self._cell_L()
        conc_mM = local_amount / max(volume_L, 1e-18)
        per_species_v = (self.config.get('per_species_vmax') or {}).get(sid, {})
        per_species_k = (self.config.get('per_species_km') or {}).get(sid, {})
        vmax = per_species_v.get(
            met_key,
            self.config['vmax_overrides'].get(met_key, self.config['default_vmax']),
        )
        km = per_species_k.get(
            met_key,
            self.config['km_overrides'].get(met_key, self.config['default_km']),
        )
        flux = vmax * conc_mM / (km + conc_mM)
        cap = self.config['uptake_caps'].get(met_key)
        if cap is not None:
            flux = min(flux, cap)
        return -float(flux)

    def _react_step(self, dt: float):
        """Apply one reactive Euler step per cell (no diffusion).

        Uptake bounds are computed from Michaelis-Menten kinetics AND then
        capped by per-step mass-balance: a cell with biomass ``b`` cannot
        consume more than ``local_amount / (b * dt)`` mmol per gDW per hr
        of any substrate in a single step. Without this cap, MM saturation
        at physically absurd local concentrations (single cells are ~4e-8 L)
        lets FBA "over-consume" relative to the true local substrate pool;
        diffusion then replenishes, and biomass grows unbounded.
        """
        cfg = self.config
        death = cfg['death_rate']
        min_b = cfg['min_biomass']
        nx, ny = cfg['grid'][0], cfg['grid'][1]

        for sid, model, ex_map in zip(
                self._model_ids, self._models, self._exchanges):
            field = self._biomass_fields[sid]
            mu_field = self._growth_rate_fields[sid]

            # Which cells actually have biomass worth solving for?
            active = np.argwhere(field > min_b)
            if len(active) == 0:
                mu_field.fill(0.0)
                continue

            for x, y in active:
                b = float(field[x, y])

                # Set MM uptake bounds from local media — capped by
                # mass-balance so no cell consumes more than is present.
                budget_denom = b * dt if (b > 0 and dt > 0) else 0.0
                for ex_id, (rxn_id, default_lb, _) in ex_map.items():
                    if default_lb >= 0:
                        continue
                    key = self._strip_compartment(ex_id)
                    local_amount = float(self._media_fields[key][x, y])
                    mm_lb = self._local_uptake_bound(sid, key, local_amount)
                    target_lb = max(default_lb, mm_lb)
                    if budget_denom > 0 and local_amount > 0:
                        max_consumable_rate = local_amount / budget_denom
                        if -target_lb > max_consumable_rate:
                            target_lb = -max_consumable_rate
                    elif budget_denom > 0 and local_amount <= 0:
                        target_lb = 0.0
                    rxn = model.reactions.get_by_id(rxn_id)
                    rxn.lower_bound = float(target_lb)

                # Solve FBA
                try:
                    sol = model.optimize()
                    mu = float(sol.objective_value or 0.0)
                    fluxes = sol.fluxes if sol.status == 'optimal' else None
                except Exception:
                    mu = 0.0
                    fluxes = None
                if not math.isfinite(mu) or mu < 0:
                    mu = 0.0
                mu_field[x, y] = mu

                # Update biomass
                new_b = b * math.exp((mu - death) * dt)
                if new_b < min_b:
                    new_b = 0.0
                avg_b = 0.5 * (b + new_b)
                field[x, y] = new_b

                # Update local media
                if fluxes is None:
                    continue
                for ex_id, (rxn_id, _, _) in ex_map.items():
                    key = self._strip_compartment(ex_id)
                    v = float(fluxes.get(rxn_id, 0.0))
                    if v == 0.0:
                        continue
                    delta = v * avg_b * dt
                    cur = float(self._media_fields[key][x, y])
                    self._media_fields[key][x, y] = max(0.0, cur + delta)

            # Cells without biomass have zero mu
            mask = field <= min_b
            mu_field[mask] = 0.0

    def _diffuse_step(self, field: np.ndarray, D: float, dt: float):
        """In-place explicit 5-point Laplacian diffusion with reflective boundaries.

        Stability: dt * D / dx**2 must be <= 0.25 for 2D explicit scheme.
        Diffusion is split into ``diffusion_substeps`` inner iterations to
        satisfy the stability condition when the outer reactive ``dt`` is
        large.
        """
        if D <= 0:
            return
        nx, ny = field.shape
        if nx == 1 and ny == 1:
            return
        dx = float(self.config['space_width'])
        # dx in cm; for D in cm^2/s, convert dt (hr) to s
        dt_s = dt * 3600.0
        r = D * dt_s / (dx * dx)
        # stability: r <= 0.25 for 2D explicit; sub-step if necessary
        n_steps = max(1, int(math.ceil(r / 0.22)))
        r_sub = r / n_steps
        for _ in range(n_steps):
            # 5-point Laplacian with reflective boundaries (np.pad edge)
            padded = np.pad(field, 1, mode='edge')
            lap = (
                padded[:-2, 1:-1] + padded[2:, 1:-1]
                + padded[1:-1, :-2] + padded[1:-1, 2:]
                - 4.0 * field
            )
            field += r_sub * lap

    def _apply_diffusion(self, dt: float):
        """Diffuse all biomass and media fields by ``dt``."""
        D_b_default = float(self.config['biomass_diffusion'])
        D_b_overrides = self.config.get('biomass_diffusion_overrides') or {}
        for sid, field in self._biomass_fields.items():
            D = float(D_b_overrides.get(sid, D_b_default))
            self._diffuse_step(field, D, dt)

        D_m_default = float(self.config['media_diffusion'])
        D_m_overrides = self.config.get('diffusion_overrides') or {}
        for key, field in self._media_fields.items():
            D = float(D_m_overrides.get(key, D_m_default))
            self._diffuse_step(field, D, dt)

    def update(self, state, interval):
        self._build()
        if interval <= 0:
            return self._read_state()
        substep = float(self.config['substep'])
        n = max(1, int(math.ceil(interval / substep)))
        dt = interval / n
        for _ in range(n):
            self._react_step(dt)
            self._apply_diffusion(dt)
        return self._read_state()


# ---------------------------------------------------------------------------
# CometsProcess — bridge around cometspy
# ---------------------------------------------------------------------------

class CometsProcess(Process):
    """Bridge Process wrapping ``cometspy`` around the COMETS Java engine.

    Each ``update(interval)`` call:

    1. Writes the current biomass and media into a fresh ``cometspy.layout``.
    2. Sets ``maxCycles = ceil(interval / time_step)``.
    3. Runs COMETS via ``cometspy.comets(layout, params).run()``.
    4. Reads the final ``total_biomass`` and media from the COMETS output
       DataFrames, stores them internally for the next call, and returns
       them as ``overwrite`` outputs.

    Requires the COMETS Java binary to be installed and the ``COMETS_HOME``
    environment variable to point to its install directory. If those are
    not available, building the Process raises :class:`RuntimeError` the
    first time ``update()`` is called.

    Config
    ------
    Same keys as :class:`DynamicFBAProcess` plus:

    grid: list[int] of size 2 — lattice dimensions (default ``[1, 1]``).
    space_width: float (cm) — lattice box side length.
    time_step: float (hr) — COMETS internal step.
    default_diff_c: float — default metabolite diffusion constant (cm2/s).
    max_cycles_per_call: int — safety cap on maxCycles per ``update()``.
    """

    config_schema = {
        # Species
        'models': {'_type': 'list', '_default': []},
        'model_ids': {'_type': 'list', '_default': []},
        'initial_biomass': {'_type': 'list', '_default': []},
        # Media
        'initial_media': {'_type': 'map[float]', '_default': {}},
        # Kinetics
        'default_vmax': {'_type': 'float', '_default': 10.0},
        'default_km': {'_type': 'float', '_default': 0.01},
        # Spatial
        'grid': {'_type': 'tuple[integer,integer]', '_default': (1, 1)},
        'space_width': {'_type': 'float', '_default': 0.02},
        'default_diff_c': {'_type': 'float', '_default': 5.0e-6},
        # Stepping
        'time_step': {'_type': 'float', '_default': 0.1},
        'max_cycles_per_call': {'_type': 'integer', '_default': 10000},
        # Passthrough tunables (forwarded into cometspy params)
        'extra_params': {'_type': 'map[float]', '_default': {}},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._model_ids: List[str] = []
        self._cobra_models: List = []
        self._biomass: Dict[str, float] = {}
        self._media: Dict[str, float] = {}
        self._growth_rates: Dict[str, float] = {}
        self._initialized = False
        self._work_dir: Optional[str] = None

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'biomass': 'overwrite[map[float]]',
            'media': 'overwrite[map[float]]',
            'growth_rates': 'overwrite[map[float]]',
            'total_biomass': 'overwrite[float]',
        }

    # ---- setup ---------------------------------------------------------

    def _require_comets_home(self):
        home = os.environ.get('COMETS_HOME')
        if not home or not os.path.isdir(home):
            raise RuntimeError(
                'CometsProcess requires the COMETS Java engine. '
                'Set the COMETS_HOME environment variable to your COMETS '
                'install directory (see https://www.runcomets.org/get-started). '
                'For a COMETS-free alternative, use DynamicFBAProcess.'
            )

    def _build(self):
        if self._initialized:
            return
        self._require_comets_home()

        import tempfile
        cfg = self.config
        specs = cfg['models']
        if not specs:
            raise ValueError('CometsProcess requires at least one model')

        self._cobra_models = [_load_cobra_model(s) for s in specs]

        ids_from_cfg = list(cfg.get('model_ids') or [])
        ids: List[str] = []
        seen: Dict[str, int] = {}
        for i, m in enumerate(self._cobra_models):
            base = ids_from_cfg[i] if i < len(ids_from_cfg) and ids_from_cfg[i] else m.id
            count = seen.get(base, 0)
            seen[base] = count + 1
            ids.append(base if count == 0 else f'{base}_{count + 1}')
        self._model_ids = ids

        bio = list(cfg.get('initial_biomass') or [1e-6] * len(ids))
        self._biomass = dict(zip(ids, (float(b) for b in bio)))
        self._media = {k: float(v) for k, v in (cfg.get('initial_media') or {}).items()}
        self._growth_rates = {sid: 0.0 for sid in ids}

        self._work_dir = tempfile.mkdtemp(prefix='pbg_comets_')
        self._initialized = True

    # ---- step ----------------------------------------------------------

    def initial_state(self):
        # Do NOT call _build here — we want initial_state() to work even
        # when COMETS is not installed (it only reports the config-specified
        # initial biomass/media). _build() is deferred until update().
        cfg = self.config
        specs = cfg['models']
        ids_from_cfg = list(cfg.get('model_ids') or [])
        ids = []
        seen: Dict[str, int] = {}
        # Determine ids without loading cobra models
        for i, s in enumerate(specs):
            if i < len(ids_from_cfg) and ids_from_cfg[i]:
                base = ids_from_cfg[i]
            elif hasattr(s, 'id'):
                base = s.id
            else:
                base = str(s)
            count = seen.get(base, 0)
            seen[base] = count + 1
            ids.append(base if count == 0 else f'{base}_{count + 1}')
        bio = list(cfg.get('initial_biomass') or [1e-6] * len(ids))
        biomass = dict(zip(ids, (float(b) for b in bio)))
        media = {k: float(v) for k, v in (cfg.get('initial_media') or {}).items()}
        return {
            'biomass': biomass,
            'media': media,
            'growth_rates': {sid: 0.0 for sid in ids},
            'total_biomass': float(sum(biomass.values())),
        }

    def update(self, state, interval):
        self._build()
        cfg = self.config
        if interval <= 0:
            return self._read_state()

        import cometspy as cs

        # Build fresh cometspy models and layout from current state
        cs_models = []
        for sid, cobra_model in zip(self._model_ids, self._cobra_models):
            cm = cs.model(cobra_model)
            cm.id = sid
            # Open exchanges so media controls uptake
            if hasattr(cm, 'open_exchanges'):
                cm.open_exchanges()
            # Place all biomass at center of grid
            cx = cfg['grid'][0] // 2
            cy = cfg['grid'][1] // 2
            cm.initial_pop = [[cx, cy, float(self._biomass[sid])]]
            cs_models.append(cm)

        layout = cs.layout(cs_models)
        layout.grid = list(cfg['grid'])
        layout.default_diff_c = float(cfg['default_diff_c'])
        for met_id, amount in self._media.items():
            if amount > 0:
                layout.set_specific_metabolite(met_id, float(amount))

        params = cs.params()
        n_cycles = max(1, int(math.ceil(interval / float(cfg['time_step']))))
        n_cycles = min(n_cycles, int(cfg['max_cycles_per_call']))
        params.set_param('timeStep', float(cfg['time_step']))
        params.set_param('maxCycles', n_cycles)
        params.set_param('spaceWidth', float(cfg['space_width']))
        params.set_param('defaultVmax', float(cfg['default_vmax']))
        params.set_param('defaultKm', float(cfg['default_km']))
        params.set_param('writeTotalBiomassLog', True)
        params.set_param('writeMediaLog', True)
        params.set_param('MediaLogRate', n_cycles)
        for k, v in (cfg.get('extra_params') or {}).items():
            params.set_param(k, v)

        sim = cs.comets(layout, params, relative_dir=os.path.basename(
            self._work_dir or '.'))
        sim.run()

        # Read final biomass row
        tb = getattr(sim, 'total_biomass', None)
        if tb is not None and len(tb) > 0:
            final_row = tb.iloc[-1]
            for sid in self._model_ids:
                if sid in final_row.index:
                    self._biomass[sid] = float(final_row[sid])
        # Growth rates estimated from last two rows
        if tb is not None and len(tb) >= 2:
            prev = tb.iloc[-2]
            curr = tb.iloc[-1]
            for sid in self._model_ids:
                if sid in curr.index and sid in prev.index:
                    p, c = float(prev[sid]), float(curr[sid])
                    if p > 0 and c > 0:
                        self._growth_rates[sid] = math.log(c / p) / max(
                            cfg['time_step'], 1e-9)
                    else:
                        self._growth_rates[sid] = 0.0

        # Read final media row
        md = getattr(sim, 'media', None)
        if md is not None and len(md) > 0:
            final_cycle = md['cycle'].max()
            final = md[md['cycle'] == final_cycle]
            summed = final.groupby('metabolite')['conc_mmol'].sum()
            for met, val in summed.items():
                self._media[met] = float(val)

        return self._read_state()

    def _read_state(self):
        return {
            'biomass': dict(self._biomass),
            'media': dict(self._media),
            'growth_rates': dict(self._growth_rates),
            'total_biomass': float(sum(self._biomass.values())),
        }

    def __del__(self):
        try:
            import shutil
            if self._work_dir and os.path.isdir(self._work_dir):
                shutil.rmtree(self._work_dir, ignore_errors=True)
        except Exception:
            pass
