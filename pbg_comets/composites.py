"""Composite document factories for COMETS-style dFBA simulations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_DEFAULT_EMIT = {
    'biomass': 'map[float]',
    'media': 'map[float]',
    'growth_rates': 'map[float]',
    'total_biomass': 'float',
    'time': 'float',
}

_SPATIAL_EMIT = {
    **_DEFAULT_EMIT,
    'biomass_grid': 'map[list]',
    'media_grid': 'map[list]',
    'growth_rate_grid': 'map[list]',
}


def _base_document(
    process_address: str,
    config: Dict[str, Any],
    interval: float,
    emit_spec: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    emit = dict(emit_spec or _DEFAULT_EMIT)
    return {
        'community': {
            '_type': 'process',
            'address': process_address,
            'config': dict(config),
            'interval': float(interval),
            'inputs': {},
            'outputs': {
                'biomass': ['stores', 'biomass'],
                'media': ['stores', 'media'],
                'growth_rates': ['stores', 'growth_rates'],
                'total_biomass': ['stores', 'total_biomass'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': emit},
            'inputs': {
                'biomass': ['stores', 'biomass'],
                'media': ['stores', 'media'],
                'growth_rates': ['stores', 'growth_rates'],
                'total_biomass': ['stores', 'total_biomass'],
                'time': ['global_time'],
            },
        },
    }


def make_dfba_document(
    models: List[Any],
    model_ids: Optional[List[str]] = None,
    initial_biomass: Optional[List[float]] = None,
    initial_media: Optional[Dict[str, float]] = None,
    volume: float = 1.0,
    default_vmax: float = 10.0,
    default_km: float = 0.01,
    vmax_overrides: Optional[Dict[str, float]] = None,
    km_overrides: Optional[Dict[str, float]] = None,
    uptake_caps: Optional[Dict[str, float]] = None,
    bound_overrides: Optional[Dict[str, Dict[str, List[float]]]] = None,
    substep: float = 0.1,
    death_rate: float = 0.0,
    interval: float = 1.0,
):
    """Build a composite document around :class:`DynamicFBAProcess`.

    See :class:`pbg_comets.processes.DynamicFBAProcess` for config semantics.
    """
    cfg: Dict[str, Any] = {
        'models': list(models),
        'model_ids': list(model_ids or []),
        'initial_biomass': list(initial_biomass or []),
        'initial_media': dict(initial_media or {}),
        'volume': float(volume),
        'default_vmax': float(default_vmax),
        'default_km': float(default_km),
        'vmax_overrides': dict(vmax_overrides or {}),
        'km_overrides': dict(km_overrides or {}),
        'uptake_caps': dict(uptake_caps or {}),
        'bound_overrides': dict(bound_overrides or {}),
        'substep': float(substep),
        'death_rate': float(death_rate),
    }
    return _base_document('local:DynamicFBAProcess', cfg, interval)


def make_spatial_dfba_document(
    models: List[Any],
    grid: List[int],
    model_ids: Optional[List[str]] = None,
    initial_biomass: Optional[List[float]] = None,
    initial_placement: Optional[Dict[str, List[List[float]]]] = None,
    initial_media: Optional[Dict[str, float]] = None,
    uniform_media: bool = True,
    initial_media_placement: Optional[Dict[str, List[List[float]]]] = None,
    space_width: float = 0.05,
    biomass_diffusion: float = 1.0e-6,
    media_diffusion: float = 5.0e-6,
    diffusion_overrides: Optional[Dict[str, float]] = None,
    biomass_diffusion_overrides: Optional[Dict[str, float]] = None,
    default_vmax: float = 10.0,
    default_km: float = 0.01,
    vmax_overrides: Optional[Dict[str, float]] = None,
    km_overrides: Optional[Dict[str, float]] = None,
    uptake_caps: Optional[Dict[str, float]] = None,
    bound_overrides: Optional[Dict[str, Dict[str, List[float]]]] = None,
    substep: float = 0.2,
    death_rate: float = 0.0,
    interval: float = 0.5,
):
    """Build a composite document around :class:`SpatialDynamicFBAProcess`.

    See :class:`pbg_comets.processes.SpatialDynamicFBAProcess` for semantics.
    The emitter collects scalar aggregates only (biomass, media totals,
    growth rates); the 2D fields live in stores and can be read directly
    from the composite state.
    """
    cfg: Dict[str, Any] = {
        'models': list(models),
        'model_ids': list(model_ids or []),
        'initial_biomass': list(initial_biomass or []),
        'initial_media': dict(initial_media or {}),
        'uniform_media': bool(uniform_media),
        'initial_media_placement': dict(initial_media_placement or {}),
        'default_vmax': float(default_vmax),
        'default_km': float(default_km),
        'vmax_overrides': dict(vmax_overrides or {}),
        'km_overrides': dict(km_overrides or {}),
        'uptake_caps': dict(uptake_caps or {}),
        'bound_overrides': dict(bound_overrides or {}),
        'substep': float(substep),
        'death_rate': float(death_rate),
        'grid': list(grid),
        'space_width': float(space_width),
        'biomass_diffusion': float(biomass_diffusion),
        'media_diffusion': float(media_diffusion),
        'diffusion_overrides': dict(diffusion_overrides or {}),
        'biomass_diffusion_overrides': dict(biomass_diffusion_overrides or {}),
        'initial_placement': dict(initial_placement or {}),
    }
    doc = _base_document(
        'local:SpatialDynamicFBAProcess', cfg, interval,
        emit_spec=_SPATIAL_EMIT)
    # Wire the grid outputs too
    doc['community']['outputs'].update({
        'biomass_grid': ['stores', 'biomass_grid'],
        'media_grid': ['stores', 'media_grid'],
        'growth_rate_grid': ['stores', 'growth_rate_grid'],
    })
    doc['emitter']['inputs'].update({
        'biomass_grid': ['stores', 'biomass_grid'],
        'media_grid': ['stores', 'media_grid'],
        'growth_rate_grid': ['stores', 'growth_rate_grid'],
    })
    return doc


def make_comets_document(
    models: List[Any],
    model_ids: Optional[List[str]] = None,
    initial_biomass: Optional[List[float]] = None,
    initial_media: Optional[Dict[str, float]] = None,
    grid: Optional[List[int]] = None,
    space_width: float = 0.02,
    time_step: float = 0.1,
    default_vmax: float = 10.0,
    default_km: float = 0.01,
    default_diff_c: float = 5.0e-6,
    extra_params: Optional[Dict[str, float]] = None,
    interval: float = 1.0,
):
    """Build a composite document around :class:`CometsProcess`.

    The resulting document can only be run if COMETS is installed on the
    host (``COMETS_HOME`` set to a valid install). See
    :class:`pbg_comets.processes.CometsProcess` for details.
    """
    cfg: Dict[str, Any] = {
        'models': list(models),
        'model_ids': list(model_ids or []),
        'initial_biomass': list(initial_biomass or []),
        'initial_media': dict(initial_media or {}),
        'grid': list(grid or [1, 1]),
        'space_width': float(space_width),
        'time_step': float(time_step),
        'default_vmax': float(default_vmax),
        'default_km': float(default_km),
        'default_diff_c': float(default_diff_c),
        'extra_params': dict(extra_params or {}),
    }
    return _base_document('local:CometsProcess', cfg, interval)
