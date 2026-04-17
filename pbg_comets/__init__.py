"""Process-bigraph wrapper for COMETS (dynamic flux balance analysis).

Exposes two processes:

- :class:`CometsProcess`: full bridge around ``cometspy`` (requires the
  COMETS Java backend, i.e. ``COMETS_HOME`` set in the environment).
- :class:`DynamicFBAProcess`: pure-Python well-mixed dynamic FBA built on
  ``cobra``. Useful as a lightweight alternative when COMETS is not
  installed, and used by the demo report.
"""

from pbg_comets.processes import (
    CometsProcess,
    DynamicFBAProcess,
    SpatialDynamicFBAProcess,
)
from pbg_comets.composites import (
    make_dfba_document,
    make_comets_document,
    make_spatial_dfba_document,
)
from pbg_comets.types import register_comets_types

__all__ = [
    'CometsProcess',
    'DynamicFBAProcess',
    'SpatialDynamicFBAProcess',
    'make_dfba_document',
    'make_comets_document',
    'make_spatial_dfba_document',
    'register_comets_types',
]
