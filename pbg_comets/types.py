"""Custom bigraph-schema types for the COMETS wrapper.

The wrapper relies on built-in PBG types (``map``, ``float``, ``overwrite``).
This module is a hook for future type registrations; calling it is harmless.
"""


def register_comets_types(core):
    """Register custom types used by the COMETS processes.

    Currently a no-op — all ports use built-in bigraph-schema types
    (``map[float]``, ``overwrite[...]``). Present so that composite
    factories can uniformly call ``register_comets_types(core)`` during
    setup without branching on whether custom types exist.
    """
    return core
