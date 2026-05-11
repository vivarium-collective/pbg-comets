"""Visualization Step subclasses for pbg-comets.

Visualizations follow the pbg-superpowers convention (v0.4.15+):
each subclass overrides ``update()`` to consume per-step state via wires
(like an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step. The composite spec wires
the input ports to store paths.

See pbg_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations

from pbg_superpowers.visualization import Visualization


class DynamicFBAPlots(Visualization):
    """Time-series HTML plot of COMETS / dFBA scalar outputs.

    Consumes ``total_biomass`` (a scalar) and ``biomass`` /
    ``growth_rates`` (maps from species id to float) at each step,
    accumulates them across calls, and emits a Plotly HTML figure on
    every update. Downstream consumers (dashboards, notebook viewers)
    read the latest 'html' from the wired store.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'COMETS dynamic FBA'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times: list[float] = []
        self.total_biomass: list[float] = []
        # Per-species histories, keyed by species id; appended in lockstep
        # with `self.times`. Missing values pad with 0.0.
        self.biomass_by_species: dict[str, list[float]] = {}
        self.growth_by_species: dict[str, list[float]] = {}

    def inputs(self):
        return {
            'total_biomass': 'float',
            'biomass': 'map[float]',
            'growth_rates': 'map[float]',
            'time': 'float',
        }

    @staticmethod
    def _append_map(history: dict[str, list[float]],
                    snapshot: dict | None,
                    n_steps: int) -> None:
        """Append the next value for each species, padding new species with
        zeros so all histories stay aligned by index."""
        snap = dict(snapshot or {})
        # New species: pad with zeros for prior steps.
        for sid in snap:
            if sid not in history:
                history[sid] = [0.0] * (n_steps - 1)
        # Append this step's value (0.0 if missing) for every known species.
        for sid in history:
            v = snap.get(sid, 0.0)
            try:
                history[sid].append(float(v) if v is not None else 0.0)
            except (TypeError, ValueError):
                history[sid].append(0.0)

    def update(self, state, interval=1.0):
        t = state.get('time')
        self.times.append(
            float(t) if t is not None
            else len(self.times) * (interval or 1.0))
        tb = state.get('total_biomass')
        try:
            self.total_biomass.append(float(tb) if tb is not None else 0.0)
        except (TypeError, ValueError):
            self.total_biomass.append(0.0)

        n = len(self.times)
        self._append_map(self.biomass_by_species, state.get('biomass'), n)
        self._append_map(self.growth_by_species, state.get('growth_rates'), n)

        title = (self.config or {}).get('title', 'COMETS dynamic FBA')

        traces = []
        # Scalar total biomass
        traces.append(
            '{"x":' + repr(self.times) + ',"y":' + repr(self.total_biomass) +
            ',"type":"scatter","mode":"lines","name":"total_biomass"}'
        )
        # Per-species biomass
        for sid, ys in self.biomass_by_species.items():
            traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines","name":"biomass:' + sid + '"}'
            )
        # Per-species growth rate (secondary visual class — same axis here for simplicity)
        for sid, ys in self.growth_by_species.items():
            traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines","name":"growth_rate:' + sid +
                '","yaxis":"y2"}'
            )

        html = (
            f'<div id="dfba" style="height:380px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("dfba",[{",".join(traces)}],'
            f'{{title:"{title}",margin:{{l:55,r:55,t:35,b:40}},'
            f'xaxis:{{title:"time (hr)"}},'
            f'yaxis:{{title:"biomass (gDW)"}},'
            f'yaxis2:{{title:"growth rate (1/hr)",overlaying:"y",side:"right"}},'
            f'legend:{{orientation:"h",y:-0.2}}}},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return {'html': html}
