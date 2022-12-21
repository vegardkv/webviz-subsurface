import dataclasses
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from dataclasses import dataclass
from typing import List

from webviz_subsurface._providers import EnsembleTableProvider
from webviz_subsurface.plugins._co2_leakage._utilities.generic import Co2Scale


def generate_summary_figure(
    table_provider: EnsembleTableProvider,
    realizations: List[int],
    scale: Co2Scale,
) -> go.Figure:
    columns = _column_subset(table_provider)
    df = _read_dataframe(table_provider, realizations, columns, scale)
    fig = go.Figure()
    showlegend = True
    for real, gf in df.groupby("realization"):
        cm = plotly.colors.qualitative.Plotly
        fig.add_scatter(
            x=gf[columns.time],
            y=gf[columns.dissolved],
            name=f"Dissolved ({columns.dissolved})",
            legendgroup="Dissolved",
            showlegend=showlegend,
            marker_color=cm[0],
        )
        fig.add_scatter(
            x=gf[columns.time],
            y=gf[columns.trapped],
            name=f"Trapped ({columns.trapped})",
            legendgroup="Trapped",
            showlegend=showlegend,
            marker_color=cm[1],
        )
        fig.add_scatter(
            x=gf[columns.time],
            y=gf[columns.mobile],
            name=f"Mobile ({columns.mobile})",
            legendgroup="Mobile",
            showlegend=showlegend,
            marker_color=cm[2],
        )
        fig.add_scatter(
            x=gf[columns.time],
            y=gf["total"],
            name="Total",
            legendgroup="Total",
            showlegend=showlegend,
            marker_color=cm[3],
        )
        showlegend = False
    fig.layout.xaxis.title = "Time"
    fig.layout.yaxis.title = f"Amount CO2 [{scale.value}]"
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.margin.b = 10
    fig.layout.margin.t = 60
    fig.layout.margin.l = 10
    fig.layout.margin.r = 10
    return fig


@dataclass
class _ColumnNames:
    time: str
    dissolved: str
    trapped: str
    mobile: str

    def values(self):
        return dataclasses.asdict(self).values()


def _read_dataframe(
    table_provider: EnsembleTableProvider,
    realizations: List[int],
    columns: _ColumnNames,
    co2_scale: Co2Scale,
) -> pd.DataFrame:
    full = pd.concat(
        [
            table_provider.get_column_data(list(columns.values()), [real]).assign(
                realization=real
            )
            for real in realizations
        ]
    )
    full["total"] = (
        full[columns.dissolved] + full[columns.trapped] + full[columns.mobile]
    )
    for c in [columns.dissolved, columns.trapped, columns.mobile, "total"]:
        if co2_scale == Co2Scale.MTONS:
            full[c] = full[c] / 1e9
        elif co2_scale == Co2Scale.NORMALIZE:
            full[c] = full[c] / full["total"].max()
    return full


def _column_subset(table_provider: EnsembleTableProvider) -> _ColumnNames:
    existing = set(table_provider.column_names())
    assert "DATE" in existing
    # Try PFLOTRAN names
    cn = _ColumnNames("DATE", "FGMDS", "FGMTR", "FGMGP")
    if set(cn.values()).issubset(existing):
        return cn
    # Try Eclipse names
    cn = _ColumnNames("DATE", "FWCD", "FGCDI", "FGCDM")
    if set(cn.values()).issubset(existing):
        return cn
    raise KeyError(f"Could not find suitable data columns among: {', '.join(existing)}")
