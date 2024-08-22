import dataclasses
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go

from webviz_subsurface._providers import EnsembleTableProvider
from webviz_subsurface.plugins._co2_leakage._utilities.containment_data_provider import ContainmentDataProvider
from webviz_subsurface.plugins._co2_leakage._utilities.generic import (
    Co2MassScale,
    Co2VolumeScale,
)


# pylint: disable=too-many-locals
def generate_summary_figure(
    table_provider_unsmry: EnsembleTableProvider,
    scale: Union[Co2MassScale, Co2VolumeScale],
    table_provider_containment: ContainmentDataProvider,
) -> go.Figure:
    columns_unsmry = _column_subset_unsmry(table_provider_unsmry)
    df_unsmry = _read_dataframe(
        table_provider_unsmry,  columns_unsmry, scale
    )
    df_containment = table_provider_containment.extract_condensed_dataframe(scale)

    r_min = min(df_unsmry.REAL)
    unsmry_last_total = df_unsmry[df_unsmry.REAL == r_min]["total"].iloc[-1]
    unsmry_last_mobile = df_unsmry[df_unsmry.REAL == r_min][columns_unsmry.mobile].iloc[
        -1
    ]
    unsmry_last_dissolved = df_unsmry[df_unsmry.REAL == r_min][
        columns_unsmry.dissolved
    ].iloc[-1]
    # TODO: expose these directly from table_provider_containment?
    containment_reference = df_containment[df_containment.REAL == r_min]
    containment_last_total = containment_reference[containment_reference["phase"] == "total"]["amount"].iloc[-1]
    containment_last_mobile = containment_reference[containment_reference["phase"] == "free_gas"]["amount"].iloc[-1]
    containment_last_dissolved = containment_reference[containment_reference["phase"] == "aqueous"]["amount"].iloc[-1]
    # ---
    last_total_err_percentage = (
        100.0 * abs(containment_last_total - unsmry_last_total) / unsmry_last_total
    )
    last_mobile_err_percentage = (
        100.0 * abs(containment_last_mobile - unsmry_last_mobile) / unsmry_last_mobile
    )
    last_dissolved_err_percentage = (
        100.0
        * abs(containment_last_dissolved - unsmry_last_dissolved)
        / unsmry_last_dissolved
    )
    last_total_err_percentage = np.round(last_total_err_percentage, 2)
    last_mobile_err_percentage = np.round(last_mobile_err_percentage, 2)
    last_dissolved_err_percentage = np.round(last_dissolved_err_percentage, 2)

    _colors = {
        'total': plotly.colors.qualitative.Plotly[3],
        'mobile': plotly.colors.qualitative.Plotly[2],
        'dissolved': plotly.colors.qualitative.Plotly[0],
        'trapped': plotly.colors.qualitative.Plotly[1],
    }

    fig = go.Figure()
    showlegend = True
    for _, sub_df in df_unsmry.groupby("realization"):
        fig.add_scatter(
            x=sub_df[columns_unsmry.time],
            y=sub_df["total"],
            name="UNSMRY",
            legendgroup="total",
            legendgrouptitle_text=f"Total ({last_total_err_percentage} %)",
            showlegend=showlegend,
            marker_color=_colors["total"],
        )
        fig.add_scatter(
            x=sub_df[columns_unsmry.time],
            y=sub_df[columns_unsmry.mobile],
            name=f"UNSMRY ({columns_unsmry.mobile})",
            legendgroup="mobile",
            legendgrouptitle_text=f"Mobile ({last_mobile_err_percentage} %)",
            showlegend=showlegend,
            marker_color=_colors["mobile"],
        )
        fig.add_scatter(
            x=sub_df[columns_unsmry.time],
            y=sub_df[columns_unsmry.dissolved],
            name=f"UNSMRY ({columns_unsmry.dissolved})",
            legendgroup="dissolved",
            legendgrouptitle_text=f"Dissolved ({last_dissolved_err_percentage} %)",
            showlegend=showlegend,
            marker_color=_colors["dissolved"],
        )
        fig.add_scatter(
            x=sub_df[columns_unsmry.time],
            y=sub_df[columns_unsmry.trapped],
            name=f"UNSMRY ({columns_unsmry.trapped})",
            legendgroup="trapped",
            legendgrouptitle_text="Trapped",
            showlegend=showlegend,
            marker_color=_colors["trapped"],
        )
        showlegend = False

    _col_names = {
        "total": "total",
        "free_gas": "mobile",
        "aqueous": "dissolved",
        "trapped_gas": "trapped",
    }

    for (real, phase), sub_df in df_containment.groupby(["REAL", "phase"]):
        fig.add_scatter(
            x=sub_df["date"],
            y=sub_df["amount"],
            name=f"Containment script ({phase})",
            legendgroup=_col_names[phase],
            showlegend=bool(real == 0),
            marker_color=_colors[_col_names[phase]],
            line_dash="dash",
        )

    fig.layout.xaxis.title = "Time"
    fig.layout.yaxis.title = f"Amount CO2 [{scale.value}]"
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.margin.b = 10
    fig.layout.margin.t = 60
    fig.layout.margin.l = 10
    fig.layout.margin.r = 10
    return fig


@dataclasses.dataclass
class _ColumnNames:
    time: str
    dissolved: str
    trapped: str
    mobile: str

    def values(self) -> Iterable[str]:
        return dataclasses.asdict(self).values()


@dataclasses.dataclass
class _ColumnNamesContainment:
    time: str
    dissolved: str
    mobile: str

    def values(self) -> Iterable[str]:
        return dataclasses.asdict(self).values()


def _read_dataframe(
    table_provider: EnsembleTableProvider,
    columns: _ColumnNames,
    co2_scale: Union[Co2MassScale, Co2VolumeScale],
) -> pd.DataFrame:
    full = pd.concat(
        [
            table_provider.get_column_data(list(columns.values()), [real]).assign(
                realization=real
            )
            for real in table_provider.realizations()
        ]
    )
    full["total"] = (
        full[columns.dissolved] + full[columns.trapped] + full[columns.mobile]
    )
    for col in [columns.dissolved, columns.trapped, columns.mobile, "total"]:
        if co2_scale == Co2MassScale.MTONS:
            full[col] = full[col] / 1e9
        elif co2_scale == Co2MassScale.NORMALIZE:
            full[col] = full[col] / full["total"].max()
    return full


def _column_subset_unsmry(table_provider: EnsembleTableProvider) -> _ColumnNames:
    existing = set(table_provider.column_names())
    assert "DATE" in existing
    # Try PFLOTRAN names
    col_names = _ColumnNames("DATE", "FGMDS", "FGMTR", "FGMGP")
    if set(col_names.values()).issubset(existing):
        return col_names
    # Try Eclipse names
    col_names = _ColumnNames("DATE", "FWCD", "FGCDI", "FGCDM")
    if set(col_names.values()).issubset(existing):
        return col_names
    raise KeyError(f"Could not find suitable data columns among: {', '.join(existing)}")
