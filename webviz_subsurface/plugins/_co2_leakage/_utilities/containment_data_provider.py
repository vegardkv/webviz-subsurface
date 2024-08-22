from functools import lru_cache
from typing import TypedDict, List, Union

import pandas as pd

from webviz_subsurface._providers import EnsembleTableProvider
from webviz_subsurface.plugins._co2_leakage._utilities.generic import Co2MassScale, Co2VolumeScale


class MenuOptions(TypedDict):
    zones: List[str]
    regions: List[str]
    phases: List[str]


class ContainmentDataProvider:
    def __init__(self, table_provider: EnsembleTableProvider):
        # TODO: perform validation
        self._provider = table_provider

    @property
    def realizations(self):
        return self._provider.realizations()

    def get_menu_options(self) -> MenuOptions:
        # TODO: set these on __init__ such that validation can be done as soon as possible

        col_names = self._provider.column_names()
        realization = self._provider.realizations()[0]
        df = self._provider.get_column_data(col_names, [realization])
        required_columns = ["date", "amount", "phase", "containment", "zone", "region"]
        missing_columns = [col for col in required_columns if col not in col_names]
        if len(missing_columns) > 0:
            raise KeyError(
                f"Missing expected columns {', '.join(missing_columns)}"
                f" in realization {realization} (and possibly other csv-files). "
                f"Provided files are likely from an old version of ccs-scripts."
            )
        zones = ["all"]
        for zone in list(df["zone"]):
            if zone not in zones:
                zones.append(zone)
        regions = ["all"]
        for region in list(df["region"]):
            if region not in regions:
                regions.append(region)
        if "free_gas" in list(df["phase"]):
            phases = ["total", "free_gas", "trapped_gas", "aqueous"]
        else:
            phases = ["total", "gas", "aqueous"]
        return {
            "zones": zones if len(zones) > 1 else [],
            "regions": regions if len(regions) > 1 else [],
            "phases": phases,
        }

    def extract_dataframe(
        self,
        realization: int,
        scale: Union[Co2MassScale, Co2VolumeScale]
    ) -> pd.DataFrame:
        df = self._provider.get_column_data(self._provider.column_names(), [realization])
        scale_factor = self._find_scale_factor(scale)
        if scale_factor == 1.0:
            return df
        df["amount"] /= scale_factor
        return df

    def extract_condensed_dataframe(
        self,
        co2_scale: Union[Co2MassScale, Co2VolumeScale],
    ) -> pd.DataFrame:
        df = self._provider.get_column_data(self._provider.column_names())
        df = df[(df["zone"] == "all") & (df["region"] == "all")]
        if co2_scale == Co2MassScale.MTONS:
            df["amount"] = df["amount"] / 1e9
        elif co2_scale == Co2MassScale.NORMALIZE:
            df["amount"] = df["amount"] / df["total"].max()
        return df

    @lru_cache
    def _find_scale_factor(
        self,
        scale: Union[Co2MassScale, Co2VolumeScale],
    ) -> float:
        if scale in (Co2MassScale.KG, Co2VolumeScale.CUBIC_METERS):
            return 1.0
        if scale in (Co2MassScale.MTONS, Co2VolumeScale.BILLION_CUBIC_METERS):
            return 1e9
        if scale in (Co2MassScale.NORMALIZE, Co2VolumeScale.NORMALIZE):
            df = self._provider.get_column_data(["total"])
            return df["total"].max()
        return 1.0
