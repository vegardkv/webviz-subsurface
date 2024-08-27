import glob
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from webviz_config import WebvizSettings

from webviz_subsurface._providers import (
    EnsembleSurfaceProvider,
    EnsembleSurfaceProviderFactory,
    EnsembleTableProvider,
    EnsembleTableProviderFactory,
)
from webviz_subsurface._utils.webvizstore_functions import read_csv
from webviz_subsurface.plugins._co2_leakage._utilities.containment_data_provider import (
    ContainmentDataProvider
)
from webviz_subsurface.plugins._co2_leakage._utilities.generic import (
    GraphSource,
    MapAttribute,
    MenuOptions,
)
from webviz_subsurface.plugins._co2_leakage._utilities.unsmry_data_provider import (
    UnsmryDataProvider
)
from webviz_subsurface.plugins._map_viewer_fmu._tmp_well_pick_provider import (
    WellPickProvider,
)

LOGGER = logging.getLogger(__name__)
WARNING_THRESHOLD_CSV_FILE_SIZE_MB = 100.0


def init_map_attribute_names(
    mapping: Optional[Dict[str, str]]
) -> Dict[MapAttribute, str]:
    if mapping is None:
        # Based on name convention of xtgeoapp_grd3dmaps:
        return {
            MapAttribute.MIGRATION_TIME_SGAS: "migrationtime_sgas",
            MapAttribute.MIGRATION_TIME_AMFG: "migrationtime_amfg",
            MapAttribute.MAX_SGAS: "max_sgas",
            MapAttribute.MAX_AMFG: "max_amfg",
            MapAttribute.MASS: "co2-mass-total",
            MapAttribute.DISSOLVED: "co2-mass-aqu-phase",
            MapAttribute.FREE: "co2-mass-gas-phase",
        }
    return {MapAttribute[key]: value for key, value in mapping.items()}


def init_surface_providers(
    webviz_settings: WebvizSettings,
    ensembles: List[str],
) -> Dict[str, EnsembleSurfaceProvider]:
    surface_provider_factory = EnsembleSurfaceProviderFactory.instance()
    return {
        ens: surface_provider_factory.create_from_ensemble_surface_files(
            webviz_settings.shared_settings["scratch_ensembles"][ens],
        )
        for ens in ensembles
    }


def init_well_pick_provider(
    well_pick_dict: Dict[str, Optional[str]],
    map_surface_names_to_well_pick_names: Optional[Dict[str, str]],
) -> Dict[str, Optional[WellPickProvider]]:
    well_pick_provider: Dict[str, Optional[WellPickProvider]] = {}
    ensembles = list(well_pick_dict.keys())
    for ens in ensembles:
        well_pick_path = well_pick_dict[ens]
        if well_pick_path is None:
            well_pick_provider[ens] = None
        else:
            try:
                well_pick_provider[ens] = WellPickProvider(
                    read_csv(well_pick_path), map_surface_names_to_well_pick_names
                )
            except OSError:
                well_pick_provider[ens] = None
    return well_pick_provider


def init_unsmry_data_providers(
    ensemble_roots: Dict[str, str],
    table_rel_path: str,
) -> Dict[str, UnsmryDataProvider]:
    factory = EnsembleTableProviderFactory.instance()
    providers = {
        ens: _init_ensemble_table_provider(factory, ens, ens_path, table_rel_path)
        for ens, ens_path in ensemble_roots.items()
    }
    return {
        k: UnsmryDataProvider(v)
        for k, v in providers.items()
        if v is not None
    }


def init_containment_data_providers(
    ensemble_roots: Dict[str, str],
    table_rel_path: str,
) -> Dict[str, ContainmentDataProvider]:
    factory = EnsembleTableProviderFactory.instance()
    providers = {
        ens: _init_ensemble_table_provider(factory, ens, ens_path, table_rel_path)
        for ens, ens_path in ensemble_roots.items()
    }
    return {
        k: ContainmentDataProvider(v)
        for k, v in providers.items()
        if v is not None
    }


def _init_ensemble_table_provider(
    factory: EnsembleTableProviderFactory,
    ens: str,
    ens_path: str,
    table_rel_path: str,
) -> Optional[EnsembleTableProvider]:
    if ens_path.endswith(".csv"):
        max_size_mb = _find_max_file_size_mb(ens_path, table_rel_path)
        if max_size_mb > WARNING_THRESHOLD_CSV_FILE_SIZE_MB:
            text = (
                "Some CSV-files are very large and might create problems when loading."
            )
            text += f"\n  ensembles: {ens}"
            text += f"\n  CSV-files: {table_rel_path}"
            text += f"\n  Max size : {max_size_mb:.2f} MB"
            LOGGER.warning(text)

    try:
        return factory.create_from_per_realization_csv_file(
            ens_path, table_rel_path
        )
    except (KeyError, ValueError) as exc:
        try:
            return factory.create_from_per_realization_arrow_file(
                ens_path, table_rel_path
            )
        except (KeyError, ValueError) as exc2:
            LOGGER.warning(
                f'Tried reading "{table_rel_path}" for ensemble "{ens}" as csv with'
                f' error {exc}, and as arrow with error {exc2}'
            )
    return None


def _find_max_file_size_mb(ens_path: str, table_rel_path: str) -> float:
    glob_pattern = os.path.join(ens_path, table_rel_path)
    paths = glob.glob(glob_pattern)
    max_size = 0.0
    for file in paths:
        if os.path.exists(file):
            file_stats = os.stat(file)
            size_in_mb = file_stats.st_size / (1024 * 1024)
            max_size = max(max_size, size_in_mb)
    return max_size


def init_menu_options(
    ensemble_roots: Dict[str, str],
    mass_table: Dict[str, ContainmentDataProvider],
    actual_volume_table: Dict[str, ContainmentDataProvider],
    unsmry_providers: Dict[str, UnsmryDataProvider],
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    options: Dict[str, Dict[str, MenuOptions]] = {}
    for ens in ensemble_roots.keys():
        options[ens] = {
            GraphSource.CONTAINMENT_MASS: mass_table[ens].menu_options,
            GraphSource.CONTAINMENT_ACTUAL_VOLUME: actual_volume_table[ens].menu_options,
            GraphSource.UNSMRY: unsmry_providers[ens].menu_options,
        }
    return options


def process_files(
    cont_bound: Optional[str],
    haz_bound: Optional[str],
    well_file: Optional[str],
    ensemble_paths: Dict[str, str],
) -> List[Dict[str, Optional[str]]]:
    """
    Checks if the files exist (otherwise gives a warning and returns None)
    Concatenates ensemble root dir and path to file if relative
    """
    ensembles = list(ensemble_paths.keys())
    return [
        {ens: _process_file(source, ensemble_paths[ens]) for ens in ensembles}
        for source in [cont_bound, haz_bound, well_file]
    ]


def _process_file(file: Optional[str], ensemble_path: str) -> Optional[str]:
    if file is not None:
        if Path(file).is_absolute():
            if os.path.isfile(Path(file)):
                return file
            warnings.warn(f"Cannot find specified file {file}.")
            return None
        file = os.path.join(Path(ensemble_path).parents[1], file)
        if not os.path.isfile(file):
            warnings.warn(
                f"Cannot find specified file {file}.\n"
                "Note that relative paths are accepted from ensemble root "
                "(directory with the realizations)."
            )
            return None
    return file
