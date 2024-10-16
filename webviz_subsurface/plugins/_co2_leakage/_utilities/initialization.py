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
from webviz_subsurface.plugins._co2_leakage._utilities.co2volume import (
    read_menu_options,
)
from webviz_subsurface.plugins._co2_leakage._utilities.ensemble_polygon_provider import (
    EnsemblePolygonProvider
)
from webviz_subsurface.plugins._co2_leakage._utilities.ensemble_well_picks import (
    EnsembleWellPicks
)
from webviz_subsurface.plugins._co2_leakage._utilities.generic import (
    GraphSource,
    MapAttribute,
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
    ensemble_paths: Dict[str, str],
    well_pick_path: Optional[str],
    map_surface_names_to_well_pick_names: Optional[Dict[str, str]],
) -> Dict[str, EnsembleWellPicks]:
    if well_pick_path is None:
        return {}

    return {
        ens: EnsembleWellPicks(
            ens_p, well_pick_path, map_surface_names_to_well_pick_names
        )
        for ens, ens_p in ensemble_paths.items()
    }


def init_hazardous_boundary_providers(
    ensemble_paths: Dict[str, str],
    poly_path: Optional[str],
) -> Dict[str, Optional[EnsemblePolygonProvider]]:
    if poly_path is None:
        return {}

    return {
        ens: _init_hazardous_boundary_provider(ens_path, poly_path)
        for ens, ens_path in ensemble_paths.items()
    }


def _init_hazardous_boundary_provider(
    ensemble_path: str, poly_path: str
) -> Optional[EnsemblePolygonProvider]:
    try:
        return EnsemblePolygonProvider(
            ensemble_path,
            poly_path,
            "Hazardous Polygon",
            "hazardous-boundary-layer",
            [200, 0, 0, 120]
        )
    except OSError as e:
        LOGGER.warning(
            f"Failed to create hazardous boundary provider for ensemble path:"
            f" '{ensemble_path}' and poly path '{poly_path}': {e}"
        )
        return None


def init_containment_boundary_providers(
    ensemble_paths: Dict[str, str],
    poly_path: Optional[str],
) -> Dict[str, Optional[EnsemblePolygonProvider]]:
    if poly_path is None:
        return {}

    return {
        ens: _init_containment_boundary_provider(ens_path, poly_path)
        for ens, ens_path in ensemble_paths.items()
    }


def _init_containment_boundary_provider(
    ensemble_path: str,
    poly_path: str,
) -> Optional[EnsemblePolygonProvider]:
    try:
        return EnsemblePolygonProvider(
            ensemble_path,
            poly_path,
            "Containment Polygon",
            "license-boundary-layer",
            [0, 172, 0, 120],
        )
    except OSError as e:
        LOGGER.warning(
            "Failed to create containment boundary provider for ensemble path:"
            f" '{ensemble_path}' and poly path '{poly_path}': {e}"
        )
        return None


def init_table_provider(
    ensemble_roots: Dict[str, str],
    table_rel_path: str,
) -> Dict[str, EnsembleTableProvider]:
    providers = {}
    factory = EnsembleTableProviderFactory.instance()
    for ens, ens_path in ensemble_roots.items():
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
            providers[ens] = factory.create_from_per_realization_csv_file(
                ens_path, table_rel_path
            )
        except (KeyError, ValueError) as exc:
            LOGGER.warning(
                f'Did not load "{table_rel_path}" for ensemble "{ens}" with error {exc}'
            )
    return providers


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
    mass_table: Dict[str, EnsembleTableProvider],
    actual_volume_table: Dict[str, EnsembleTableProvider],
    mass_relpath: str,
    volume_relpath: str,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    options: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for ens in ensemble_roots.keys():
        options[ens] = {}
        for source, table, relpath in zip(
            [GraphSource.CONTAINMENT_MASS, GraphSource.CONTAINMENT_ACTUAL_VOLUME],
            [mass_table, actual_volume_table],
            [mass_relpath, volume_relpath],
        ):
            real = table[ens].realizations()[0]
            options[ens][source] = read_menu_options(table[ens], real, relpath)
        options[ens][GraphSource.UNSMRY] = {
            "zones": [],
            "regions": [],
            "phases": ["total", "gas", "aqueous"],
        }
    return options


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
