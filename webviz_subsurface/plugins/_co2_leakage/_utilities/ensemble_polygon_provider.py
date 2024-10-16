from pathlib import Path
from typing import Dict, Any, List, Optional

from webviz_subsurface._utils.webvizstore_functions import read_csv
from webviz_subsurface.plugins._co2_leakage._utilities._misc import realization_paths


class EnsemblePolygonProvider:
    def __init__(
        self,
        ens_path: str,
        poly_path: str,
        layer_name: str,
        layer_id: str,
        layer_color: List[int],
    ):
        pp = Path(poly_path)

        self._layer_name = layer_name
        self._layer_id = layer_id
        self._layer_color = layer_color

        self._absolute_polygon = None
        self._per_real_polygons = None

        if pp.is_absolute():
            self._absolute_polygon = _parse_polygon_file(pp)
        else:
            self._per_real_polygons = {
                i: _parse_polygon_file(Path(r) / pp)
                for i, r in realization_paths(ens_path).items()
            }

    def geojson_layer(self, realization: int) -> Optional[Dict[str, Any]]:
        if self._absolute_polygon is not None:
            data = self._absolute_polygon
        else:
            data = self._per_real_polygons.get(realization, None)

        if data is None:
            return None

        return {
            "@@type": "GeoJsonLayer",
            "name": self._layer_name,
            "id": self._layer_id,
            "data": data,
            "stroked": False,
            "getFillColor": self._layer_color,
            "visible": True,
        }


def _parse_polygon_file(filename: Path) -> Optional[Dict[str, Any]]:
    try:
        df = read_csv(filename)
    except OSError:
        return None

    if "x" in df.columns:
        xyz = df[["x", "y"]].values
    elif "X_UTME" in df.columns:
        if "POLY_ID" in df.columns:
            xyz = [gf[["X_UTME", "Y_UTMN"]].values for _, gf in df.groupby("POLY_ID")]
        else:
            xyz = df[["X_UTME", "Y_UTMN"]].values
    else:
        # Attempt to use the first two columns as the x and y coordinates
        xyz = df.values[:, :2]
    if isinstance(xyz, list):
        poly_type = "MultiPolygon"
        coords = [[arr.tolist()] for arr in xyz]
    else:
        poly_type = "Polygon"
        coords = [xyz.tolist()]
    as_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": poly_type,
                    "coordinates": coords,
                },
            }
        ],
    }
    return as_geojson
