from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from shapely import wkt


REPO_ROOT = Path(__file__).resolve().parents[2]

CITY_NAMES = {
    "sh": "Shanghai",
    "cq": "Chongqing",
    "hz": "Hangzhou",
    "jl": "Jilin",
    "yt": "Yantai",
}


def get_road_data(
    city_code: str,
    max_segments: int = 3000,
) -> pd.DataFrame:
    """
    Load the real road-network geometry for the selected city.

    The road geometry is already stored as WGS84
    longitude/latitude WKT.

    max_segments limits only the number of road segments
    displayed in Streamlit for performance.
    """

    if city_code not in CITY_NAMES:
        return pd.DataFrame()

    road_path = (
        REPO_ROOT
        / "data"
        / "raw"
        / "road_network"
        / f"roads_{city_code}.parquet"
    )

    if not road_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(road_path)

    if df.empty:
        return pd.DataFrame()

    required_column = "geometry_wkt_4326"

    if required_column not in df.columns:
        return pd.DataFrame()

    df = df.dropna(
        subset=[required_column]
    ).copy()

    # Limit only the visualization size.
    # The original road-network dataset is unchanged.
    if len(df) > max_segments:
        df = df.sample(
            n=max_segments,
            random_state=42,
        )

    return df


def make_road_figure(
    df: pd.DataFrame,
    title: str = "Real Road Network",
) -> go.Figure:
    """
    Create a geographic visualization of the real road network.

    Coordinates:
        X = longitude
        Y = latitude

    The road geometry comes from the provided
    OpenStreetMap-based road-network dataset.
    """

    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            title=title,
            height=650,
        )
        return fig

    all_lon = []
    all_lat = []

    # ---------------------------------------------------------------
    # Convert WKT geometries to Plotly coordinate arrays
    # ---------------------------------------------------------------

    for geometry_wkt in df["geometry_wkt_4326"]:

        try:
            geometry = wkt.loads(geometry_wkt)
        except Exception:
            continue

        if geometry.geom_type == "LineString":

            for lon, lat, *rest in geometry.coords:
                all_lon.append(lon)
                all_lat.append(lat)

            # Break between separate road segments
            all_lon.append(None)
            all_lat.append(None)

        elif geometry.geom_type == "MultiLineString":

            for line in geometry.geoms:

                for lon, lat, *rest in line.coords:
                    all_lon.append(lon)
                    all_lat.append(lat)

                # Break between separate lines
                all_lon.append(None)
                all_lat.append(None)

    if not all_lon:
        fig.update_layout(
            title=title,
            height=650,
        )
        return fig

    # ---------------------------------------------------------------
    # Calculate geographic center
    # ---------------------------------------------------------------

    valid_lon = [
        value
        for value in all_lon
        if value is not None
    ]

    valid_lat = [
        value
        for value in all_lat
        if value is not None
    ]

    center_lon = sum(valid_lon) / len(valid_lon)
    center_lat = sum(valid_lat) / len(valid_lat)

    # ---------------------------------------------------------------
    # Real road-network lines
    # ---------------------------------------------------------------

    fig.add_trace(
        go.Scattergeo(
            lon=all_lon,
            lat=all_lat,
            mode="lines",
            line=dict(
                width=1,
            ),
            name="Road segments",
            hoverinfo="skip",
        )
    )

    # ---------------------------------------------------------------
    # Geographic map configuration
    # ---------------------------------------------------------------

    fig.update_layout(
        title=title,
        height=650,

        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0,
        ),

        geo=dict(
            projection_type="mercator",

            center=dict(
                lon=center_lon,
                lat=center_lat,
            ),

            fitbounds="locations",

            showland=True,
            showcountries=True,
            showcoastlines=True,
            showlakes=True,
            showrivers=True,

            landcolor="rgb(240, 240, 240)",
            oceancolor="rgb(220, 235, 240)",
            showocean=True,
        ),

        showlegend=True,
    )

    return fig