from pathlib import Path

import networkx as nx
import pandas as pd
from shapely import wkt


REPO_ROOT = Path(__file__).resolve().parents[2]


def _node(point):
    """Create a stable geographic node from a coordinate."""
    return (
        round(float(point[0]), 6),
        round(float(point[1]), 6),
    )


def load_road_graph(city_code: str) -> nx.Graph:
    """
    Build a road graph from the provided city road-network data.

    The complete city road dataset is used.
    The largest connected component is retained so that
    route calculation is performed on a connected network.
    """

    road_path = (
        REPO_ROOT
        / "data"
        / "raw"
        / "road_network"
        / f"roads_{city_code}.parquet"
    )

    if not road_path.exists():
        return nx.Graph()

    df = pd.read_parquet(
        road_path,
        columns=[
            "road_id",
            "length_km",
            "geometry_wkt_4326",
        ],
    )

    df = df.dropna(
        subset=["geometry_wkt_4326"]
    )

    graph = nx.Graph()

    for row in df.itertuples(index=False):

        try:
            geometry = wkt.loads(
                row.geometry_wkt_4326
            )
        except Exception:
            continue

        if geometry.geom_type == "LineString":

            lines = [geometry]

        elif geometry.geom_type == "MultiLineString":

            lines = list(geometry.geoms)

        else:

            continue

        for line in lines:

            coordinates = list(
                line.coords
            )

            if len(coordinates) < 2:
                continue

            start = _node(
                coordinates[0]
            )

            end = _node(
                coordinates[-1]
            )

            if start == end:
                continue

            try:
                length = float(
                    row.length_km
                )
            except Exception:
                continue

            if length <= 0:
                continue

            graph.add_edge(
                start,
                end,
                weight=length,
                road_id=row.road_id,
                geometry=coordinates,
            )

    if graph.number_of_nodes() == 0:
        return graph

    # Keep only the largest connected road network.
    largest_component = max(
        nx.connected_components(graph),
        key=len,
    )

    return graph.subgraph(
        largest_component
    ).copy()


def nearest_node(
    graph: nx.Graph,
    longitude: float,
    latitude: float,
):
    """Find the nearest road-network node."""

    if graph.number_of_nodes() == 0:
        return None

    return min(
        graph.nodes,
        key=lambda node: (
            (node[0] - longitude) ** 2
            + (node[1] - latitude) ** 2
        ),
    )


def find_route(
    graph: nx.Graph,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
):
    """
    Calculate the shortest-distance physical road route.
    """

    start_node = nearest_node(
        graph,
        start_lon,
        start_lat,
    )

    end_node = nearest_node(
        graph,
        end_lon,
        end_lat,
    )

    if (
        start_node is None
        or end_node is None
    ):
        return (
            None,
            0.0,
            start_node,
            end_node,
        )

    try:

        route = nx.shortest_path(
            graph,
            start_node,
            end_node,
            weight="weight",
        )

        distance = nx.shortest_path_length(
            graph,
            start_node,
            end_node,
            weight="weight",
        )

    except nx.NetworkXNoPath:

        return (
            None,
            0.0,
            start_node,
            end_node,
        )

    return (
        route,
        float(distance),
        start_node,
        end_node,
    )


def route_coordinates(
    graph: nx.Graph,
    route,
):
    """
    Reconstruct the actual stored road geometries
    along the calculated route.
    """

    if not route:
        return []

    if len(route) == 1:
        return [
            route[0]
        ]

    coordinates = []

    for start, end in zip(
        route[:-1],
        route[1:],
    ):

        edge_data = graph.get_edge_data(
            start,
            end,
        )

        if not edge_data:
            continue

        edge_coordinates = list(
            edge_data["geometry"]
        )

        # Make sure geometry follows the
        # direction of the calculated route.
        if (
            _node(edge_coordinates[0])
            != start
        ):
            edge_coordinates.reverse()

        if not coordinates:

            coordinates.extend(
                edge_coordinates
            )

        else:

            coordinates.extend(
                edge_coordinates[1:]
            )

    return coordinates


def sample_route_endpoints(
    graph: nx.Graph,
):
    """
    Select two deterministic nodes from the
    connected road network for a ready-to-run demo.
    """

    nodes = list(
        graph.nodes
    )

    if len(nodes) < 2:
        return None, None

    start = min(
        nodes,
        key=lambda node:
        node[0] + node[1],
    )

    end = max(
        nodes,
        key=lambda node:
        node[0] + node[1],
    )

    return start, end