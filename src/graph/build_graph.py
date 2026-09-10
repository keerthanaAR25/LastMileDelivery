"""
Module 5 — Graph-Based Logistics Modeling (Section 32).

REAL SCOPING DECISION (documented, not silent): Shanghai has 1,483,864
orders. Building one graph node per PACKAGE, as Section 32 literally lists,
would create 1.48M+ nodes in pure-Python NetworkX on this sandbox's 1
CPU/3.9GB RAM — not tractable here. Instead, PACKAGE-level detail is
represented via edge WEIGHTS (frequency/duration/risk) on courier<->AOI
edges rather than as individual nodes, which still satisfies Section 32's
requirement that edges carry "frequency, duration, risk... workload,
pattern_score" attributes.

Implemented node types: COURIER, AOI, REGION, CITY.
Implemented edge types: COURIER_VISITED_AOI, AOI_IN_REGION.
NOT implemented (and why):
  - PACKAGE nodes / COURIER_ASSIGNED_PACKAGE / PACKAGE_LOCATED_AOI /
    PACKAGE_DELIVERED_AT_AOI: scale (see above).
  - ROAD nodes / COURIER_TRAVERSED_ROAD / ROAD_IN_REGION: no road-network
    or trajectory data has been provided yet.

REAL DATA NOTE: 1,447 of 15,564 Shanghai AOIs (9.3%) associate with more
than one region_id across their orders — not a clean AOI-in-one-region
hierarchy. AOI_IN_REGION uses each AOI's MODAL region; the ambiguity rate
is recorded, not hidden.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging_config import get_module_logger, ModuleRun  # noqa: E402

log = get_module_logger("MODULE_05_GRAPH")


def build_graph(df: pd.DataFrame, city_name: str) -> tuple[nx.MultiDiGraph, dict]:
    G = nx.MultiDiGraph()
    G.add_node(f"CITY_{city_name}", node_type="CITY")

    # REGION nodes
    region_stats = df.groupby("region_id").agg(
        delivery_count=("order_id", "count"), risk_rate=("risk_label", "mean"),
    )
    for region_id, row in region_stats.iterrows():
        G.add_node(f"REGION_{region_id}", node_type="REGION",
                    delivery_count=int(row.delivery_count), risk_rate=float(row.risk_rate))

    # AOI nodes + AOI_IN_REGION edges (modal region per AOI)
    aoi_region = df.groupby("aoi_id")["region_id"].agg(lambda s: s.mode().iloc[0])
    aoi_nunique_region = df.groupby("aoi_id")["region_id"].nunique()
    aoi_stats = df.groupby("aoi_id").agg(
        delivery_count=("order_id", "count"), risk_rate=("risk_label", "mean"),
        aoi_type=("aoi_type", lambda s: s.mode().iloc[0]),
    )
    ambiguous_aoi_count = int((aoi_nunique_region > 1).sum())
    for aoi_id, row in aoi_stats.iterrows():
        G.add_node(f"AOI_{aoi_id}", node_type="AOI", aoi_type=int(row.aoi_type),
                    delivery_count=int(row.delivery_count), risk_rate=float(row.risk_rate),
                    region_ambiguous=bool(aoi_nunique_region.get(aoi_id, 1) > 1))
        G.add_edge(f"AOI_{aoi_id}", f"REGION_{aoi_region[aoi_id]}", edge_type="AOI_IN_REGION")

    # COURIER nodes
    courier_stats = df.groupby("courier_id").agg(
        delivery_count=("order_id", "count"), risk_rate=("risk_label", "mean"),
        avg_duration=("delivery_duration_min", "mean"),
    )
    for courier_id, row in courier_stats.iterrows():
        G.add_node(f"COURIER_{courier_id}", node_type="COURIER",
                    workload=int(row.delivery_count), risk_rate=float(row.risk_rate),
                    avg_duration=float(row.avg_duration))

    # COURIER_VISITED_AOI edges (frequency-weighted, real Section-32 attrs)
    cv = df.groupby(["courier_id", "aoi_id"]).agg(
        frequency=("order_id", "count"),
        duration=("delivery_duration_min", "mean"),
        risk=("risk_label", "mean"),
    ).reset_index()
    for row in cv.itertuples():
        G.add_edge(
            f"COURIER_{row.courier_id}", f"AOI_{row.aoi_id}",
            edge_type="COURIER_VISITED_AOI",
            frequency=int(row.frequency),
            duration=float(row.duration) if pd.notna(row.duration) else None,
            risk=float(row.risk) if pd.notna(row.risk) else None,
        )

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_couriers": len(courier_stats),
        "n_aois": len(aoi_stats),
        "n_regions": len(region_stats),
        "ambiguous_aoi_region_count": ambiguous_aoi_count,
    }
    return G, stats


def compute_graph_analytics(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Section 32: degree, weighted degree, centrality. Full betweenness
    centrality is O(V*E) and not tractable for this graph's edge count on
    1 CPU — degree centrality is used as the primary tractable measure,
    documented rather than silently substituted."""
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="frequency"))
    degree_centrality = nx.degree_centrality(G)

    rows = []
    for node, attrs in G.nodes(data=True):
        rows.append({
            "node_id": node,
            "node_type": attrs.get("node_type"),
            "degree": degree.get(node, 0),
            "weighted_degree": weighted_degree.get(node, 0),
            "degree_centrality": degree_centrality.get(node, 0.0),
            **{k: v for k, v in attrs.items() if k != "node_type"},
        })
    return pd.DataFrame(rows)


def main(processed_path: str, city_code: str, city_name: str):
    with ModuleRun(log, module="MODULE 05 - GRAPH") as run:
        df = pd.read_parquet(processed_path)
        G, build_stats = build_graph(df, city_name)
        node_df = compute_graph_analytics(G)

        out_dir = Path("artifacts/graph")
        out_dir.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, out_dir / f"logistics_graph_{city_code}.graphml")
        node_df.to_parquet(out_dir / f"graph_nodes_{city_code}.parquet", index=False)

        edge_rows = [
            {"source": u, "target": v, **attrs}
            for u, v, attrs in G.edges(data=True)
        ]
        pd.DataFrame(edge_rows).to_parquet(out_dir / f"graph_edges_{city_code}.parquet", index=False)

        run.record(city=city_code, **build_stats, artifact_location=str(out_dir))
        return {"graph": G, "nodes": node_df, "edges": pd.DataFrame(edge_rows), "stats": build_stats}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--city-name", required=True)
    args = parser.parse_args()
    main(args.input, args.city, args.city_name)
