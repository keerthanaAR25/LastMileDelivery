import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services import data_access as svc
from app.services.city_context import get_selected_city


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NexusFlow - Graph Analytics",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.header("Graph Analytics — Logistics Relationship Network")

city_code = get_selected_city()
city_name = svc.CITY_CODE_TO_NAME.get(city_code, city_code)

st.caption(
    f"Real logistics graph for {city_name}: nodes and relationships "
    "loaded directly from the PostgreSQL graph tables."
)


# ============================================================
# LOAD REAL GRAPH DATA
# ============================================================

try:

    nodes = svc._query_df(
        """
        SELECT
            node_id,
            node_type,
            attributes_json
        FROM analytics.logistics_nodes
        WHERE node_id LIKE %s
        """,
        (f"{city_code}_%",)
    )

    edges = svc._query_df(
        """
        SELECT
            edge_id,
            source_node_id,
            target_node_id,
            edge_type,
            frequency,
            duration,
            risk,
            congestion,
            workload,
            pattern_score
        FROM analytics.logistics_edges
        WHERE source_node_id LIKE %s
           OR target_node_id LIKE %s
        """,
        (f"{city_code}_%", f"{city_code}_%")
    )

except Exception as e:

    st.error(f"Unable to load graph data: {e}")
    st.stop()


# ============================================================
# VALIDATE NODES
# ============================================================

if nodes.empty:

    st.warning(
        f"No logistics graph nodes are available for {city_name}."
    )

    st.stop()


nodes = nodes.copy()

nodes["node_id"] = nodes["node_id"].astype(str)

nodes["node_type"] = (
    nodes["node_type"]
    .fillna("UNKNOWN")
    .astype(str)
)


# ============================================================
# CLEAN REAL EDGES
# ============================================================

if not edges.empty:

    edges = edges.copy()

    edges = edges[
        edges["source_node_id"].notna()
        & edges["target_node_id"].notna()
    ].copy()

    edges["source_node_id"] = (
        edges["source_node_id"]
        .astype(str)
    )

    edges["target_node_id"] = (
        edges["target_node_id"]
        .astype(str)
    )

    # Only keep relationships whose endpoints
    # actually exist in the real node table.
    valid_nodes = set(nodes["node_id"])

    edges = edges[
        edges["source_node_id"].isin(valid_nodes)
        & edges["target_node_id"].isin(valid_nodes)
    ].copy()


# ============================================================
# BUILD REAL NETWORKX GRAPH
# ============================================================

G = nx.Graph()


# ------------------------------------------------------------
# Add real nodes
# ------------------------------------------------------------

for _, row in nodes.iterrows():

    G.add_node(
        row["node_id"],
        node_type=row["node_type"]
    )


# ------------------------------------------------------------
# Add real relationships
# ------------------------------------------------------------

for _, row in edges.iterrows():

    G.add_edge(
        row["source_node_id"],
        row["target_node_id"],
        edge_type=(
            str(row["edge_type"])
            if pd.notna(row["edge_type"])
            else "RELATED"
        )
    )


# ============================================================
# SUMMARY METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Real Nodes",
    f"{G.number_of_nodes():,}"
)


c2.metric(
    "Real Relationships",
    f"{G.number_of_edges():,}"
)


c3.metric(
    "Node Types",
    f"{nodes['node_type'].nunique():,}"
)


c4.metric(
    "Relationship Types",
    (
        f"{edges['edge_type'].nunique():,}"
        if not edges.empty
        else "0"
    )
)


# ============================================================
# NODE TYPE DISTRIBUTION
# ============================================================

st.subheader("Logistics Node Structure")

node_counts = (
    nodes["node_type"]
    .value_counts()
    .rename_axis("Node Type")
    .reset_index(name="Count")
)

st.bar_chart(
    node_counts.set_index("Node Type")
)


# ============================================================
# RELATIONSHIP TYPE DISTRIBUTION
# ============================================================

if not edges.empty:

    st.subheader("Real Relationship Types")

    edge_counts = (
        edges["edge_type"]
        .fillna("UNKNOWN")
        .value_counts()
        .rename_axis("Relationship Type")
        .reset_index(name="Count")
    )

    st.bar_chart(
        edge_counts.set_index("Relationship Type")
    )


# ============================================================
# REAL GRAPH VISUALIZATION
# ============================================================

st.subheader("Interactive Logistics Graph")


if G.number_of_edges() == 0:

    st.warning(
        "Graph nodes exist, but no valid relationships were found "
        f"for {city_name}."
    )

else:

    # ========================================================
    # SELECT A REAL CONNECTED SUBGRAPH
    # ========================================================
    #
    # We do NOT simply select the top 250 nodes.
    #
    # Instead:
    #
    # 1. Find highly connected real nodes.
    # 2. Start from those real hubs.
    # 3. Add their real neighbors.
    # 4. Keep only real relationships.
    #
    # This prevents the previous "ring of dots" effect.
    # ========================================================

    MAX_GRAPH_NODES = 250

    degree_dict = dict(G.degree())

    degree_series = (
        pd.Series(degree_dict, name="degree")
        .sort_values(ascending=False)
    )


    # --------------------------------------------------------
    # Start with highly connected real hubs
    # --------------------------------------------------------

    hub_candidates = list(
        degree_series.head(25).index
    )


    selected_nodes = set()


    # --------------------------------------------------------
    # Add hubs and their real neighbors
    # --------------------------------------------------------

    for hub in hub_candidates:

        selected_nodes.add(hub)

        neighbors = list(
            G.neighbors(hub)
        )

        # Add the strongest neighbors first
        neighbors = sorted(
            neighbors,
            key=lambda n: degree_dict.get(n, 0),
            reverse=True
        )

        for neighbor in neighbors:

            selected_nodes.add(neighbor)

            if len(selected_nodes) >= MAX_GRAPH_NODES:
                break

        if len(selected_nodes) >= MAX_GRAPH_NODES:
            break


    # --------------------------------------------------------
    # Create REAL subgraph
    # --------------------------------------------------------

    H = G.subgraph(
        selected_nodes
    ).copy()


    # --------------------------------------------------------
    # Remove isolated nodes
    # --------------------------------------------------------

    isolated_nodes = list(
        nx.isolates(H)
    )

    if isolated_nodes:

        H.remove_nodes_from(
            isolated_nodes
        )


    # --------------------------------------------------------
    # Keep the largest connected component
    # --------------------------------------------------------

    if H.number_of_nodes() > 0:

        components = list(
            nx.connected_components(H)
        )

        largest_component = max(
            components,
            key=len
        )

        H = H.subgraph(
            largest_component
        ).copy()


    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    st.caption(
        f"Displaying {H.number_of_nodes():,} connected real nodes "
        f"and {H.number_of_edges():,} real relationships "
        f"from {G.number_of_nodes():,} total nodes and "
        f"{G.number_of_edges():,} total relationships."
    )


    if H.number_of_nodes() == 0:

        st.warning(
            "No connected graph neighborhood could be displayed."
        )

    else:

        # ====================================================
        # REAL NETWORK LAYOUT
        # ====================================================
        #
        # NetworkX determines only visual positions.
        #
        # The nodes and edges themselves are real database
        # records.
        # ====================================================

        pos = nx.spring_layout(
            H,
            seed=42,
            k=0.8,
            iterations=150
        )


        # ====================================================
        # REAL EDGE TRACE
        # ====================================================

        edge_x = []
        edge_y = []

        for source, target in H.edges():

            x0, y0 = pos[source]
            x1, y1 = pos[target]

            edge_x.extend(
                [
                    x0,
                    x1,
                    None
                ]
            )

            edge_y.extend(
                [
                    y0,
                    y1,
                    None
                ]
            )


        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(
                width=1
            ),
            hoverinfo="none",
            showlegend=False
        )


        # ====================================================
        # REAL NODE TRACE
        # ====================================================

        node_x = []
        node_y = []
        node_text = []
        node_sizes = []


        for node in H.nodes():

            x, y = pos[node]

            node_x.append(x)
            node_y.append(y)

            node_type = H.nodes[node].get(
                "node_type",
                "UNKNOWN"
            )

            degree = H.degree(node)

            # Node size represents real graph degree.
            node_sizes.append(
                max(
                    10,
                    min(
                        35,
                        10 + degree * 1.8
                    )
                )
            )

            node_text.append(
                f"<b>{node}</b><br>"
                f"Type: {node_type}<br>"
                f"Connections: {degree}"
            )


        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_text,
            hovertemplate=(
                "%{text}"
                "<extra></extra>"
            ),
            marker=dict(
                size=node_sizes,
                line=dict(
                    width=1
                )
            ),
            showlegend=False
        )


        # ====================================================
        # BUILD FIGURE
        # ====================================================

        fig = go.Figure(
            data=[
                edge_trace,
                node_trace
            ]
        )


        fig.update_layout(
            height=750,

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),

            paper_bgcolor="white",
            plot_bgcolor="white",

            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),

            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),

            hovermode="closest"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# DATABASE RELATIONSHIPS
# ============================================================

st.subheader("Database Relationships")

if not edges.empty:

    display_edges = edges.copy()

    # AOI_IN_REGION relationships only represent
    # spatial containment, so their operational metrics
    # are not applicable.
    containment_mask = (
        display_edges["edge_type"] == "AOI_IN_REGION"
    )

    for column in [
        "frequency",
        "duration",
        "risk",
        "congestion",
        "workload",
        "pattern_score",
    ]:
        if column in display_edges.columns:
            display_edges.loc[
                containment_mask,
                column
            ] = "N/A"

    display_columns = [
        "source_node_id",
        "target_node_id",
        "edge_type",
        "frequency",
        "duration",
        "risk",
        "congestion",
        "workload",
        "pattern_score",
    ]

    display_columns = [
        c for c in display_columns
        if c in display_edges.columns
    ]

    st.dataframe(
        display_edges[
            display_columns
        ].head(1000),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No relationships available.")

# ============================================================
# GRAPH INTERPRETATION
# ============================================================

st.subheader("Graph Interpretation")

st.write(
    f"""
This graph represents the actual logistics relationships stored for
**{city_name}**. Nodes represent logistics entities such as couriers,
regions, AOIs and other entities present in the graph database.

Edges represent relationships generated by the NexusFlow graph model.

The displayed network is a connected subset of the real database graph,
selected around highly connected logistics entities for readability.

No synthetic nodes or relationships are created.

Node size represents the number of observed graph connections.

The NetworkX layout determines only the visual position of nodes;
it does not represent geographic coordinates.
"""
)