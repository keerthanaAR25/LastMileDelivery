"""NexusFlow — Trajectory-Derived Congestion Display.
Reads existing congestion_by_cell_hour.parquet. The raw trajectory coordinates are privacy-transformed and are NOT map-matched to roads.
"""
from pathlib import Path
import pandas as pd
import plotly.express as px
REPO_ROOT = Path(__file__).resolve().parents[2]

def get_congestion_data(max_rows: int = 10000, hour: int | None = None) -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / "congestion_by_cell_hour.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    required = ["cell_id","hour","avg_speed","reference_speed","n_observations","centroid_x","centroid_y","congestion_index"]
    df = df[[c for c in required if c in df.columns]].copy()
    if hour is not None and "hour" in df.columns:
        df = df[df["hour"] == hour].copy()
    if len(df) > max_rows:
        df = df.nlargest(max_rows, "n_observations") if "n_observations" in df.columns else df.head(max_rows)
    return df.reset_index(drop=True)

def make_congestion_figure(df: pd.DataFrame, title: str = "Trajectory-Derived Spatial Congestion"):
    if df.empty:
        return None
    fig = px.scatter(df, x="centroid_x", y="centroid_y", size="n_observations", color="congestion_index", hover_data=["cell_id","hour","avg_speed","reference_speed","n_observations"], title=title, labels={"centroid_x":"Trajectory X (transformed space)","centroid_y":"Trajectory Y (transformed space)","congestion_index":"Congestion Index"})
    fig.update_layout(height=650, margin={"l":20,"r":20,"t":55,"b":20})
    return fig

def congestion_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"available":False,"records":0,"mean_congestion":None,"high_congestion_share":None}
    c = pd.to_numeric(df["congestion_index"], errors="coerce").dropna()
    return {"available":True,"records":int(len(df)),"mean_congestion":round(float(c.mean()),4) if len(c) else None,"high_congestion_share":round(float((c >= 0.7).mean()),4) if len(c) else None}
