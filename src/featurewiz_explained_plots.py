"""Plotly figures for the featurewiz, explained Lessons.

Static Plotly only (no widgets), so charts work in Kaggle's read-only view.
Colours follow the dataviz reference palette; identity always has a text label too.
"""
from __future__ import annotations

import json
import math
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.io._base_renderers import MimetypeRenderer
from plotly.offline import get_plotlyjs_version

from featurewiz_explained import RecursiveXgbResult, SulovResult, SyntheticDataset

ROLE_COLOURS = {"informative": "#2a78d6", "redundant": "#eb6834", "noise": "#1baf7a"}
KEPT, DROPPED = "#2a78d6", "#a3a29c"
TEMPLATE = "plotly_white"

PLOTLY_JS = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
_DRAW = """<div id="{id}" style="height:{height}px; width:100%;"></div>
<script>(function () {{
  var fig = {fig};
  // Evaluate plotly.js with define/module/exports shadowed, so it always sets window.Plotly
  // and never touches the page's require.js.
  window._fwxPlotly = window._fwxPlotly || (window.Plotly ? Promise.resolve() :
    fetch("{js}").then(function (r) {{ return r.text(); }}).then(function (code) {{
      new Function("define", "module", "exports", code).call(window);
    }}));
  window._fwxPlotly.then(function () {{
    window.Plotly.newPlot("{id}", fig.data, fig.layout, {{responsive: true}});
  }});
}})();</script>"""


class _SelfLoadingRenderer(MimetypeRenderer):
    """HTML that loads plotly.js itself, out of reach of the page's require.js.

    Kaggle's notebook viewer loads require.js, so plotly's own <script src> output registers
    an AMD module instead of window.Plotly and every chart fails with "Plotly is not defined".
    """

    def to_mimebundle(self, fig_dict):
        fig = json.loads(pio.to_json(fig_dict, validate=False))
        height = fig.get("layout", {}).get("height", 450)
        html = _DRAW.format(id=uuid.uuid4(), height=height, fig=json.dumps(fig).replace("</", "<\\/"), js=PLOTLY_JS)
        return {"text/html": html}


pio.renderers["featurewiz_explained"] = _SelfLoadingRenderer()
pio.renderers.default = "featurewiz_explained"


def correlation_heatmap(frame: pd.DataFrame, features: list[str], corr_limit: float = 0.8) -> go.Figure:
    corr = frame[features].corr(method="pearson").abs()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=features, y=features, zmin=0, zmax=1, colorscale="Blues",
        colorbar=dict(title="|corr|"),
        hovertemplate="%{y} vs %{x}<br>|corr| = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, title=f"Absolute Pearson correlation (SULOV pairs need ≥ {corr_limit})",
        yaxis=dict(autorange="reversed"), height=520,
    )
    return fig


def sulov_graph(result: SulovResult, frame: pd.DataFrame) -> go.Figure:
    """Features that are in a correlated pair, on a circle. Edges = pairs. Size = mutual information."""
    nodes = list(result.mutual_info)
    angle = {f: 2 * math.pi * i / max(len(nodes), 1) for i, f in enumerate(nodes)}
    pos = {f: (math.cos(a), math.sin(a)) for f, a in angle.items()}
    corr = frame[nodes].corr(method="pearson").abs() if nodes else None
    fig = go.Figure()
    for a, b in result.correlated_pairs:
        c = corr.loc[a, b]
        fig.add_trace(go.Scatter(
            x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
            line=dict(color="#c3c2b7", width=1 + 4 * c), hoverinfo="text",
            text=f"{a} – {b}: |corr| = {c:.3f}", showlegend=False,
        ))
    biggest = max(result.mutual_info.values(), default=1) or 1
    for label, colour, members in (
        ("kept", KEPT, [f for f in nodes if f in result.kept]),
        ("dropped", DROPPED, [f for f in nodes if f in result.dropped]),
    ):
        fig.add_trace(go.Scatter(
            x=[pos[f][0] for f in members], y=[pos[f][1] for f in members], mode="markers+text",
            name=label, text=[f"{f} ({label})" for f in members], textposition="top center",
            marker=dict(size=[14 + 36 * result.mutual_info[f] / biggest for f in members],
                        color=colour, line=dict(color="white", width=2)),
            customdata=[result.mutual_info[f] for f in members],
            hovertemplate="%{text}<br>mutual information = %{customdata:.4f}<extra></extra>",
        ))
    fig.update_layout(
        template=TEMPLATE, title="SULOV: bigger circle = more mutual information; thicker line = stronger correlation",
        xaxis=dict(visible=False, range=[-1.5, 1.5]), yaxis=dict(visible=False, range=[-1.4, 1.5], scaleanchor="x"),
        height=560,
    )
    return fig


def rounds_heatmap(result: RecursiveXgbResult) -> go.Figure:
    """Rows = features (input order), columns = Rounds. Cell = share of that Round's total gain.
    Blank = not in that Round's window. ★ = nominated."""
    features = result.rounds[0].window if result.rounds else []
    z, text = [], []
    for f in features:
        row_z, row_t = [], []
        for r in result.rounds:
            total = sum(r.gains.values()) or 1.0
            row_z.append(r.gains.get(f, 0.0) / total if f in r.window else np.nan)
            row_t.append("★" if f in r.nominated else "")
        z.append(row_z)
        text.append(row_t)
    labels = [f"Round {i + 1}" for i in range(len(result.rounds))]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=features, text=text, texttemplate="%{text}", colorscale="Blues", zmin=0,
        xgap=2, ygap=2,
        colorbar=dict(title="gain share"),
        hovertemplate="%{y}, %{x}<br>share of gain = %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        template=TEMPLATE, height=max(320, 26 * len(features)),
        title=f"Recursive XGBoost: step {result.step}, top {result.top_num} per Round (★ = nominated, grey = outside window)",
        yaxis=dict(autorange="reversed"), plot_bgcolor="#e4e3df",
    )
    return fig


def selection_table(selected: list[str], dataset: SyntheticDataset) -> go.Figure:
    features = dataset.features
    roles = [dataset.roles[f] for f in features]
    plain, ink = ["white"] * len(features), ["#0b0b0b"] * len(features)
    fig = go.Figure(go.Table(
        header=dict(values=["feature", "planted role", "signal group", "featurewiz"], align="left"),
        cells=dict(
            values=[features, roles, [dataset.groups.get(f, "") for f in features],
                    ["✓ kept" if f in selected else "—" for f in features]],
            # one list per column; plotly repeats the last list for any columns left over
            fill_color=[plain, [ROLE_COLOURS[r] for r in roles], plain, plain],
            font=dict(color=[ink, plain, ink, ink]),
            align="left",
        ),
    ))
    fig.update_layout(template=TEMPLATE, height=50 + 21 * len(features), margin=dict(t=10, b=10))
    return fig
