import plotly.graph_objects as go

import featurewiz_explained as fwx
import featurewiz_explained_plots as plots


def test_correlation_heatmap_is_absolute_and_square():
    data = fwx.make_canonical_dataset()
    fig = plots.correlation_heatmap(data.frame, data.features)
    assert isinstance(fig, go.Figure)
    heat = fig.data[0]
    assert heat.type == "heatmap" and len(heat.z) == 12 and len(heat.z[0]) == 12
    assert min(min(row) for row in heat.z) >= 0


def test_sulov_graph_has_one_edge_per_pair_and_labels_every_node():
    data = fwx.make_canonical_dataset()
    result = fwx.sulov(data.frame, data.features, data.target)
    fig = plots.sulov_graph(result, data.frame)
    edges = [t for t in fig.data if t.mode == "lines"]
    nodes = [t for t in fig.data if t.mode == "markers+text"]
    assert len(edges) == len(result.correlated_pairs)
    labels = [label for t in nodes for label in t.text]
    assert sorted(labels) == sorted(
        [f"{f} (kept)" for f in result.mutual_info if f in result.kept]
        + [f"{f} (dropped)" for f in result.mutual_info if f in result.dropped]
    )


def test_rounds_heatmap_marks_nominations():
    data = fwx.make_canonical_dataset()
    kept = fwx.sulov(data.frame, data.features, data.target).kept
    result = fwx.recursive_xgboost(data.frame, kept, data.target)
    heat = plots.rounds_heatmap(result).data[0]
    assert list(heat.y) == kept
    assert len(heat.x) == len(result.rounds)
    assert sum(cell == "★" for row in heat.text for cell in row) == sum(len(r.nominated) for r in result.rounds)


def test_selection_table_lists_every_feature_with_its_role():
    data = fwx.make_canonical_dataset()
    table = plots.selection_table(["x1", "x2"], data).data[0]
    feature_col, role_col, kept_col = table.cells.values[0], table.cells.values[1], table.cells.values[3]
    assert list(feature_col) == data.features
    assert role_col[0] == "informative"
    assert kept_col[0] == "✓ kept" and kept_col[2] == "—"
    fills = table.cells.fill.color
    assert len(fills) == 4 and set(fills[3]) == {"white"}  # role colour only in the role column
