"""
kg_bilinear_geometric.py

Geometric Plotly visualizations for bilinear knowledge graph embedding models:

- DistMult
- ComplEx

Designed for Jupyter notebook tutorials with census-record examples.

Usage
-----
from kg_bilinear_geometric import (
    make_distmult_geometric_plot,
    make_complex_geometric_plot,
    plot_bilinear_geometric_models
)

fig = make_distmult_geometric_plot()
fig.show()

fig = make_complex_geometric_plot()
fig.show()

fig = plot_bilinear_geometric_models()
fig.show()
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Basic scoring helpers
# ---------------------------------------------------------------------

def dot_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Standard dot product score.
    """
    return float(np.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    """
    return float(
        np.dot(a, b)
        / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12))
    )


def score_distmult(h: np.ndarray, r: np.ndarray, t: np.ndarray) -> float:
    """
    DistMult score:

        score(h, r, t) = sum_i h_i r_i t_i

    Equivalent geometric form:

        score(h, r, t) = (h ⊙ r)^T t

    The relation rescales the head vector dimension by dimension,
    then the model computes a dot product with the tail vector.
    """
    h = np.asarray(h, dtype=float)
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)

    return float(np.sum(h * r * t))


def score_complex_1d(h: complex, r: complex, t: complex) -> float:
    """
    One-dimensional ComplEx score:

        score(h, r, t) = Re(h * r * conjugate(t))

    In the complex plane, multiplying by r rotates and scales h.
    """
    return float(np.real(h * r * np.conj(t)))


def score_complex(
    h_re: np.ndarray,
    h_im: np.ndarray,
    r_re: np.ndarray,
    r_im: np.ndarray,
    t_re: np.ndarray,
    t_im: np.ndarray,
) -> float:
    """
    Multi-dimensional ComplEx score:

        score(h, r, t) = Re(<h, r, conjugate(t)>)

    Expanded form:

        sum_i [
            h_re_i r_re_i t_re_i
          + h_re_i r_im_i t_im_i
          + h_im_i r_re_i t_im_i
          - h_im_i r_im_i t_re_i
        ]
    """
    h_re = np.asarray(h_re, dtype=float)
    h_im = np.asarray(h_im, dtype=float)
    r_re = np.asarray(r_re, dtype=float)
    r_im = np.asarray(r_im, dtype=float)
    t_re = np.asarray(t_re, dtype=float)
    t_im = np.asarray(t_im, dtype=float)

    terms = (
        h_re * r_re * t_re
        + h_re * r_im * t_im
        + h_im * r_re * t_im
        - h_im * r_im * t_re
    )

    return float(np.sum(terms))


# ---------------------------------------------------------------------
# Text panels
# ---------------------------------------------------------------------

DISTMULT_PROPERTIES = (
    "<b>Relation properties</b><br><br>"
    "Symmetry: <b style='color:#15803d'>Strong</b><br>"
    "score(h,r,t) = score(t,r,h).<br>"
    "Good for same_person_as.<br><br>"
    "Antisymmetry: <b style='color:#b91c1c'>Weak</b><br>"
    "Direction is not naturally represented.<br><br>"
    "Inversion: <b style='color:#b91c1c'>Weak</b><br>"
    "Hard to distinguish inverse relations.<br><br>"
    "Composition: <b style='color:#ca8a04'>Moderate</b><br>"
    "Can learn correlations, but composition<br>"
    "is not geometrically explicit."
)


COMPLEX_PROPERTIES = (
    "<b>Relation properties</b><br><br>"
    "Symmetry: <b style='color:#15803d'>Good</b><br>"
    "Can model symmetric relations.<br><br>"
    "Antisymmetry: <b style='color:#15803d'>Good</b><br>"
    "Forward and reverse scores may differ.<br><br>"
    "Inversion: <b style='color:#15803d'>Good</b><br>"
    "Complex conjugation helps encode<br>"
    "inverse roles.<br><br>"
    "Composition: <b style='color:#ca8a04'>Moderate</b><br>"
    "Can learn composed patterns, though<br>"
    "less explicitly than vector addition."
)


CENSUS_PATTERNS = (
    "<b>Census relation examples</b><br>"
    "Symmetry: John Smith 1880 <b>same_person_as</b> J. Smith 1890<br>"
    "Antisymmetry: Mary Smith <b>parent_of</b> John Smith<br>"
    "Inversion: parent_of / child_of<br>"
    "Composition: lives_in + located_in ⇒ lives_in_country"
)


# ---------------------------------------------------------------------
# DistMult geometric visualization
# ---------------------------------------------------------------------

def make_distmult_geometric_plot(
    h: np.ndarray | None = None,
    r: np.ndarray | None = None,
    t: np.ndarray | None = None,
    head_label: str = "John Smith<br>Census 1880",
    tail_label: str = "J. Smith<br>Census 1890",
    relation_label: str = "same_person_as",
    width: int = 1150,
    height: int = 700,
) -> go.Figure:
    """
    Create a geometric Plotly visualization of DistMult.

    DistMult can be written as:

        score(h,r,t) = sum_i h_i r_i t_i
                     = (h ⊙ r)^T t

    Geometric interpretation:
    - r rescales each dimension of h.
    - This produces the relation-specific vector h ⊙ r.
    - The score is the dot product between h ⊙ r and t.
    - A high score means h ⊙ r and t are well aligned.

    Parameters
    ----------
    h, r, t:
        2D vectors used for visualization.
        If None, default illustrative vectors are used.

    head_label, tail_label, relation_label:
        Semantic census labels used in the annotation panels.

    width, height:
        Plot size.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    if h is None:
        h = np.array([1.0, 0.7], dtype=float)
    if r is None:
        r = np.array([1.6, 0.5], dtype=float)
    if t is None:
        t = np.array([1.3, 0.4], dtype=float)

    h = np.asarray(h, dtype=float)
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)

    if h.shape != (2,) or r.shape != (2,) or t.shape != (2,):
        raise ValueError("For this 2D visualization, h, r, and t must be arrays of shape (2,).")

    h_rel = h * r

    score = score_distmult(h, r, t)
    cos = cosine_similarity(h_rel, t)

    fig = go.Figure()

    # Axes.
    fig.add_shape(type="line", x0=-0.3, y0=0, x1=2.6, y1=0, line=dict(width=1))
    fig.add_shape(type="line", x0=0, y0=-0.3, x1=0, y1=2.1, line=dict(width=1))

    # h vector.
    fig.add_annotation(
        x=h[0],
        y=h[1],
        ax=0,
        ay=0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowwidth=3,
        text="",
    )

    # h relation-scaled vector.
    fig.add_annotation(
        x=h_rel[0],
        y=h_rel[1],
        ax=0,
        ay=0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowwidth=4,
        text="",
    )

    # t vector.
    fig.add_annotation(
        x=t[0],
        y=t[1],
        ax=0,
        ay=0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowwidth=3,
        text="",
    )

    # Key vector labels.
    fig.add_trace(
        go.Scatter(
            x=[h[0], h_rel[0], t[0]],
            y=[h[1], h_rel[1], t[1]],
            mode="markers+text",
            text=["h", "h ⊙ r", "t"],
            textposition=["top center", "top center", "bottom center"],
            marker=dict(
                size=[12, 14, 12],
                symbol=["circle", "circle-open", "circle"],
            ),
            name="vectors",
        )
    )

    # Projection of t onto h_rel direction.
    u = h_rel / (np.linalg.norm(h_rel) + 1e-12)
    proj_len = np.dot(t, u)
    proj = proj_len * u

    fig.add_shape(
        type="line",
        x0=t[0],
        y0=t[1],
        x1=proj[0],
        y1=proj[1],
        line=dict(dash="dash", width=2),
    )

    fig.add_trace(
        go.Scatter(
            x=[proj[0]],
            y=[proj[1]],
            mode="markers+text",
            text=["projection"],
            textposition="bottom center",
            marker=dict(size=10, symbol="diamond"),
            name="projection of t",
        )
    )

    # Explanation panel.
    fig.add_annotation(
        x=1.03,
        y=1.00,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        bordercolor="#7c3aed",
        borderwidth=1,
        bgcolor="rgba(250,245,255,0.98)",
        width=350,
        text=(
            "<b>DistMult geometry</b><br><br>"
            "<b>Scoring function</b><br>"
            "score(h,r,t) = Σ hᵢ rᵢ tᵢ<br>"
            "= (h ⊙ r)ᵀ t<br><br>"
            f"<b>Score:</b> {score:.3f}<br>"
            f"<b>cos(h ⊙ r, t):</b> {cos:.3f}<br><br>"
            "<b>Geometric intuition</b><br>"
            "1. r rescales each dimension of h.<br>"
            "2. This gives h ⊙ r.<br>"
            "3. The score is a dot product with t.<br><br>"
            "<b>High score means</b><br>"
            "h ⊙ r and t are well aligned."
        ),
        font=dict(size=12),
    )

    # Properties panel.
    fig.add_annotation(
        x=1.03,
        y=0.38,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        bordercolor="#64748b",
        borderwidth=1,
        bgcolor="rgba(248,250,252,0.98)",
        width=350,
        text=DISTMULT_PROPERTIES,
        font=dict(size=11),
    )

    # Census example panel.
    fig.add_annotation(
        x=0.5,
        y=-0.20,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        align="center",
        showarrow=False,
        bordercolor="#94a3b8",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.98)",
        width=720,
        text=(
            "<b>Census example</b><br>"
            f"h: {head_label.replace('<br>', ' ')} &nbsp;&nbsp; | &nbsp;&nbsp;"
            f"r: {relation_label} &nbsp;&nbsp; | &nbsp;&nbsp;"
            f"t: {tail_label.replace('<br>', ' ')}"
        ),
        font=dict(size=11),
    )

    fig.update_layout(
        title="DistMult: relation-weighted dot product",
        width=width,
        height=height,
        margin=dict(l=70, r=410, t=90, b=160),
        xaxis=dict(
            range=[-0.3, 2.6],
            title="embedding dimension 1",
            zeroline=False,
        ),
        yaxis=dict(
            range=[-0.3, 2.1],
            title="embedding dimension 2",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------
# ComplEx geometric visualization
# ---------------------------------------------------------------------

def make_complex_geometric_plot(
    h: complex = 1.0 + 0.5j,
    r: complex = 0.9 + 0.7j,
    t: complex = 0.6 + 1.1j,
    head_label: str = "Mary Smith<br>Census 1880",
    tail_label: str = "John Smith<br>Census 1880",
    relation_label: str = "parent_of",
    width: int = 1150,
    height: int = 700,
) -> go.Figure:
    """
    Create a geometric Plotly visualization of one ComplEx dimension.

    ComplEx score for one complex dimension:

        score(h,r,t) = Re(h * r * conjugate(t))

    Geometric interpretation:
    - h, r, and t are complex numbers.
    - Multiplying h by r rotates and scales h.
    - The model compares h * r with t.
    - Because complex multiplication is directional, the forward
      and reverse scores may differ.

    Parameters
    ----------
    h, r, t:
        Complex numbers used for one-dimensional visualization.

    head_label, tail_label, relation_label:
        Semantic census labels used in the annotation panels.

    width, height:
        Plot size.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    h = complex(h)
    r = complex(r)
    t = complex(t)

    h_times_r = h * r

    score_forward = score_complex_1d(h, r, t)
    score_reverse = score_complex_1d(t, r, h)

    fig = go.Figure()

    values = {
        "h": h,
        "r": r,
        "h × r": h_times_r,
        "t": t,
    }

    # Axes.
    fig.add_shape(type="line", x0=-1.6, y0=0, x1=2.1, y1=0, line=dict(width=1))
    fig.add_shape(type="line", x0=0, y0=-1.6, x1=0, y1=2.1, line=dict(width=1))

    # Vectors from origin.
    for z in values.values():
        fig.add_annotation(
            x=np.real(z),
            y=np.imag(z),
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowwidth=3,
            text="",
        )

    fig.add_trace(
        go.Scatter(
            x=[np.real(z) for z in values.values()],
            y=[np.imag(z) for z in values.values()],
            mode="markers+text",
            text=list(values.keys()),
            textposition="top center",
            marker=dict(
                size=[12, 12, 14, 12],
                symbol=["circle", "circle", "circle-open", "circle"],
            ),
            name="complex vectors",
        )
    )

    # Visual comparison between h*r and t.
    fig.add_shape(
        type="line",
        x0=np.real(h_times_r),
        y0=np.imag(h_times_r),
        x1=np.real(t),
        y1=np.imag(t),
        line=dict(dash="dash", width=2),
    )

    fig.add_annotation(
        x=(np.real(h) + np.real(h_times_r)) / 2,
        y=(np.imag(h) + np.imag(h_times_r)) / 2 + 0.15,
        text="multiply by r<br>= rotate + scale",
        showarrow=False,
        font=dict(size=12),
    )

    # Explanation panel.
    fig.add_annotation(
        x=1.03,
        y=1.00,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        bordercolor="#0f766e",
        borderwidth=1,
        bgcolor="rgba(240,253,250,0.98)",
        width=360,
        text=(
            "<b>ComplEx geometry</b><br><br>"
            "<b>Scoring function</b><br>"
            "score(h,r,t) = Re(h · r · conj(t))<br><br>"
            f"<b>Forward score:</b> {score_forward:.3f}<br>"
            f"<b>Reverse score:</b> {score_reverse:.3f}<br><br>"
            "<b>Geometric intuition</b><br>"
            "1. Each dimension is a complex plane.<br>"
            "2. Multiplying by r rotates and scales h.<br>"
            "3. The model compares h·r with t.<br><br>"
            "<b>Why this matters</b><br>"
            "The forward score can differ from<br>"
            "the reverse score."
        ),
        font=dict(size=12),
    )

    # Properties panel.
    fig.add_annotation(
        x=1.03,
        y=0.38,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        bordercolor="#64748b",
        borderwidth=1,
        bgcolor="rgba(248,250,252,0.98)",
        width=360,
        text=COMPLEX_PROPERTIES,
        font=dict(size=11),
    )

    # Census example panel.
    fig.add_annotation(
        x=0.5,
        y=-0.20,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        align="center",
        showarrow=False,
        bordercolor="#94a3b8",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.98)",
        width=720,
        text=(
            "<b>Census example</b><br>"
            f"h: {head_label.replace('<br>', ' ')} &nbsp;&nbsp; | &nbsp;&nbsp;"
            f"r: {relation_label} &nbsp;&nbsp; | &nbsp;&nbsp;"
            f"t: {tail_label.replace('<br>', ' ')}<br>"
            "Useful for directional relations such as parent_of / child_of."
        ),
        font=dict(size=11),
    )

    fig.update_layout(
        title="ComplEx: rotation and scaling in the complex plane",
        width=width,
        height=height,
        margin=dict(l=70, r=430, t=90, b=160),
        xaxis=dict(
            range=[-1.6, 2.1],
            title="real axis",
            zeroline=False,
        ),
        yaxis=dict(
            range=[-1.6, 2.1],
            title="imaginary axis",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------
# Combined dropdown figure
# ---------------------------------------------------------------------

def plot_bilinear_geometric_models(
    distmult_h: np.ndarray | None = None,
    distmult_r: np.ndarray | None = None,
    distmult_t: np.ndarray | None = None,
    complex_h: complex = 1.0 + 0.5j,
    complex_r: complex = 0.9 + 0.7j,
    complex_t: complex = 0.6 + 1.1j,
    distmult_head_label: str = "John Smith<br>Census 1880",
    distmult_tail_label: str = "J. Smith<br>Census 1890",
    distmult_relation_label: str = "same_person_as",
    complex_head_label: str = "Mary Smith<br>Census 1880",
    complex_tail_label: str = "John Smith<br>Census 1880",
    complex_relation_label: str = "parent_of",
    width: int = 1150,
    height: int = 700,
) -> go.Figure:
    """
    Create one interactive Plotly figure with a dropdown for DistMult and ComplEx.

    This version avoids ipywidgets and should work in Jupyter using Plotly alone.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    # Build individual figures first.
    distmult_fig = make_distmult_geometric_plot(
        h=distmult_h,
        r=distmult_r,
        t=distmult_t,
        head_label=distmult_head_label,
        tail_label=distmult_tail_label,
        relation_label=distmult_relation_label,
        width=width,
        height=height,
    )

    complex_fig = make_complex_geometric_plot(
        h=complex_h,
        r=complex_r,
        t=complex_t,
        head_label=complex_head_label,
        tail_label=complex_tail_label,
        relation_label=complex_relation_label,
        width=width,
        height=height,
    )

    fig = go.Figure()

    # Add DistMult traces and shapes.
    for trace in distmult_fig.data:
        trace.visible = True
        fig.add_trace(trace)

    n_distmult_traces = len(distmult_fig.data)

    # Add ComplEx traces.
    for trace in complex_fig.data:
        trace.visible = False
        fig.add_trace(trace)

    n_complex_traces = len(complex_fig.data)

    visible_distmult = [True] * n_distmult_traces + [False] * n_complex_traces
    visible_complex = [False] * n_distmult_traces + [True] * n_complex_traces

    # Shapes must also change with the dropdown.
    distmult_shapes = list(distmult_fig.layout.shapes)
    complex_shapes = list(complex_fig.layout.shapes)

    distmult_annotations = list(distmult_fig.layout.annotations)
    complex_annotations = list(complex_fig.layout.annotations)

    fig.update_layout(
        title="DistMult: relation-weighted dot product",
        annotations=distmult_annotations,
        shapes=distmult_shapes,
        width=width,
        height=height,
        margin=dict(l=70, r=430, t=90, b=160),
        xaxis=dict(
            range=[-0.3, 2.6],
            title="embedding dimension 1",
            zeroline=False,
        ),
        yaxis=dict(
            range=[-0.3, 2.1],
            title="embedding dimension 2",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False,
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="DistMult",
                        method="update",
                        args=[
                            {"visible": visible_distmult},
                            {
                                "title": "DistMult: relation-weighted dot product",
                                "annotations": distmult_annotations,
                                "shapes": distmult_shapes,
                                "xaxis": {
                                    "range": [-0.3, 2.6],
                                    "title": "embedding dimension 1",
                                    "zeroline": False,
                                },
                                "yaxis": {
                                    "range": [-0.3, 2.1],
                                    "title": "embedding dimension 2",
                                    "zeroline": False,
                                    "scaleanchor": "x",
                                    "scaleratio": 1,
                                },
                            },
                        ],
                    ),
                    dict(
                        label="ComplEx",
                        method="update",
                        args=[
                            {"visible": visible_complex},
                            {
                                "title": "ComplEx: rotation and scaling in the complex plane",
                                "annotations": complex_annotations,
                                "shapes": complex_shapes,
                                "xaxis": {
                                    "range": [-1.6, 2.1],
                                    "title": "real axis",
                                    "zeroline": False,
                                },
                                "yaxis": {
                                    "range": [-1.6, 2.1],
                                    "title": "imaginary axis",
                                    "zeroline": False,
                                    "scaleanchor": "x",
                                    "scaleratio": 1,
                                },
                            },
                        ],
                    ),
                ],
                direction="down",
                x=0.01,
                y=1.15,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
    )

    return fig
