"""
kg_bilinear_geometry.py

Interactive Plotly visualizations for bilinear / semantic matching
knowledge graph embedding models:

- DistMult
- ComplEx

Designed for Jupyter notebook tutorials.

Usage
-----
from kg_bilinear_geometry import plot_bilinear_models_clean

fig = plot_bilinear_models_clean()
fig.show()

If Plotly does not render:
fig.show(renderer="notebook_connected")
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def score_distmult(h: np.ndarray, r: np.ndarray, t: np.ndarray) -> float:
    """
    DistMult score:

        score(h, r, t) = sum_i h_i r_i t_i

    DistMult uses multiplicative interactions between corresponding
    dimensions of h, r, and t.
    """
    return float(np.sum(h * r * t))


def score_complex(
    h_re: np.ndarray,
    h_im: np.ndarray,
    r_re: np.ndarray,
    r_im: np.ndarray,
    t_re: np.ndarray,
    t_im: np.ndarray,
) -> float:
    """
    ComplEx score:

        score(h, r, t) = Re(<h, r, conjugate(t)>)

    Expanded form:

        sum_i [
            h_re_i r_re_i t_re_i
          + h_re_i r_im_i t_im_i
          + h_im_i r_re_i t_im_i
          - h_im_i r_im_i t_re_i
        ]

    ComplEx extends DistMult by using complex-valued embeddings.
    The imaginary part allows asymmetric relations.
    """
    contribution = (
        h_re * r_re * t_re
        + h_re * r_im * t_im
        + h_im * r_re * t_im
        - h_im * r_im * t_re
    )
    return float(np.sum(contribution))


def complex_dimension_contributions(
    h_re: np.ndarray,
    h_im: np.ndarray,
    r_re: np.ndarray,
    r_im: np.ndarray,
    t_re: np.ndarray,
    t_im: np.ndarray,
) -> np.ndarray:
    """Per-dimension contribution to the ComplEx score."""
    return (
        h_re * r_re * t_re
        + h_re * r_im * t_im
        + h_im * r_re * t_im
        - h_im * r_im * t_re
    )


PROPERTY_TEXT = {
    "DistMult": (
        "<b>DistMult properties</b><br><br>"
        "Symmetry: <b style='color:#15803d'>Strong</b><br>"
        "score(h,r,t) = score(t,r,h).<br>"
        "Good for same_person_as.<br><br>"
        "Antisymmetry: <b style='color:#b91c1c'>Weak</b><br>"
        "Cannot naturally distinguish<br>"
        "parent_of from its reverse.<br><br>"
        "Inversion: <b style='color:#b91c1c'>Weak</b><br>"
        "Inverse relations are hard because<br>"
        "the score is symmetric in h and t.<br><br>"
        "Composition: <b style='color:#ca8a04'>Moderate</b><br>"
        "Can capture some regularities,<br>"
        "but composition is not explicit."
    ),
    "ComplEx": (
        "<b>ComplEx properties</b><br><br>"
        "Symmetry: <b style='color:#15803d'>Good</b><br>"
        "Can model symmetric relations<br>"
        "when imaginary interaction is small.<br><br>"
        "Antisymmetry: <b style='color:#15803d'>Good</b><br>"
        "The imaginary part allows directionality.<br><br>"
        "Inversion: <b style='color:#15803d'>Good</b><br>"
        "Inverse relations can be represented<br>"
        "through complex conjugate structure.<br><br>"
        "Composition: <b style='color:#ca8a04'>Moderate</b><br>"
        "More expressive than DistMult,<br>"
        "but composition is still indirect."
    ),
}


CENSUS_EXAMPLES = (
    "<b>Census examples</b><br>"
    "Symmetry: John Smith 1880 <b>same_person_as</b> J. Smith 1890 &nbsp;&nbsp; | &nbsp;&nbsp;"
    "Antisymmetry: Mary Smith <b>parent_of</b> John Smith<br>"
    "Inversion: parent_of / child_of &nbsp;&nbsp; | &nbsp;&nbsp;"
    "Composition: lives_in + located_in ⇒ lives_in_country"
)


def _clean_label(label: str) -> str:
    return label.replace("<br>", " ").replace("<b>", "").replace("</b>", "")


def make_distmult_clean(
    h: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    head_label: str,
    tail_label: str,
    relation_label: str,
):
    """
    Build traces and annotations for DistMult.

    The figure shows:
    - vector components of h, r, and t
    - per-dimension multiplicative interactions h_i r_i t_i
    - the final score as the sum of contributions
    """
    contribution = h * r * t
    score = score_distmult(h, r, t)
    dims = [f"d{i+1}" for i in range(len(h))]

    traces = [
        go.Bar(x=dims, y=h, name="h: head", visible=True, offsetgroup="h"),
        go.Bar(x=dims, y=r, name="r: relation", visible=True, offsetgroup="r"),
        go.Bar(x=dims, y=t, name="t: tail", visible=True, offsetgroup="t"),
        go.Scatter(
            x=dims,
            y=contribution,
            mode="lines+markers+text",
            text=[f"{v:.2f}" for v in contribution],
            textposition="top center",
            name="hᵢ rᵢ tᵢ",
            visible=True,
            line=dict(width=4),
        ),
    ]

    annotations = [
        dict(
            x=0.5,
            y=1.11,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            align="center",
            showarrow=False,
            text=(
                "<b>DistMult geometric intuition</b><br>"
                "Each dimension is a semantic feature. "
                "The relation weights which dimensions matter."
            ),
            font=dict(size=13),
        ),
        dict(
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
            width=330,
            text=(
                "<b>DistMult</b><br><br>"
                "<b>Scoring function</b><br>"
                "score(h,r,t) = Σ hᵢ rᵢ tᵢ<br><br>"
                f"<b>Score:</b> {score:.3f}<br><br>"
                "<b>Interpretation</b><br>"
                "A triple is plausible when h and t<br>"
                "activate similar dimensions that are<br>"
                "important for relation r.<br><br>"
                "<b>Census example</b><br>"
                f"h: {_clean_label(head_label)}<br>"
                f"r: {relation_label}<br>"
                f"t: {_clean_label(tail_label)}"
            ),
            font=dict(size=12),
        ),
        dict(
            x=1.03,
            y=0.43,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bordercolor="#64748b",
            borderwidth=1,
            bgcolor="rgba(248,250,252,0.98)",
            width=330,
            text=PROPERTY_TEXT["DistMult"],
            font=dict(size=11),
        ),
        dict(
            x=0.5,
            y=-0.25,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            align="center",
            showarrow=False,
            bordercolor="#94a3b8",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.98)",
            width=780,
            text=CENSUS_EXAMPLES,
            font=dict(size=11),
        ),
    ]

    title = "DistMult: multiplicative semantic matching"
    return traces, annotations, title


def make_complex_clean(
    h_re: np.ndarray,
    h_im: np.ndarray,
    r_re: np.ndarray,
    r_im: np.ndarray,
    t_re: np.ndarray,
    t_im: np.ndarray,
    head_label: str,
    tail_label: str,
    relation_label: str,
):
    """
    Build traces and annotations for ComplEx.

    The figure shows:
    - real and imaginary parts of the head and tail embeddings
    - how each dimension contributes to the final score
    - why imaginary components help represent directionality
    """
    contribution = complex_dimension_contributions(h_re, h_im, r_re, r_im, t_re, t_im)
    score = score_complex(h_re, h_im, r_re, r_im, t_re, t_im)
    dims = [f"d{i+1}" for i in range(len(h_re))]

    traces = [
        go.Bar(x=dims, y=h_re, name="Re(h)", visible=True, offsetgroup="h_re"),
        go.Bar(x=dims, y=h_im, name="Im(h)", visible=True, offsetgroup="h_im"),
        go.Bar(x=dims, y=t_re, name="Re(t)", visible=True, offsetgroup="t_re"),
        go.Bar(x=dims, y=t_im, name="Im(t)", visible=True, offsetgroup="t_im"),
        go.Scatter(
            x=dims,
            y=contribution,
            mode="lines+markers+text",
            text=[f"{v:.2f}" for v in contribution],
            textposition="top center",
            name="per-dim score",
            visible=True,
            line=dict(width=4),
        ),
    ]

    annotations = [
        dict(
            x=0.5,
            y=1.11,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            align="center",
            showarrow=False,
            text=(
                "<b>ComplEx geometric intuition</b><br>"
                "Embeddings have real and imaginary parts. "
                "The imaginary part introduces direction."
            ),
            font=dict(size=13),
        ),
        dict(
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
            width=330,
            text=(
                "<b>ComplEx</b><br><br>"
                "<b>Scoring function</b><br>"
                "score(h,r,t) = Re(&lt;h,r,conj(t)&gt;)<br><br>"
                f"<b>Score:</b> {score:.3f}<br><br>"
                "<b>Interpretation</b><br>"
                "Real parts capture semantic similarity.<br>"
                "Imaginary parts help encode direction,<br>"
                "so parent_of and child_of can differ.<br><br>"
                "<b>Census example</b><br>"
                f"h: {_clean_label(head_label)}<br>"
                f"r: {relation_label}<br>"
                f"t: {_clean_label(tail_label)}"
            ),
            font=dict(size=12),
        ),
        dict(
            x=1.03,
            y=0.43,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bordercolor="#64748b",
            borderwidth=1,
            bgcolor="rgba(248,250,252,0.98)",
            width=330,
            text=PROPERTY_TEXT["ComplEx"],
            font=dict(size=11),
        ),
        dict(
            x=0.5,
            y=-0.25,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            align="center",
            showarrow=False,
            bordercolor="#94a3b8",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.98)",
            width=780,
            text=CENSUS_EXAMPLES,
            font=dict(size=11),
        ),
    ]

    title = "ComplEx: complex-valued semantic matching"
    return traces, annotations, title


def plot_bilinear_models_clean(
    h: np.ndarray | None = None,
    r: np.ndarray | None = None,
    t: np.ndarray | None = None,
    h_re: np.ndarray | None = None,
    h_im: np.ndarray | None = None,
    r_re: np.ndarray | None = None,
    r_im: np.ndarray | None = None,
    t_re: np.ndarray | None = None,
    t_im: np.ndarray | None = None,
    head_label: str = "John Smith<br>Census 1880",
    tail_label: str = "J. Smith<br>Census 1890",
    relation_label: str = "same_person_as",
) -> go.Figure:
    """
    Create a clean interactive Plotly figure comparing DistMult and ComplEx.

    Parameters
    ----------
    h, r, t:
        Real-valued embeddings for DistMult.
        If None, small illustrative vectors are used.

    h_re, h_im, r_re, r_im, t_re, t_im:
        Real and imaginary parts for ComplEx embeddings.
        If None, illustrative vectors are used.

    head_label, tail_label, relation_label:
        Semantic labels shown in the plot.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with a dropdown to switch between DistMult and ComplEx.
    """
    if h is None:
        h = np.array([0.9, 0.7, 0.2, 0.8])
    if r is None:
        r = np.array([1.2, 1.0, 0.3, 1.1])
    if t is None:
        t = np.array([0.8, 0.6, 0.1, 0.9])

    if h_re is None:
        h_re = np.array([0.9, 0.7, 0.2, 0.8])
    if h_im is None:
        h_im = np.array([0.1, 0.4, 0.8, 0.2])
    if r_re is None:
        r_re = np.array([1.1, 0.9, 0.3, 1.0])
    if r_im is None:
        r_im = np.array([0.2, -0.6, 0.7, -0.3])
    if t_re is None:
        t_re = np.array([0.8, 0.6, 0.1, 0.9])
    if t_im is None:
        t_im = np.array([0.2, -0.3, 0.7, 0.1])

    h = np.asarray(h, dtype=float)
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)

    h_re = np.asarray(h_re, dtype=float)
    h_im = np.asarray(h_im, dtype=float)
    r_re = np.asarray(r_re, dtype=float)
    r_im = np.asarray(r_im, dtype=float)
    t_re = np.asarray(t_re, dtype=float)
    t_im = np.asarray(t_im, dtype=float)

    distmult_traces, distmult_ann, distmult_title = make_distmult_clean(
        h=h,
        r=r,
        t=t,
        head_label=head_label,
        tail_label=tail_label,
        relation_label=relation_label,
    )

    complex_traces, complex_ann, complex_title = make_complex_clean(
        h_re=h_re,
        h_im=h_im,
        r_re=r_re,
        r_im=r_im,
        t_re=t_re,
        t_im=t_im,
        head_label=head_label,
        tail_label=tail_label,
        relation_label=relation_label,
    )

    fig = go.Figure()

    for trace in distmult_traces:
        trace.visible = True
        fig.add_trace(trace)

    for trace in complex_traces:
        trace.visible = False
        fig.add_trace(trace)

    n_distmult = len(distmult_traces)
    n_complex = len(complex_traces)

    visible_distmult = [True] * n_distmult + [False] * n_complex
    visible_complex = [False] * n_distmult + [True] * n_complex

    fig.update_layout(
        title=distmult_title,
        annotations=distmult_ann,
        width=1200,
        height=760,
        margin=dict(l=70, r=410, t=100, b=170),
        barmode="group",
        xaxis=dict(title="embedding dimension", zeroline=True),
        yaxis=dict(title="component value / contribution", zeroline=True),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="DistMult",
                        method="update",
                        args=[
                            {"visible": visible_distmult},
                            {"title": distmult_title, "annotations": distmult_ann, "barmode": "group"},
                        ],
                    ),
                    dict(
                        label="ComplEx",
                        method="update",
                        args=[
                            {"visible": visible_complex},
                            {"title": complex_title, "annotations": complex_ann, "barmode": "group"},
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


def plot_distmult_only(
    h: np.ndarray | None = None,
    r: np.ndarray | None = None,
    t: np.ndarray | None = None,
    head_label: str = "John Smith<br>Census 1880",
    tail_label: str = "J. Smith<br>Census 1890",
    relation_label: str = "same_person_as",
) -> go.Figure:
    """Create a standalone DistMult figure."""
    if h is None:
        h = np.array([0.9, 0.7, 0.2, 0.8])
    if r is None:
        r = np.array([1.2, 1.0, 0.3, 1.1])
    if t is None:
        t = np.array([0.8, 0.6, 0.1, 0.9])

    traces, annotations, title = make_distmult_clean(
        np.asarray(h, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(t, dtype=float),
        head_label,
        tail_label,
        relation_label,
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        annotations=annotations,
        width=1200,
        height=760,
        margin=dict(l=70, r=410, t=100, b=170),
        barmode="group",
        xaxis=dict(title="embedding dimension"),
        yaxis=dict(title="component value / contribution"),
    )
    return fig


def plot_complex_only(
    h_re: np.ndarray | None = None,
    h_im: np.ndarray | None = None,
    r_re: np.ndarray | None = None,
    r_im: np.ndarray | None = None,
    t_re: np.ndarray | None = None,
    t_im: np.ndarray | None = None,
    head_label: str = "Mary Smith<br>Census 1880",
    tail_label: str = "John Smith<br>Census 1880",
    relation_label: str = "parent_of",
) -> go.Figure:
    """Create a standalone ComplEx figure."""
    if h_re is None:
        h_re = np.array([0.9, 0.7, 0.2, 0.8])
    if h_im is None:
        h_im = np.array([0.1, 0.4, 0.8, 0.2])
    if r_re is None:
        r_re = np.array([1.1, 0.9, 0.3, 1.0])
    if r_im is None:
        r_im = np.array([0.2, -0.6, 0.7, -0.3])
    if t_re is None:
        t_re = np.array([0.8, 0.6, 0.1, 0.9])
    if t_im is None:
        t_im = np.array([0.2, -0.3, 0.7, 0.1])

    traces, annotations, title = make_complex_clean(
        np.asarray(h_re, dtype=float),
        np.asarray(h_im, dtype=float),
        np.asarray(r_re, dtype=float),
        np.asarray(r_im, dtype=float),
        np.asarray(t_re, dtype=float),
        np.asarray(t_im, dtype=float),
        head_label,
        tail_label,
        relation_label,
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        annotations=annotations,
        width=1200,
        height=760,
        margin=dict(l=70, r=410, t=100, b=170),
        barmode="group",
        xaxis=dict(title="embedding dimension"),
        yaxis=dict(title="component value / contribution"),
    )
    return fig
