"""
kg_translation_geometry.py

Reusable Plotly visualizations for teaching the geometric intuition of
translation-based Knowledge Graph Embedding models:

- TransE
- TransH
- TransR

The module is designed for Jupyter notebooks. It does not require ipywidgets;
interactivity is handled with Plotly dropdown menus.

Example
-------
from kg_translation_geometry import plot_translation_models_clean

fig = plot_translation_models_clean()
fig.show()
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------


def lp_norm(x: np.ndarray, p: str = "L2") -> float:
    """
    Compute the L1 or L2 norm of a vector.

    Parameters
    ----------
    x:
        Input vector.
    p:
        Either "L1" or "L2".

    Returns
    -------
    float
        Norm value.
    """
    if p == "L1":
        return float(np.sum(np.abs(x)))
    if p == "L2":
        return float(np.linalg.norm(x))
    raise ValueError("p must be either 'L1' or 'L2'")



def project_transh(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the hyperplane orthogonal to w.

    TransH uses a relation-specific hyperplane. The normal vector w defines
    the hyperplane, and entities are projected before applying the translation.
    """
    w = w / (np.linalg.norm(w) + 1e-12)
    return x - np.dot(w, x) * w



def project_transr(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Project an entity vector into a relation-specific space.

    TransR uses a relation-specific projection matrix M_r.
    """
    return matrix @ x


# -----------------------------------------------------------------------------
# Text panels
# -----------------------------------------------------------------------------


PROPERTY_TEXT = {
    "TransE": (
        "<b>TransE properties</b><br><br>"
        "Symmetry: <b style='color:#b91c1c'>Weak</b><br>"
        "Difficult because h + r ≈ t and t + r ≈ h<br>"
        "usually forces r close to zero.<br><br>"
        "Antisymmetry: <b style='color:#15803d'>Good</b><br>"
        "A non-zero relation vector gives direction.<br><br>"
        "Inversion: <b style='color:#15803d'>Good</b><br>"
        "Inverse relations can be opposite vectors:<br>"
        "r_child_of ≈ -r_parent_of.<br><br>"
        "Composition: <b style='color:#15803d'>Good</b><br>"
        "Relations can compose by vector addition."
    ),
    "TransH": (
        "<b>TransH properties</b><br><br>"
        "Symmetry: <b style='color:#ca8a04'>Moderate</b><br>"
        "Projection gives more flexibility than TransE.<br><br>"
        "Antisymmetry: <b style='color:#15803d'>Good</b><br>"
        "Translation on the hyperplane keeps direction.<br><br>"
        "Inversion: <b style='color:#15803d'>Good</b><br>"
        "Inverse relations can use opposite translations.<br><br>"
        "Composition: <b style='color:#ca8a04'>Moderate</b><br>"
        "Possible, but projections make it less direct."
    ),
    "TransR": (
        "<b>TransR properties</b><br><br>"
        "Symmetry: <b style='color:#ca8a04'>Moderate</b><br>"
        "Relation-specific spaces increase flexibility.<br><br>"
        "Antisymmetry: <b style='color:#15803d'>Good</b><br>"
        "Directional relations can be represented well.<br><br>"
        "Inversion: <b style='color:#15803d'>Good</b><br>"
        "Inverse relations can use different projections<br>"
        "and opposite translations.<br><br>"
        "Composition: <b style='color:#15803d'>Good</b><br>"
        "More expressive for composed patterns."
    ),
}


DEFAULT_CENSUS_PANEL = (
    "<b>Census examples</b><br>"
    "Symmetry: John Smith 1880 <b>same_person_as</b> J. Smith 1890 &nbsp;&nbsp; | &nbsp;&nbsp;"
    "Antisymmetry: Mary Smith <b>parent_of</b> John Smith<br>"
    "Inversion: parent_of / child_of &nbsp;&nbsp; | &nbsp;&nbsp;"
    "Composition: lives_in + located_in ⇒ lives_in_country"
)


# -----------------------------------------------------------------------------
# Model-specific geometry builders
# -----------------------------------------------------------------------------


def make_transe_clean(
    h: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    norm_type: str = "L2",
    head_label: str = "h<br>John 1880",
    tail_label: str = "t<br>John 1890",
    relation_label: str = "same_person_as",
):
    """
    Create traces and annotations for TransE.

    TransE scoring function:
        score(h, r, t) = -||h + r - t||
    """
    h_plus_r = h + r
    distance = lp_norm(h_plus_r - t, norm_type)
    score = -distance

    traces = [
        go.Scatter(
            x=[h[0], t[0], h_plus_r[0]],
            y=[h[1], t[1], h_plus_r[1]],
            mode="markers+text",
            text=[head_label, tail_label, "h + r"],
            textposition=["bottom center", "top center", "top center"],
            marker=dict(
                size=[18, 18, 16],
                symbol=["circle", "circle", "circle-open"],
            ),
            name="entities",
        ),
        go.Scatter(
            x=[h[0], h_plus_r[0]],
            y=[h[1], h_plus_r[1]],
            mode="lines+markers",
            line=dict(width=4),
            name="translation r",
        ),
        go.Scatter(
            x=[h_plus_r[0], t[0]],
            y=[h_plus_r[1], t[1]],
            mode="lines",
            line=dict(dash="dash", width=3),
            name="distance",
        ),
    ]

    annotations = [
        dict(
            x=(h[0] + h_plus_r[0]) / 2,
            y=(h[1] + h_plus_r[1]) / 2 + 0.3,
            text=f"<b>r</b><br>{relation_label}",
            showarrow=False,
            font=dict(size=12),
        )
    ]

    explanation = (
        "<b>Geometric intuition</b><br><br>"
        "The relation is a vector translation.<br>"
        "A triple is plausible when:<br><br>"
        "<b>h + r is close to t</b><br><br>"
        "Scoring function:<br>"
        "score(h,r,t) = -||h + r - t||<br><br>"
        f"Distance = {distance:.3f}<br>"
        f"Score = {score:.3f}"
    )

    return traces, annotations, "TransE: translation in entity space", explanation



def make_transh_clean(
    h: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    norm_type: str = "L2",
    head_label: str = "h<br>John 1880",
    tail_label: str = "t<br>John 1890",
    w: np.ndarray | None = None,
):
    """
    Create traces and annotations for TransH.

    TransH scoring function:
        score(h, r, t) = -||h_perp + r - t_perp||
    """
    if w is None:
        w = np.array([1.0, 1.0], dtype=float)
    w = w / np.linalg.norm(w)

    h_proj = project_transh(h, w)
    t_proj = project_transh(t, w)
    h_plus_r = h_proj + r

    distance = lp_norm(h_plus_r - t_proj, norm_type)
    score = -distance

    direction = np.array([-w[1], w[0]])
    line_pts = np.array([direction * -6, direction * 6])

    traces = [
        go.Scatter(
            x=line_pts[:, 0],
            y=line_pts[:, 1],
            mode="lines",
            line=dict(dash="dot", width=3),
            name="hyperplane",
        ),
        go.Scatter(
            x=[h[0], t[0]],
            y=[h[1], t[1]],
            mode="markers+text",
            text=[head_label, tail_label],
            textposition=["bottom center", "top center"],
            marker=dict(size=18),
            name="original entities",
        ),
        go.Scatter(
            x=[h_proj[0], t_proj[0], h_plus_r[0]],
            y=[h_proj[1], t_proj[1], h_plus_r[1]],
            mode="markers+text",
            text=["h⊥", "t⊥", "h⊥ + r"],
            textposition=["bottom center", "top center", "top center"],
            marker=dict(size=[15, 15, 16], symbol=["diamond", "diamond", "circle-open"]),
            name="projected entities",
        ),
        go.Scatter(
            x=[h[0], h_proj[0], None, t[0], t_proj[0]],
            y=[h[1], h_proj[1], None, t[1], t_proj[1]],
            mode="lines",
            line=dict(dash="dot", width=2),
            name="projection",
        ),
        go.Scatter(
            x=[h_proj[0], h_plus_r[0]],
            y=[h_proj[1], h_plus_r[1]],
            mode="lines+markers",
            line=dict(width=4),
            name="translation r",
        ),
        go.Scatter(
            x=[h_plus_r[0], t_proj[0]],
            y=[h_plus_r[1], t_proj[1]],
            mode="lines",
            line=dict(dash="dash", width=3),
            name="distance",
        ),
    ]

    annotations = [
        dict(
            x=w[0] * 2.4,
            y=w[1] * 2.4,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
        ),
        dict(
            x=w[0] * 2.75,
            y=w[1] * 2.75,
            text="<b>w</b>",
            showarrow=False,
            font=dict(size=12),
        ),
        dict(
            x=(h_proj[0] + h_plus_r[0]) / 2,
            y=(h_proj[1] + h_plus_r[1]) / 2 + 0.3,
            text="<b>r</b>",
            showarrow=False,
            font=dict(size=12),
        ),
    ]

    explanation = (
        "<b>Geometric intuition</b><br><br>"
        "Each relation defines a hyperplane.<br>"
        "Entities are first projected onto it:<br><br>"
        "<b>h → h⊥</b>, <b>t → t⊥</b><br><br>"
        "Then translation is applied:<br>"
        "<b>h⊥ + r ≈ t⊥</b><br><br>"
        "Scoring function:<br>"
        "score(h,r,t) = -||h⊥ + r - t⊥||<br><br>"
        f"Distance = {distance:.3f}<br>"
        f"Score = {score:.3f}"
    )

    return traces, annotations, "TransH: projection onto relation hyperplane", explanation



def make_transr_clean(
    h: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    norm_type: str = "L2",
    head_label: str = "h<br>John 1880",
    tail_label: str = "t<br>John 1890",
    matrix: np.ndarray | None = None,
):
    """
    Create traces and annotations for TransR.

    TransR scoring function:
        score(h, r, t) = -||M_r h + r - M_r t||
    """
    if matrix is None:
        matrix = np.array([[1.2, 0.3], [-0.2, 0.8]], dtype=float)

    h_proj = project_transr(h, matrix)
    t_proj = project_transr(t, matrix)
    h_plus_r = h_proj + r

    distance = lp_norm(h_plus_r - t_proj, norm_type)
    score = -distance

    traces = [
        go.Scatter(
            x=[h[0], t[0]],
            y=[h[1], t[1]],
            mode="markers+text",
            text=[head_label, tail_label],
            textposition=["bottom center", "top center"],
            marker=dict(size=18),
            name="entity space",
        ),
        go.Scatter(
            x=[h_proj[0], t_proj[0], h_plus_r[0]],
            y=[h_proj[1], t_proj[1], h_plus_r[1]],
            mode="markers+text",
            text=["Mᵣh", "Mᵣt", "Mᵣh + r"],
            textposition=["bottom center", "top center", "top center"],
            marker=dict(size=[15, 15, 16], symbol=["diamond", "diamond", "circle-open"]),
            name="relation space",
        ),
        go.Scatter(
            x=[h[0], h_proj[0], None, t[0], t_proj[0]],
            y=[h[1], h_proj[1], None, t[1], t_proj[1]],
            mode="lines",
            line=dict(dash="dot", width=2),
            name="projection Mᵣ",
        ),
        go.Scatter(
            x=[h_proj[0], h_plus_r[0]],
            y=[h_proj[1], h_plus_r[1]],
            mode="lines+markers",
            line=dict(width=4),
            name="translation r",
        ),
        go.Scatter(
            x=[h_plus_r[0], t_proj[0]],
            y=[h_plus_r[1], t_proj[1]],
            mode="lines",
            line=dict(dash="dash", width=3),
            name="distance",
        ),
    ]

    annotations = [
        dict(
            x=(h_proj[0] + h_plus_r[0]) / 2,
            y=(h_proj[1] + h_plus_r[1]) / 2 + 0.3,
            text="<b>r</b>",
            showarrow=False,
            font=dict(size=12),
        )
    ]

    explanation = (
        "<b>Geometric intuition</b><br><br>"
        "Entities start in entity space.<br>"
        "Each relation has a projection matrix Mᵣ.<br><br>"
        "The model first projects:<br>"
        "<b>h → Mᵣh</b>, <b>t → Mᵣt</b><br><br>"
        "Then translation is applied:<br>"
        "<b>Mᵣh + r ≈ Mᵣt</b><br><br>"
        "Scoring function:<br>"
        "score(h,r,t) = -||Mᵣh + r - Mᵣt||<br><br>"
        f"Distance = {distance:.3f}<br>"
        f"Score = {score:.3f}"
    )

    return traces, annotations, "TransR: relation-specific projection space", explanation


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------


def _full_annotations(model_annotations, explanation, properties, census_panel):
    """
    Compose plot annotations with explanation, properties, and semantic examples.
    """
    return model_annotations + [
        dict(
            x=1.03,
            y=1.00,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bordercolor="#334155",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.98)",
            width=330,
            text=explanation,
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
            bordercolor="#334155",
            borderwidth=1,
            bgcolor="rgba(248,250,252,0.98)",
            width=330,
            text=properties,
            font=dict(size=11),
        ),
        dict(
            x=0.5,
            y=-0.22,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            align="center",
            showarrow=False,
            bordercolor="#94a3b8",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.98)",
            width=760,
            text=census_panel,
            font=dict(size=11),
        ),
    ]



def plot_translation_models_clean(
    h: np.ndarray | None = None,
    r: np.ndarray | None = None,
    t: np.ndarray | None = None,
    norm_type: str = "L2",
    head_label: str = "h<br>John 1880",
    tail_label: str = "t<br>John 1890",
    relation_label: str = "same_person_as",
    transh_w: np.ndarray | None = None,
    transr_matrix: np.ndarray | None = None,
    census_panel: str = DEFAULT_CENSUS_PANEL,
    width: int = 1200,
    height: int = 780,
) -> go.Figure:
    """
    Create an interactive Plotly figure comparing TransE, TransH, and TransR.

    The figure uses a dropdown menu and does not require ipywidgets.

    Parameters
    ----------
    h:
        2D vector for the head entity.
    r:
        2D vector for the relation.
    t:
        2D vector for the tail entity.
    norm_type:
        Either "L1" or "L2".
    head_label:
        HTML label for the head entity.
    tail_label:
        HTML label for the tail entity.
    relation_label:
        Label for the relation vector in the TransE plot.
    transh_w:
        Optional normal vector for the TransH hyperplane.
    transr_matrix:
        Optional 2x2 projection matrix for TransR.
    census_panel:
        HTML string with semantic examples shown below the plot.
    width:
        Figure width in pixels.
    height:
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure.
    """
    if h is None:
        h = np.array([1.0, 1.0], dtype=float)
    if r is None:
        r = np.array([2.0, 1.0], dtype=float)
    if t is None:
        t = np.array([3.4, 2.7], dtype=float)

    h = np.array(h, dtype=float)
    r = np.array(r, dtype=float)
    t = np.array(t, dtype=float)

    if h.shape != (2,) or r.shape != (2,) or t.shape != (2,):
        raise ValueError("h, r, and t must be 2D vectors with shape (2,)")

    transe_traces, transe_ann, transe_title, transe_exp = make_transe_clean(
        h=h,
        r=r,
        t=t,
        norm_type=norm_type,
        head_label=head_label,
        tail_label=tail_label,
        relation_label=relation_label,
    )

    transh_traces, transh_ann, transh_title, transh_exp = make_transh_clean(
        h=h,
        r=r,
        t=t,
        norm_type=norm_type,
        head_label=head_label,
        tail_label=tail_label,
        w=transh_w,
    )

    transr_traces, transr_ann, transr_title, transr_exp = make_transr_clean(
        h=h,
        r=r,
        t=t,
        norm_type=norm_type,
        head_label=head_label,
        tail_label=tail_label,
        matrix=transr_matrix,
    )

    fig = go.Figure()

    for trace in transe_traces:
        trace.visible = True
        fig.add_trace(trace)

    for trace in transh_traces:
        trace.visible = False
        fig.add_trace(trace)

    for trace in transr_traces:
        trace.visible = False
        fig.add_trace(trace)

    n_transe = len(transe_traces)
    n_transh = len(transh_traces)
    n_transr = len(transr_traces)

    visible_transe = [True] * n_transe + [False] * n_transh + [False] * n_transr
    visible_transh = [False] * n_transe + [True] * n_transh + [False] * n_transr
    visible_transr = [False] * n_transe + [False] * n_transh + [True] * n_transr

    transe_annotations = _full_annotations(
        transe_ann,
        transe_exp,
        PROPERTY_TEXT["TransE"],
        census_panel,
    )
    transh_annotations = _full_annotations(
        transh_ann,
        transh_exp,
        PROPERTY_TEXT["TransH"],
        census_panel,
    )
    transr_annotations = _full_annotations(
        transr_ann,
        transr_exp,
        PROPERTY_TEXT["TransR"],
        census_panel,
    )

    fig.update_layout(
        title=transe_title,
        annotations=transe_annotations,
        width=width,
        height=height,
        margin=dict(l=70, r=410, t=90, b=180),
        xaxis=dict(
            range=[-5, 5],
            zeroline=True,
            title="embedding dimension 1",
        ),
        yaxis=dict(
            range=[-5, 5],
            zeroline=True,
            title="embedding dimension 2",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="TransE",
                        method="update",
                        args=[
                            {"visible": visible_transe},
                            {"title": transe_title, "annotations": transe_annotations},
                        ],
                    ),
                    dict(
                        label="TransH",
                        method="update",
                        args=[
                            {"visible": visible_transh},
                            {"title": transh_title, "annotations": transh_annotations},
                        ],
                    ),
                    dict(
                        label="TransR",
                        method="update",
                        args=[
                            {"visible": visible_transr},
                            {"title": transr_title, "annotations": transr_annotations},
                        ],
                    ),
                ],
                direction="down",
                x=0.01,
                y=1.13,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
    )

    return fig


# -----------------------------------------------------------------------------
# Convenience function for notebook tutorials
# -----------------------------------------------------------------------------


def show_default_translation_demo(renderer: str | None = None) -> go.Figure:
    """
    Create and show the default translation-model visualization.

    Parameters
    ----------
    renderer:
        Optional Plotly renderer, e.g. "notebook_connected", "iframe", "browser".

    Returns
    -------
    plotly.graph_objects.Figure
        The generated figure.
    """
    fig = plot_translation_models_clean()
    if renderer is None:
        fig.show()
    else:
        fig.show(renderer=renderer)
    return fig


if __name__ == "__main__":
    # This is useful if running the file directly in an environment that supports
    # opening Plotly figures in a browser.
    show_default_translation_demo(renderer="browser")
