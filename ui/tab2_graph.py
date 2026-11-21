"""Tab 2 — Interactive spherical knowledge graph canvas (3D) with live node/relation filtering."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TYPE_COLORS = {"Authority": "#FF7A3D", "Statute": "#63B3ED", "Section": "#68D391", "Rule": "#68D391",
               "Obligation": "#B794F4", "Penalty": "#FC8181", "Contract": "#F6E05E", "Clause": "#F6E05E",
               "Threshold": "#F6AD55", "Concept": "#A0AEC0", "Entity": "#A39B94"}


def fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    pts = []
    phi = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y = 1 - (i / max(n - 1, 1)) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        t = phi * i
        pts.append((math.cos(t) * r, y, math.sin(t) * r))
    return pts


def _layout(g) -> dict[str, tuple[float, float, float]]:
    """Order nodes by type then degree so related nodes land near each other on the sphere."""
    order = sorted(g.nodes, key=lambda n: (g.nodes[n].get("type", ""), -g.degree(n)))
    return dict(zip(order, fibonacci_sphere(len(order))))


def figure(kb, active_nodes: set[str], active_rels: set[str], focus: str | None, dark: bool) -> go.Figure:
    g = kb.graph
    pos = _layout(g)
    fg = "#F5F2EF" if dark else "#1F1F1F"
    fig = go.Figure()
    ex, ey, ez, hx, hy, hz = [], [], [], [], [], []
    for u, v, d in g.edges(data=True):
        if u in active_nodes and v in active_nodes and d["relation"] in active_rels:
            hl = focus in (u, v)
            tx, ty, tz = (hx, hy, hz) if hl else (ex, ey, ez)
            tx += [pos[u][0], pos[v][0], None]; ty += [pos[u][1], pos[v][1], None]; tz += [pos[u][2], pos[v][2], None]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", hoverinfo="skip", name="relations",
                               line=dict(color="rgba(160,150,140,0.35)", width=2)))
    fig.add_trace(go.Scatter3d(x=hx, y=hy, z=hz, mode="lines", hoverinfo="skip", name="focus relations",
                               line=dict(color="#FF7A3D", width=6)))
    names = list(g.nodes)
    act = [n in active_nodes for n in names]
    fig.add_trace(go.Scatter3d(
        x=[pos[n][0] for n in names], y=[pos[n][1] for n in names], z=[pos[n][2] for n in names],
        mode="markers+text", name="entities", text=[n if (a and g.degree(n) >= 3) or n == focus else "" for n, a in zip(names, act)],
        textfont=dict(size=10, color=fg), textposition="top center",
        customdata=[[g.nodes[n].get("type", "Entity"), g.degree(n)] for n in names],
        hovertemplate="<b>%{hovertext}</b><br>type: %{customdata[0]}<br>degree: %{customdata[1]}<extra></extra>",
        hovertext=names,
        marker=dict(size=[(7 + min(g.degree(n), 10) * 1.6) if a else 4 for n, a in zip(names, act)],
                    color=[TYPE_COLORS.get(g.nodes[n].get("type", "Entity"), "#A39B94") if a else "rgba(120,120,120,0.35)"
                           for n, a in zip(names, act)],
                    line=dict(width=0))))
    if focus in pos:
        fx, fy, fz = pos[focus]
        fig.add_trace(go.Scatter3d(x=[fx], y=[fy], z=[fz], mode="markers", hoverinfo="skip", name="focus",
                                   marker=dict(size=20, color="rgba(255,122,61,0.35)", line=dict(width=0))))
    ax = dict(visible=False)
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", bgcolor="rgba(0,0,0,0)"))
    return fig


def render() -> None:
    s = st.session_state
    kb = s.kb
    if kb is None or kb.graph.number_of_nodes() == 0:
        st.info("Index a corpus in the first tab to build the knowledge graph.", icon=":material/info:")
        return
    g = kb.graph
    all_nodes = sorted(g.nodes)
    all_rels = sorted({d["relation"] for _, _, d in g.edges(data=True)})
    active_nodes = set(all_nodes) if s.active_nodes is None else set(s.active_nodes) & set(all_nodes)
    active_rels = set(all_rels) if s.active_rels is None else set(s.active_rels) & set(all_rels)

    top = st.container(horizontal=True, vertical_alignment="center")
    top.caption(f"{len(active_nodes)}/{len(all_nodes)} entities and {len(active_rels)}/{len(all_rels)} relation types "
                "are active. Inactive items are excluded from Graph RAG retrieval.")
    if top.button("Reset filters", icon=":material/restart_alt:", key="reset_graph_filters"):
        s.active_nodes = None; s.active_rels = None
        s.pop("node_editor", None)
        st.rerun()

    canvas, panel = st.columns([3, 2], gap="medium")
    with panel:
        focus = st.selectbox("Inspect entity", [""] + all_nodes, key="graph_focus",
                             format_func=lambda x: x or "Select an entity…")
        if focus:
            with st.container(border=True):
                n = g.nodes[focus]
                st.markdown(f"**{focus}** · :orange-badge[{n.get('type', 'Entity')}]")
                if n.get("aliases"):
                    st.caption("Aliases resolved: " + ", ".join(sorted(n["aliases"])))
                rows = [{"direction": "out", "relation": d["relation"], "entity": v} for _, v, d in g.out_edges(focus, data=True)] + \
                       [{"direction": "in", "relation": d["relation"], "entity": u} for u, _, d in g.in_edges(focus, data=True)]
                st.dataframe(pd.DataFrame(rows), hide_index=True, height=180, alt=f"Relations of {focus}")
        st.markdown("**Relation types**")
        rels = st.pills("Relation types", all_rels, selection_mode="multi", default=sorted(active_rels),
                        key="rel_pills", label_visibility="collapsed")
        s.active_rels = list(rels) if len(rels) != len(all_rels) else None
        active_rels = set(rels)
        st.markdown("**Entities**")
        df = pd.DataFrame({"Active": [n in active_nodes for n in all_nodes], "Entity": all_nodes,
                           "Type": [g.nodes[n].get("type", "Entity") for n in all_nodes],
                           "Degree": [g.degree(n) for n in all_nodes]})
        edited = st.data_editor(df, hide_index=True, height=300, key="node_editor", disabled=["Entity", "Type", "Degree"],
                                column_config={"Active": st.column_config.CheckboxColumn(width="small")},
                                alt="Entity list with active checkboxes")
        chosen = set(edited.loc[edited["Active"], "Entity"])
        s.active_nodes = list(chosen) if len(chosen) != len(all_nodes) else None
        active_nodes = chosen
    with canvas:
        dark = (st.context.theme.type == "dark") if getattr(st.context, "theme", None) else True
        st.plotly_chart(figure(kb, active_nodes, active_rels, focus or None, dark), key="graph3d",
                        config={"displaylogo": False}, theme=None, width="stretch")
        legend = st.container(horizontal=True)
        for t, c in [("Authority", "orange"), ("Statute", "blue"), ("Section / Rule", "green"), ("Obligation", "violet"),
                     ("Penalty", "red"), ("Contract / Clause", "yellow"), ("Other", "gray")]:
            legend.badge(t, color=c)
