"""FinGraph investigator dashboard.

A lightweight Streamlit front end over the analysis pipeline: browse the ranked
alert queue, select an account, and see its risk score, the evidence behind the
flag, the plain-English case note, and a visual of its transaction neighbourhood.

Run with:
    streamlit run src/fingraph/dashboard/app.py
"""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from fingraph.pipeline import Pipeline


@st.cache_resource
def load_pipeline() -> Pipeline:
    # cache_resource keeps one built pipeline alive across reruns and sessions,
    # so Streamlit's re-execution on every interaction stays fast.
    return Pipeline().build()


def neighbourhood_figure(pipeline: Pipeline, account_id: str) -> go.Figure:
    """Draw the account's local transaction graph, coloured by risk."""
    ego = pipeline.ego_graph(account_id, radius=1)
    layout = nx.spring_layout(ego, seed=42)

    edge_x, edge_y = [], []
    for u, v in ego.edges():
        x0, y0 = layout[u]
        x1, y1 = layout[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#bbb"),
        hoverinfo="none",
    )

    node_x, node_y, colours, labels = [], [], [], []
    for node in ego.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)
        score = float(pipeline.risk_scores.get(node, 0.0))
        # The focus account is drawn distinctly; neighbours shade by their risk.
        colours.append(1.5 if node == account_id else score)
        labels.append(f"{node}<br>risk {score:.2f}")

    nodes = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=labels,
        hoverinfo="text",
        marker=dict(
            size=[22 if n == account_id else 13 for n in ego.nodes()],
            color=colours,
            colorscale="Reds",
            cmin=0,
            cmax=1,
            line=dict(width=1, color="#333"),
        ),
    )

    fig = go.Figure(data=[edges, nodes])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="FinGraph", layout="wide")
    st.title("FinGraph — Transaction Intelligence")
    st.caption("Graph-based detection and explanation of suspicious financial activity.")

    pipeline = load_pipeline()
    metrics = pipeline.metrics()

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", metrics["roc_auc"])
    c2.metric("Avg precision", metrics["avg_precision"])
    c3.metric("Accounts analysed", metrics["num_accounts"])

    st.divider()
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Alert queue")
        alerts = pipeline.alerts(top_n=25)
        st.dataframe(alerts, use_container_width=True, hide_index=True)
        account_id = st.selectbox("Investigate account", alerts["account_id"])

    with right:
        st.subheader(f"Investigation — {account_id}")
        case = pipeline.investigation(account_id)
        st.markdown(f"**Risk band:** `{case.risk_band}`  ·  **Score:** `{case.risk_score:.2f}`")
        st.info(case.case_note)

        if case.reasons:
            st.markdown("**Evidence**")
            for reason in case.reasons:
                st.markdown(f"- `{reason.severity.upper()}` — {reason.detail}")

        st.markdown("**Transaction neighbourhood**")
        st.plotly_chart(neighbourhood_figure(pipeline, account_id), use_container_width=True)


if __name__ == "__main__":
    main()
