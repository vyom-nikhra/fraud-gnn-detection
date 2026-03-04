import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config & Environment
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

st.set_page_config(
    page_title="AML Fraud Ring Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Import Google Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ──────────────────────────────────────────────────── */
:root {
    --bg-primary: #0a0e17;
    --bg-card: rgba(15, 23, 42, 0.65);
    --bg-card-hover: rgba(15, 23, 42, 0.85);
    --border-card: rgba(99, 102, 241, 0.15);
    --border-glow: rgba(99, 102, 241, 0.4);
    --accent-indigo: #818cf8;
    --accent-cyan: #22d3ee;
    --accent-red: #f87171;
    --accent-green: #34d399;
    --accent-amber: #fbbf24;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gradient-hero: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
    --gradient-safe: linear-gradient(135deg, #059669 0%, #34d399 100%);
}

/* ── Global resets ───────────────────────────────────────────────────── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar {width: 6px;}
::-webkit-scrollbar-track {background: var(--bg-primary);}
::-webkit-scrollbar-thumb {background: var(--text-muted); border-radius: 3px;}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-card) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
}

/* ── Text input ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stTextInput > div > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    transition: border 0.3s ease, box-shadow 0.3s ease;
}
section[data-testid="stSidebar"] .stTextInput > div > div:focus-within {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}
section[data-testid="stSidebar"] .stTextInput input {
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}

/* ── Primary button ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button {
    background: var(--gradient-hero) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.025em !important;
    width: 100% !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
}
section[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Glass card helper ───────────────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 1.6rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.08);
}

/* ── Metric cards ────────────────────────────────────────────────────── */
.metric-card {
    text-align: center;
    padding: 1.8rem 1.2rem;
}
.metric-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.35rem;
    background: var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-value.danger {
    background: var(--gradient-danger);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-value.safe {
    background: var(--gradient-safe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
}

/* ── Section header ──────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-card);
}
.section-header .icon {
    font-size: 1.35rem;
}
.section-header .title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
}
.section-header .badge {
    margin-left: auto;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
}
.badge-danger {
    background: rgba(239, 68, 68, 0.15);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.badge-safe {
    background: rgba(52, 211, 153, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.badge-info {
    background: rgba(34, 211, 238, 0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(34, 211, 238, 0.3);
}

/* ── SAR report body ─────────────────────────────────────────────────── */
.sar-body {
    color: var(--text-secondary);
    font-size: 0.92rem;
    line-height: 1.75;
    white-space: pre-wrap;
}

/* ── Hero banner ─────────────────────────────────────────────────────── */
.hero {
    position: relative;
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    margin-bottom: 1.5rem;
    overflow: hidden;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid var(--border-card);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero p {
    color: var(--text-secondary);
    font-size: 1rem;
    max-width: 600px;
    margin: 0 auto;
    font-weight: 400;
}

/* ── Status pill in sidebar ──────────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-online {
    background: rgba(52, 211, 153, 0.12);
    color: var(--accent-green);
    border: 1px solid rgba(52, 211, 153, 0.25);
}
.dot-pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* ── Waiting state ───────────────────────────────────────────────────── */
.waiting-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
}
.waiting-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.waiting-state .msg { font-size: 1rem; font-weight: 500; }
.waiting-state .sub { font-size: 0.85rem; margin-top: 0.4rem; }

/* ── Error box ───────────────────────────────────────────────────────── */
.error-box {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: var(--accent-red);
    font-size: 0.9rem;
}

/* ── Plotly chart container ──────────────────────────────────────────── */
.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Override markdown colors in main area ───────────────────────────── */
.stApp .stMarkdown h1, .stApp .stMarkdown h2, .stApp .stMarkdown h3 {
    color: var(--text-primary) !important;
}
.stApp .stMarkdown p, .stApp .stMarkdown li {
    color: var(--text-secondary) !important;
}

/* ── Separator ───────────────────────────────────────────────────────── */
.sep { height: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.8rem;">
        <div style="font-size:2.2rem;">🛡️</div>
        <div style="font-size:1.15rem; font-weight:700; color:#f1f5f9; margin-top:0.3rem;">
            Fraud Ring Detector
        </div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:0.15rem;">
            GraphSAGE + Gemini AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.75rem; font-weight:600; text-transform:uppercase;
                letter-spacing:0.1em; color:#94a3b8; margin-bottom:0.5rem;">
        🔍 &nbsp;Investigation Parameters
    </div>
    """, unsafe_allow_html=True)

    tx_id_input = st.text_input("Transaction ID", value="232438397", label_visibility="collapsed",
                                 placeholder="Enter Transaction ID…")
    analyze_btn = st.button("⚡  Analyze Transaction", type="primary", use_container_width=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # Status indicator
    st.markdown("""
    <div style="font-size:0.75rem; font-weight:600; text-transform:uppercase;
                letter-spacing:0.1em; color:#94a3b8; margin-bottom:0.6rem;">
        System Status
    </div>
    <div class="status-pill status-online">
        <span class="dot-pulse"></span> API Online
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#475569; line-height:1.6;">
        <strong style="color:#94a3b8;">How it works</strong><br>
        1. Fetches 2-hop subgraph from Neo4j<br>
        2. Runs GraphSAGE inference on subgraph<br>
        3. Generates SAR narrative via Gemini AI<br>
        4. Renders interactive network topology
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🚨 Enterprise AML Fraud Ring Detection</h1>
    <p>Analyze Bitcoin transaction networks using Graph Neural Networks and Generative AI
    to surface suspicious activity in real-time.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Fetch graph data from Neo4j
# ---------------------------------------------------------------------------
def fetch_graph_data(tx_id):
    """Fetches edges directly from Neo4j for visualization."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")

    query = """
    MATCH (target:Transaction {txId: $tx_id})
    OPTIONAL MATCH (target)-[*1..2]-(m:Transaction)
    WITH target, collect(DISTINCT m) AS neighbors
    WITH [target] + neighbors AS nodes
    UNWIND nodes AS n1
    MATCH (n1)-[:PAYS]->(n2:Transaction)
    WHERE n2 IN nodes
    RETURN DISTINCT n1.txId AS source, n2.txId AS target
    """
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as session:
        result = session.run(query, tx_id=int(tx_id))
        edges = [(record["source"], record["target"]) for record in result]
    driver.close()
    return edges


# ---------------------------------------------------------------------------
# Helper: Build interactive Plotly network graph
# ---------------------------------------------------------------------------
def build_plotly_graph(edges, target_tx_id):
    """Creates a stunning interactive Plotly network visualization."""
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # ── Edge traces ──
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='rgba(99,102,241,0.25)'),
        hoverinfo='none',
        mode='lines',
    )

    # ── Node traces ──
    node_x, node_y, node_color, node_size, node_text, node_border = [], [], [], [], [], []
    target_id = int(target_tx_id)

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)

        if node == target_id:
            node_color.append('#f87171')
            node_size.append(28)
            node_text.append(f"<b>TX {node}</b><br>Target Transaction<br>Connections: {degree}")
            node_border.append('rgba(239,68,68,0.6)')
        else:
            # Size by degree
            sz = max(7, min(18, 5 + degree * 2))
            node_color.append('#818cf8')
            node_size.append(sz)
            node_text.append(f"TX {node}<br>Connections: {degree}")
            node_border.append('rgba(99,102,241,0.3)')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1.5, color=node_border),
            opacity=0.92,
        ),
    )

    # ── Layout ──
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            plot_bgcolor='rgba(10,14,23,0)',
            paper_bgcolor='rgba(10,14,23,0)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=480,
            font=dict(family='Inter, sans-serif', color='#94a3b8'),
            dragmode='pan',
        ),
    )
    fig.update_layout(
        modebar=dict(bgcolor='rgba(0,0,0,0)', color='#475569', activecolor='#818cf8'),
    )
    return fig, len(G.nodes()), len(G.edges())


# ---------------------------------------------------------------------------
# Helper: Render a metric card
# ---------------------------------------------------------------------------
def metric_card(icon, value, label, style_class=""):
    return f"""
    <div class="glass-card metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value {style_class}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------
if analyze_btn:
    # Progress bar animation
    progress = st.progress(0, text="Initializing investigation…")
    for i in range(25):
        time.sleep(0.012)
        progress.progress(i + 1, text="Connecting to backend…")

    try:
        # 1. Call the FastAPI backend
        response = requests.get(f"http://127.0.0.1:8000/predict/{tx_id_input}")

        for i in range(25, 60):
            time.sleep(0.01)
            progress.progress(i + 1, text="Running GraphSAGE inference…")

        if response.status_code == 200:
            data = response.json()

            for i in range(60, 80):
                time.sleep(0.01)
                progress.progress(i + 1, text="Generating SAR with Gemini…")

            is_illicit = data['prediction'] == "Illicit"
            pred_style = "danger" if is_illicit else "safe"
            pred_icon = "🚫" if is_illicit else "✅"
            conf_icon = "🎯"
            nodes_icon = "🔗"

            # ── Metric Cards ──
            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                st.markdown(metric_card(pred_icon, data['prediction'], "AI Prediction", pred_style),
                            unsafe_allow_html=True)
            with col2:
                st.markdown(metric_card(conf_icon, f"{data['confidence']}%", "Model Confidence"),
                            unsafe_allow_html=True)
            with col3:
                st.markdown(metric_card(nodes_icon, data['nodes_in_subgraph'], "Nodes in Subgraph"),
                            unsafe_allow_html=True)

            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

            # ── SAR Report ──
            badge_cls = "badge-danger" if is_illicit else "badge-safe"
            badge_txt = "HIGH RISK" if is_illicit else "LOW RISK"

            st.markdown(f"""
            <div class="glass-card">
                <div class="section-header">
                    <span class="icon">📝</span>
                    <span class="title">Suspicious Activity Report</span>
                    <span class="badge {badge_cls}">{badge_txt}</span>
                </div>
                <div class="sar-body">{data['suspicious_activity_report']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

            # ── Network Graph ──
            for i in range(80, 95):
                time.sleep(0.01)
                progress.progress(i + 1, text="Fetching network topology from Neo4j…")

            edges = fetch_graph_data(tx_id_input)

            progress.progress(100, text="Investigation complete ✓")
            time.sleep(0.4)
            progress.empty()

            if edges:
                fig, n_nodes, n_edges = build_plotly_graph(edges, tx_id_input)

                st.markdown(f"""
                <div class="glass-card" style="padding-bottom:0.4rem;">
                    <div class="section-header">
                        <span class="icon">🕸️</span>
                        <span class="title">Network Topology — 2-Hop Neighborhood</span>
                        <span class="badge badge-info">{n_nodes} nodes · {n_edges} edges</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                })
            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding:2.5rem;">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">🔍</div>
                    <div style="color:#94a3b8; font-weight:500;">No network connections found for this transaction.</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            progress.empty()
            detail = response.json().get('detail', 'Unknown error')
            st.markdown(f"""
            <div class="error-box">⚠️ &nbsp;<strong>API Error</strong>: {detail}</div>
            """, unsafe_allow_html=True)

    except Exception as e:
        progress.empty()
        st.markdown(f"""
        <div class="error-box">
            ⚠️ &nbsp;<strong>Connection Failed</strong><br>
            Could not reach the backend API. Please ensure the FastAPI server is running.<br>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; opacity:0.7;">{e}</span>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Waiting / Empty State ──
    st.markdown("""
    <div class="glass-card waiting-state">
        <div class="icon">🔎</div>
        <div class="msg">Enter a Transaction ID and click <strong>Analyze</strong> to begin investigation</div>
        <div class="sub">The system will query the graph database, run GNN inference, and generate an AI-powered report.</div>
    </div>
    """, unsafe_allow_html=True)