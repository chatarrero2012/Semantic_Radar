import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Semantic Radar",
    layout="wide",
    page_icon="🧠"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 0%, #0b1020, #05070f);
    color: #e6edf3;
}
h1 {
    font-size: 3rem;
    background: linear-gradient(90deg, #4cc9f0, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
textarea {
    background-color: rgba(0,0,0,0.6) !important;
    color: #e6edf3 !important;
    border-radius: 12px !important;
}
button {
    background: linear-gradient(90deg, #4cc9f0, #a78bfa) !important;
    color: black !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Semantic Radar")
st.caption("Real-time narrative mapping")

# =========================
# INPUT UI
# =========================
col1, col2 = st.columns([2,1])

with col1:
    raw = st.text_area(
        "Input stream",
        height=220,
        placeholder="Paste your text..."
    )

with col2:
    mode = st.selectbox("Mode", ["SAFE", "EMBEDDINGS"])
    run = st.button("🚀 Analyze", use_container_width=True)

texts = [t for t in raw.split("\n") if t.strip()]

# =========================
# SAFE VECTOR
# =========================
def ascii_vector(texts):
    return np.array([
        [ord(c) for c in t[:60]] + [0]*(60-len(t[:60]))
        for t in texts
    ])

# =========================
# EMBEDDINGS
# =========================
@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def embedding_vector(texts):
    try:
        model = load_model()
        return np.array(model.encode(texts))
    except:
        st.warning("⚠ fallback to SAFE mode")
        return ascii_vector(texts)

# =========================
# PROCESS
# =========================
if run:

    if len(texts) < 3:
        st.warning("Need at least 3 messages")
    else:
        with st.spinner("Mapping semantic space..."):

            data = ascii_vector(texts) if mode == "SAFE" else embedding_vector(texts)
            coords = PCA(n_components=3).fit_transform(data)

        x, y, z = coords[:,0], coords[:,1], coords[:,2]

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(
                size=6,
                color=np.arange(len(x)),
                colorscale='Turbo'
            ),
            line=dict(width=6),
            text=texts,
            hovertemplate="""
            <b>Message:</b><br>%{text}<br>
            <b>Index:</b> %{marker.color}
            <extra></extra>
            """
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.markdown("### 📊 Insights")
        c1, c2 = st.columns(2)
        c1.metric("Messages", len(texts))
        c2.metric("Unique", len(set(texts)))
