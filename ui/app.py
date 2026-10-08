"""Streamlit entry point: page config, the documents sidebar, and navigation between the pages in `ui/views/`.

Run: `streamlit run ui/app.py` (FINCHAT_API_URL defaults to http://localhost:8000).
"""

import sys
from pathlib import Path

# `streamlit run ui/app.py` only puts ui/ on sys.path; the pages import the `ui` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from ui import components  # noqa: E402

st.set_page_config(page_title="FinChat", page_icon="📄", layout="wide")
components.inject_style()

components.sidebar(components.get_client())

pages = [
    st.Page("views/ask.py", title="Ask", icon="💬", default=True),
    st.Page("views/metrics.py", title="Metrics", icon="📊"),
]
st.navigation(pages).run()
