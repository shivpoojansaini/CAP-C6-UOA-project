import streamlit as st
import os
import sys

# Debug: Print to stderr (shows in logs)
print(f"[DEBUG] Python: {sys.executable}", file=sys.stderr)
print(f"[DEBUG] PORT env: {os.environ.get('PORT', 'not set')}", file=sys.stderr)
print(f"[DEBUG] Starting Streamlit app...", file=sys.stderr)

st.set_page_config(page_title="Test App", page_icon="🧪")

st.title("🧪 Test Streamlit App")
st.write("If you can see this, Streamlit is working!")

st.balloons()
