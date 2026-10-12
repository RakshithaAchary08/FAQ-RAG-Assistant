"""Streamlit multipage navigation entry point."""

import os

import streamlit as st
from dotenv import load_dotenv

from chat_helpers import is_admin_page_enabled

load_dotenv()

pages = [
    st.Page("app.py", title="Chat", icon="💬"),
]
auth_enabled = os.getenv("AUTH_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
authenticated = st.session_state.get("auth_mode") == "user"
if (
    is_admin_page_enabled(os.getenv("ADMIN_PASSWORD"))
    and (not auth_enabled or authenticated)
):
    pages.append(st.Page("pages/Admin.py", title="Admin", icon="🔒"))

navigation = st.navigation(pages)
navigation.run()
