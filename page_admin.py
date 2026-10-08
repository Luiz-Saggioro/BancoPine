"""Page: User administration (admins only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import pine_auth as auth
from pine_config import ROLES
from pine_datasets import engine
from ui_components import current_user


def render() -> None:
    """Render the page."""
    if current_user().role != "admin":
        st.error("Administrator access required.")
        return
    st.title("Users and access")
    users = pd.DataFrame(auth.list_users(engine()))
    st.dataframe(users, hide_index=True)

    left, right = st.columns(2)
    with left, st.form("new_user", clear_on_submit=True):
        st.markdown("**Create user**")
        u = st.text_input("Username")
        n = st.text_input("Full name")
        r = st.selectbox("Role", ROLES, index=2,
                         help="admin: manage users. analyst: save scenarios. viewer: read only.")
        p = st.text_input("Temporary password", type="password")
        if st.form_submit_button("Create"):
            try:
                auth.create_user(engine(), u, n, r, p)
                st.success("User created.")
                st.rerun()
            except auth.AuthError as exc:
                st.error(str(exc))
    with right, st.form("manage_user"):
        st.markdown("**Manage user**")
        names = users["username"].tolist() if not users.empty else []
        who = st.selectbox("User", names)
        active = st.radio("Status", ["Active", "Disabled"], horizontal=True)
        newp = st.text_input("New password (optional)", type="password")
        if st.form_submit_button("Apply"):
            try:
                if who == current_user().username and active == "Disabled":
                    raise auth.AuthError("You cannot disable your own account.")
                auth.set_active(engine(), who, active == "Active")
                if newp:
                    auth.reset_password(engine(), who, newp)
                st.success("Updated.")
                st.rerun()
            except auth.AuthError as exc:
                st.error(str(exc))
