"""Banco Pine Intelligence - Streamlit entry point (layout, auth gate and routing only)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import streamlit as st

import page_admin as admin
import page_clients as clients
import page_credit as credit
import page_ledger as ledger
import page_macro as macro
import page_market as market
import page_treasury as treasury
import pine_auth as auth
from pine_config import get_settings
from pine_datasets import engine
from ui_components import sidebar_controls
from ui_theme import APP_CSS, register_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="Banco Pine Intelligence", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(APP_CSS, unsafe_allow_html=True)
register_template()
settings = get_settings()


def _session_valid() -> bool:
    user = st.session_state.get("user")
    started = st.session_state.get("login_at")
    if not user or not started:
        return False
    if datetime.now(UTC) - started > timedelta(hours=settings.session_hours):
        st.session_state.clear()
        return False
    return True


def _login_page() -> None:
    try:
        created = auth.ensure_bootstrap_admin(engine(), settings.admin_username,
                                              settings.admin_password, settings.admin_name)
        if created:
            st.toast("Administrator account created from secrets.")
    except auth.AuthError as exc:
        st.error(f"Bootstrap admin not created: {exc}")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.title("Banco Pine Intelligence")
        st.caption("Predictive analytics for market, credit, clients, treasury and capital.")
        with st.form("login"):
            username = st.text_input("Username", max_chars=64)
            password = st.text_input("Password", type="password", max_chars=128)
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            try:
                user = auth.authenticate(engine(), username, password,
                                         settings.max_failed_logins, settings.lockout_minutes)
                st.session_state["user"] = user
                st.session_state["login_at"] = datetime.now(UTC)
                st.rerun()
            except auth.AuthError as exc:
                st.error(str(exc))


def main() -> None:
    """Route to login or to the app pages."""
    if not _session_valid():
        _login_page()
        return
    user = st.session_state["user"]
    pages = {
        "Analysis": [
            st.Page(market.render, title="Market and valuation", url_path="market", default=True),
            st.Page(macro.render, title="Macro drivers", url_path="macro"),
            st.Page(credit.render, title="Credit portfolio", url_path="credit"),
            st.Page(clients.render, title="Clients and origination", url_path="clients"),
            st.Page(treasury.render, title="Treasury and capital", url_path="treasury"),
        ],
        "Governance": [st.Page(ledger.render, title="Prediction ledger", url_path="ledger")],
    }
    if user.role == "admin":
        pages["Governance"].append(st.Page(admin.render, title="Users and access", url_path="admin"))
    nav = st.navigation(pages)
    with st.sidebar:
        st.markdown(f"**{user.full_name}**  \n{user.role.title()}")
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
        st.divider()
    sidebar_controls()
    nav.run()


main()
