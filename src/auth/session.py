"""Streamlit session helpers — the login gate that fronts every page.

Usage in a page (inserted between ``load_css()`` and ``render_sidebar()``)::

    load_css()
    require_login()          # renders login form + st.stop() if not authenticated
    render_sidebar()

Admin-only page::

    load_css()
    require_admin()          # also blocks non-admins (even via direct URL)

The active user lives in ``st.session_state["auth_user"]`` (a dict) and survives
page navigation automatically, like Streamlit's other session state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from src.auth import auth_db

_ROOT = Path(__file__).resolve().parents[2]
LOGO_PATH = _ROOT / "assets" / "logo" / "bedas_logo.png"

# Brand palette (task spec). Literal hex so it never loses to Streamlit defaults.
BEDAS_BLUE = "#003A8C"
BEDAS_GREEN = "#009A3A"

_SESSION_KEY = "auth_user"

# Page paths for post-login redirect (relative to pages/ auto-discovery root).
_ADMIN_PAGE = "pages/5_Admin.py"
_DASHBOARD_PAGE = "pages/1_Dashboard.py"


# --------------------------------------------------------------------------- #
# Session accessors                                                            #
# --------------------------------------------------------------------------- #
def current_user() -> Optional[dict]:
    """Return the logged-in user dict, or ``None`` if not authenticated."""
    return st.session_state.get(_SESSION_KEY)


def is_logged_in() -> bool:
    return current_user() is not None


def is_admin() -> bool:
    user = current_user()
    return bool(user and user.get("role") == "admin")


def logout() -> None:
    """Clear the session and log the event."""
    user = current_user()
    if user:
        try:
            auth_db.log_activity(user["username"], "logout", "Kullanici cikis yapti")
        except Exception:
            pass
    st.session_state.pop(_SESSION_KEY, None)
    try:
        st.switch_page("app.py")
    except Exception:
        st.rerun()


# --------------------------------------------------------------------------- #
# Login form                                                                   #
# --------------------------------------------------------------------------- #
def _render_login_form() -> None:
    """Render the centered, branded login card. Sets session + redirects on
    successful authentication."""
    # Lazy import avoids a circular dependency (app.py imports this module).
    from src.dashboard.app import get_current_theme, init_theme, logo_img_html

    init_theme()
    theme = get_current_theme()

    # Center the card in the middle column.
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<div style='height:3vh;'></div>", unsafe_allow_html=True)

        # Logo on a white rounded container, otherwise a styled emoji/text fallback.
        logo_html = logo_img_html(200, container=True, margin="0 auto 0.5rem auto")
        if logo_html:
            st.markdown(logo_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style="text-align:center; font-size:3rem; line-height:1;
                            margin-bottom:0.25rem;">⚡</div>
                <div style="text-align:center; color:{BEDAS_BLUE} !important;
                            font-weight:800; font-size:1.1rem; letter-spacing:1px;
                            margin-bottom:0.5rem;">BEDAŞ</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <h1 style="text-align:center; color:{BEDAS_BLUE} !important;
                       font-weight:700; font-size:1.8rem; margin:0.25rem 0 0 0;
                       letter-spacing:-0.02em;">HHATS Giriş</h1>
            <p style="text-align:center; color:{theme['text_secondary']};
                      font-size:0.95rem; margin:0.35rem 0 1.5rem 0;">
                Havai Hat Arıza Tespit Sistemi
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Kullanıcı Adı", key="login_username")
            password = st.text_input(
                "Şifre", type="password", key="login_password"
            )
            # Green "Giriş" button per spec — scoped style for this form only.
            st.markdown(
                f"""
                <style>
                div[data-testid="stForm"] .stFormSubmitButton button {{
                    background: {BEDAS_GREEN} !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 10px !important;
                    font-weight: 700 !important;
                    width: 100% !important;
                    padding: 0.6rem 1.5rem !important;
                }}
                div[data-testid="stForm"] .stFormSubmitButton button:hover {{
                    background: {BEDAS_GREEN} !important;
                    opacity: 0.92 !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("Giriş", width="stretch")

        if submitted:
            _handle_login_submit(username, password)


def _handle_login_submit(username: str, password: str) -> None:
    """Validate credentials, set session, log, and redirect by role."""
    username = (username or "").strip()
    if not username or not password:
        st.error("Kullanıcı adı ve şifre gerekli.")
        return

    user = auth_db.verify_user(username, password)
    if user is None:
        # Do NOT log the attempted password — only the username.
        try:
            auth_db.log_activity(username, "login_failed", "Hatalı giriş denemesi")
        except Exception:
            pass
        st.error("Kullanıcı adı veya şifre hatalı (ya da hesap pasif).")
        return

    st.session_state[_SESSION_KEY] = user
    try:
        auth_db.log_activity(username, "login_success", f"Rol: {user['role']}")
    except Exception:
        pass

    # Requirement: admin -> admin panel, normal user -> dashboard.
    target = _ADMIN_PAGE if user["role"] == "admin" else _DASHBOARD_PAGE
    try:
        st.switch_page(target)
    except Exception:
        # AppTest / isolated env: switch_page is unavailable -> just rerun so the
        # now-authenticated session re-renders the current page.
        st.rerun()


# --------------------------------------------------------------------------- #
# Guards                                                                       #
# --------------------------------------------------------------------------- #
def require_login() -> dict:
    """Gate the current page behind authentication.

    If no user is logged in, render the login form and ``st.stop()`` (page content
    below this call never runs). Otherwise return the active user dict.
    """
    auth_db.init_db()
    user = current_user()
    if user is None:
        _render_login_form()
        st.stop()
    return user


def require_admin() -> dict:
    """Gate the current page behind the *admin* role.

    Blocks unauthenticated users (login form) and authenticated non-admins
    (unauthorized notice) — even if they reach the page by direct URL.
    """
    user = require_login()
    if user.get("role") != "admin":
        st.error("⛔ Bu sayfaya erişim yetkiniz yok (yalnızca yönetici).")
        st.info("Yönetici değilseniz Dashboard'a dönebilirsiniz.")
        st.stop()
    return user


# --------------------------------------------------------------------------- #
# Sidebar account section (rendered inside the existing branded sidebar)       #
# --------------------------------------------------------------------------- #
def render_account_sidebar() -> None:
    """Render the logged-in user block: identity, change-password, logout, and an
    admin-only link to the admin panel. Assumes it is called inside an open
    ``st.sidebar`` context (see :func:`src.dashboard.app.render_sidebar`)."""
    user = current_user()
    if user is None:
        return

    role_label = "Yönetici" if user.get("role") == "admin" else "Kullanıcı"
    st.markdown("---")
    st.markdown(f"### 👤 {user['username']}")
    st.caption(f"Rol: {role_label}")

    # Admin-only shortcut to the admin panel.
    if user.get("role") == "admin":
        if st.button("🛡️ Admin Paneli", key="sidebar_admin_btn", width="stretch"):
            try:
                st.switch_page(_ADMIN_PAGE)
            except Exception:
                pass

    # Self-service password change (current-password verification required).
    with st.expander("🔑 Şifre Değiştir"):
        with st.form("change_pw_sidebar_form", clear_on_submit=True):
            old_pw = st.text_input("Mevcut Şifre", type="password", key="cpw_old")
            new_pw = st.text_input("Yeni Şifre", type="password", key="cpw_new")
            new_pw2 = st.text_input(
                "Yeni Şifre (Tekrar)", type="password", key="cpw_new2"
            )
            ok = st.form_submit_button("Güncelle", width="stretch")
        if ok:
            if not old_pw or not new_pw:
                st.error("Tüm alanları doldurun.")
            elif new_pw != new_pw2:
                st.error("Yeni şifreler eşleşmiyor.")
            elif len(new_pw) < 4:
                st.error("Yeni şifre en az 4 karakter olmalı.")
            elif auth_db.change_password(
                user["username"], new_pw, old_password=old_pw
            ):
                auth_db.log_activity(
                    user["username"], "password_change", "Kendi şifresini değiştirdi"
                )
                st.success("Şifre güncellendi.")
            else:
                st.error("Mevcut şifre hatalı.")

    if st.button("🚪 Çıkış", key="sidebar_logout_btn", width="stretch"):
        logout()
