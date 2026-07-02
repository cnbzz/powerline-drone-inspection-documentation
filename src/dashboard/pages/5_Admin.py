"""Admin panel — user management + activity logs (admin role only).

Gated by :func:`src.auth.session.require_admin`, so unauthenticated users see the
login form and authenticated non-admins are blocked, even via direct URL.

This page only touches the *auth* database (``outputs/db/auth.db``). It never reads
or writes the fault store.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="Admin - HHATS", page_icon="🛡️", layout="wide")

from src.dashboard.app import (  # noqa: E402
    get_current_theme,
    load_css,
    logo_img_html,
    render_sidebar,
)
from src.auth import auth_db  # noqa: E402
from src.auth.session import require_admin  # noqa: E402

load_css()
admin_user = require_admin()  # giriş + admin rol kapısı
render_sidebar()

theme = get_current_theme()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _esc(value: object) -> str:
    """HTML-escape a value for safe table rendering."""
    return html.escape("" if value is None else str(value))


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a themed HTML table string (values are pre-escaped by caller)."""
    head_cells = "".join(
        f"<th style='text-align:left; padding:10px 14px; "
        f"background:{theme['bg_primary']}; color:{theme['text_primary']}; "
        f"font-weight:600; border-bottom:2px solid {theme['border_medium']};'>"
        f"{h}</th>"
        for h in headers
    )
    body_rows = ""
    for row in rows:
        cells = "".join(
            f"<td style='padding:9px 14px; color:{theme['text_primary']}; "
            f"border-bottom:1px solid {theme['border_light']};'>{c}</td>"
            for c in row
        )
        body_rows += f"<tr>{cells}</tr>"
    return (
        f"<table style='width:100%; border-collapse:collapse; "
        f"background:{theme['bg_tertiary']}; border-radius:12px; overflow:hidden; "
        f"border:1px solid {theme['border_light']}; font-size:0.9rem;'>"
        f"<thead><tr>{head_cells}</tr></thead><tbody>{body_rows}</tbody></table>"
    )


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}; color:white; padding:3px 10px; "
        f"border-radius:6px; font-size:0.8rem; font-weight:600;'>{text}</span>"
    )


# --------------------------------------------------------------------------- #
# Header                                                                       #
# --------------------------------------------------------------------------- #
# Küçük logo başlığın üstünde (yoksa atla — fallback çökme yok).
_admin_logo = logo_img_html(100, container=True, margin="0 0 0.6rem 0", align="left")
if _admin_logo:
    st.markdown(_admin_logo, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🛡️ Admin Paneli</h1>', unsafe_allow_html=True)
st.markdown(
    f"<p class='subtitle' style='color:{theme['text_secondary']};'>"
    f"Kullanıcı yönetimi ve sistem aktivite kayıtları</p>",
    unsafe_allow_html=True,
)

tab_users, tab_new, tab_pw, tab_logs = st.tabs(
    ["👥 Kullanıcılar", "➕ Yeni Kullanıcı", "🔑 Şifre Sıfırla", "📜 Aktivite Logları"]
)

# --------------------------------------------------------------------------- #
# Tab: user list                                                              #
# --------------------------------------------------------------------------- #
with tab_users:
    st.markdown(
        f"<h3 style='color:{theme['text_primary']} !important;'>Kullanıcı Listesi</h3>",
        unsafe_allow_html=True,
    )
    users = auth_db.list_users()
    rows = []
    for u in users:
        role_badge = _badge(
            "Yönetici" if u["role"] == "admin" else "Kullanıcı",
            theme["accent_primary"] if u["role"] == "admin" else theme["text_secondary"],
        )
        active_badge = (
            _badge("Aktif", theme["success"])
            if u["is_active"]
            else _badge("Pasif", theme["danger"])
        )
        rows.append([
            _esc(u["username"]),
            role_badge,
            active_badge,
            _esc(u["last_login"] or "—"),
            _esc(u["created_at"]),
        ])
    st.markdown(
        _html_table(
            ["Kullanıcı Adı", "Rol", "Durum", "Son Giriş", "Oluşturulma"], rows
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<h4 style='color:{theme['text_primary']} !important;'>"
        f"Aktif/Pasif Durumu Değiştir</h4>",
        unsafe_allow_html=True,
    )
    names = [u["username"] for u in users]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        target = st.selectbox("Kullanıcı seç", names, key="toggle_target")
    with col_b:
        sel = next((u for u in users if u["username"] == target), None)
        currently_active = bool(sel and sel["is_active"])
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if currently_active:
            if st.button("⛔ Pasifleştir", key="deactivate_btn", width="stretch"):
                if target == admin_user["username"]:
                    st.error("Kendi hesabınızı pasifleştiremezsiniz.")
                else:
                    auth_db.set_active(target, False)
                    auth_db.log_activity(
                        admin_user["username"], "user_deactivated",
                        f"Hedef: {target}",
                    )
                    st.success(f"{target} pasifleştirildi.")
                    st.rerun()
        else:
            if st.button("✅ Aktifleştir", key="activate_btn", width="stretch"):
                auth_db.set_active(target, True)
                auth_db.log_activity(
                    admin_user["username"], "user_activated", f"Hedef: {target}"
                )
                st.success(f"{target} aktifleştirildi.")
                st.rerun()

# --------------------------------------------------------------------------- #
# Tab: new user                                                               #
# --------------------------------------------------------------------------- #
with tab_new:
    st.markdown(
        f"<h3 style='color:{theme['text_primary']} !important;'>Yeni Kullanıcı Ekle</h3>",
        unsafe_allow_html=True,
    )
    with st.form("new_user_form", clear_on_submit=True):
        nu_name = st.text_input("Kullanıcı Adı", key="nu_name")
        nu_pw = st.text_input("Şifre", type="password", key="nu_pw")
        nu_role = st.selectbox(
            "Rol", ["user", "admin"], format_func=lambda r: "Kullanıcı" if r == "user" else "Yönetici",
            key="nu_role",
        )
        nu_submit = st.form_submit_button("Kullanıcı Oluştur", width="stretch")
    if nu_submit:
        if not nu_name or not nu_pw:
            st.error("Kullanıcı adı ve şifre gerekli.")
        elif len(nu_pw) < 4:
            st.error("Şifre en az 4 karakter olmalı.")
        elif auth_db.create_user(nu_name, nu_pw, role=nu_role):
            auth_db.log_activity(
                admin_user["username"], "user_created",
                f"Yeni kullanıcı: {nu_name} (rol: {nu_role})",
            )
            st.success(f"'{nu_name}' oluşturuldu.")
            st.rerun()
        else:
            st.error("Bu kullanıcı adı zaten mevcut.")

# --------------------------------------------------------------------------- #
# Tab: password reset                                                         #
# --------------------------------------------------------------------------- #
with tab_pw:
    st.markdown(
        f"<h3 style='color:{theme['text_primary']} !important;'>"
        f"Kullanıcı Şifresi Sıfırla</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Yönetici olarak herhangi bir kullanıcının şifresini sıfırlayabilirsiniz "
               "(mevcut şifre gerekmez).")
    all_names = [u["username"] for u in auth_db.list_users()]
    with st.form("reset_pw_form", clear_on_submit=True):
        rp_target = st.selectbox("Kullanıcı", all_names, key="rp_target")
        rp_new = st.text_input("Yeni Şifre", type="password", key="rp_new")
        rp_submit = st.form_submit_button("Şifreyi Sıfırla", width="stretch")
    if rp_submit:
        if not rp_new:
            st.error("Yeni şifre gerekli.")
        elif len(rp_new) < 4:
            st.error("Şifre en az 4 karakter olmalı.")
        elif auth_db.change_password(rp_target, rp_new):
            auth_db.log_activity(
                admin_user["username"], "password_reset", f"Hedef: {rp_target}"
            )
            st.success(f"'{rp_target}' şifresi sıfırlandı.")
        else:
            st.error("Şifre sıfırlanamadı.")

# --------------------------------------------------------------------------- #
# Tab: activity logs                                                          #
# --------------------------------------------------------------------------- #
with tab_logs:
    st.markdown(
        f"<h3 style='color:{theme['text_primary']} !important;'>Aktivite Logları</h3>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        log_user = st.text_input("Kullanıcıya göre filtrele (boş = hepsi)", key="log_user")
    with col2:
        action_options = [
            "(hepsi)", "login_success", "login_failed", "logout",
            "password_change", "password_reset", "user_created",
            "user_deactivated", "user_activated",
        ]
        log_action = st.selectbox("İşleme göre filtrele", action_options, key="log_action")

    logs = auth_db.get_activity_log(
        username=log_user.strip() or None,
        action=None if log_action == "(hepsi)" else log_action,
        limit=500,
    )
    st.caption(f"{len(logs)} kayıt gösteriliyor (en yeni önce).")

    log_rows = [
        [
            _esc(r["timestamp"]),
            _esc(r["username"] or "—"),
            _esc(r["action"]),
            _esc(r["detail"] or ""),
        ]
        for r in logs
    ]
    st.markdown(
        _html_table(["Zaman", "Kullanıcı", "İşlem", "Detay"], log_rows),
        unsafe_allow_html=True,
    )
