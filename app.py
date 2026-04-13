"""
frontend/app.py
Streamlit multi-page application:
  - Login / Register (with consent checkbox)
  - Dashboard (real-time sensor graphs via WebSocket)
  - AI Chat (per-user session history)
  - Profile (update, change password, delete account)
  - Admin Panel (user management — admin only)

Run: streamlit run frontend/app.py
"""

import asyncio
import json
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

import httpx
import plotly.graph_objects as go
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────
import os
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
WS_URL  = os.getenv("WS_URL", "ws://localhost:8000/ws/dashboard")
MAX_POINTS = 60   # Rolling window for charts

st.set_page_config(
    page_title="AI QC Insights",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────
for key, default in {
    "token": None,
    "user": None,
    "page": "login",
    "pending_verify_email": None,   # email waiting for OTP verification
    "otp_demo_code": None,          # static OTP shown in demo mode
    "chat_sessions": [],
    "active_session_id": None,
    "messages": [],
    "ws_data": deque(maxlen=MAX_POINTS),
    "ws_connected": False,
    "ws_thread_started": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── HTTP helpers ───────────────────────────────────────────────────

def api(method: str, path: str, *, json_body=None, auth=True) -> dict | None:
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        r = httpx.request(method, f"{BACKEND}{path}", json=json_body, headers=headers, timeout=20)
        if r.status_code in (200, 201):
            return r.json()
        st.error(f"API error {r.status_code}: {r.json().get('detail', r.text)}")
    except httpx.RequestError as e:
        st.error(f"Connection error: {e}")
    return None


# ── WebSocket listener (background thread) ─────────────────────────

def _ws_listener():
    """Connects to FastAPI WebSocket and appends events to session state."""
    import websocket  # websocket-client
    def on_message(ws, msg):
        try:
            data = json.loads(msg)
            if data.get("type") == "sensor_update":
                st.session_state.ws_data.append(data)
                st.session_state.ws_connected = True
        except Exception:
            pass

    def on_error(ws, err):
        st.session_state.ws_connected = False

    def on_close(ws, *_):
        st.session_state.ws_connected = False

    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever()
        except Exception:
            pass
        time.sleep(5)   # Reconnect after 5 s


def ensure_ws():
    if not st.session_state.ws_thread_started:
        t = threading.Thread(target=_ws_listener, daemon=True)
        t.start()
        st.session_state.ws_thread_started = True


# ── Sidebar navigation ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/factory.png", width=60)
        st.markdown("### AI QC Insights")

        if st.session_state.user:
            u = st.session_state.user
            st.markdown(f"**{u['username']}** `{u['role']}`")
            st.divider()

            pages = {
                "🏠 Dashboard": "dashboard",
                "💬 AI Chat": "chat",
                "👤 Profile": "profile",
            }
            if u["role"] == "admin":
                pages["🛡 Admin Panel"] = "admin"

            for label, page_id in pages.items():
                if st.button(label, use_container_width=True):
                    st.session_state.page = page_id
                    st.rerun()

            st.divider()
            if st.button("🚪 Logout", use_container_width=True):
                for k in ("token", "user", "chat_sessions", "active_session_id", "messages"):
                    st.session_state[k] = None if k in ("token", "user", "active_session_id") else []
                st.session_state.page = "login"
                st.rerun()
        else:
            if st.button("Login", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()
            if st.button("Register", use_container_width=True):
                st.session_state.page = "register"
                st.rerun()


# ══════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════

# ── OTP Verification ───────────────────────────────────────────────

def page_verify_otp():
    st.title("✉️ Verify Your Email")

    email = st.session_state.get("pending_verify_email", "")
    demo_otp = st.session_state.get("otp_demo_code")

    if not email:
        st.warning("No pending verification. Please register first.")
        if st.button("Go to Register"):
            st.session_state.page = "register"
            st.rerun()
        return

    # ── Demo OTP banner ────────────────────────────────────────────
    st.info(
        f"A 6-digit OTP has been generated for **{email}**.\n\n"
        "In a production system this would be sent to your inbox. "
        "For this demo, the OTP is shown below."
    )

    if demo_otp:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                margin: 16px 0;
            ">
                <p style="color: #e0d7ff; font-size: 14px; margin: 0 0 8px 0;">
                    🔐 Your Demo OTP Code
                </p>
                <p style="
                    color: white;
                    font-size: 42px;
                    font-weight: 700;
                    letter-spacing: 12px;
                    font-family: monospace;
                    margin: 0;
                ">{demo_otp}</p>
                <p style="color: #c4b5fd; font-size: 12px; margin: 8px 0 0 0;">
                    Valid for 10 minutes
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(f"**Verifying:** `{email}`")

    col, _ = st.columns([1, 1])
    with col:
        with st.form("otp_form"):
            otp_input = st.text_input(
                "Enter 6-digit OTP",
                placeholder="123456",
                max_chars=6,
            )
            submitted = st.form_submit_button("✅ Verify OTP", use_container_width=True)

        if submitted:
            if len(otp_input) != 6 or not otp_input.isdigit():
                st.error("Please enter a valid 6-digit OTP.")
            else:
                result = api(
                    "POST", "/otp/verify",
                    json_body={"email": email, "otp_code": otp_input},
                    auth=False,
                )
                if result:
                    st.success("✅ Email verified successfully! You can now log in.")
                    st.session_state.pending_verify_email = None
                    st.session_state.otp_demo_code = None
                    time.sleep(1.5)
                    st.session_state.page = "login"
                    st.rerun()

        st.divider()
        st.markdown("Didn't receive the OTP?")
        resend_col, back_col = st.columns(2)

        with resend_col:
            if st.button("🔄 Resend OTP", use_container_width=True):
                result = api(
                    "POST", "/otp/resend",
                    json_body={"email": email},
                    auth=False,
                )
                if result:
                    st.session_state.otp_demo_code = result.get("otp_code")
                    st.success("New OTP generated!")
                    st.rerun()

        with back_col:
            if st.button("← Back to Register", use_container_width=True):
                st.session_state.pending_verify_email = None
                st.session_state.otp_demo_code = None
                st.session_state.page = "register"
                st.rerun()


# ── Login ──────────────────────────────────────────────────────────

def page_login():
    st.title("🔐 Login")
    col, _ = st.columns([1, 1])
    with col:
        with st.form("login_form"):
            email    = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            result = api("POST", "/auth/login", json_body={"email": email, "password": password}, auth=False)
            if result:
                st.session_state.token = result["access_token"]
                user = api("GET", "/users/me")
                if user:
                    st.session_state.user = user
                    st.session_state.page = "dashboard"
                    st.rerun()
            else:
                # Check if the failure is due to unverified email and redirect
                try:
                    r = httpx.post(
                        f"{BACKEND}/auth/login",
                        json={"email": email, "password": password},
                        timeout=10,
                    )
                    if r.status_code == 403 and "not verified" in r.text.lower():
                        st.warning("Your email is not verified yet.")
                        otp_res = api("POST", "/otp/send", json_body={"email": email}, auth=False)
                        st.session_state.pending_verify_email = email
                        st.session_state.otp_demo_code = otp_res.get("otp_code") if otp_res else None
                        if st.button("Verify Email Now →"):
                            st.session_state.page = "verify_otp"
                            st.rerun()
                except Exception:
                    pass

        if st.button("Don't have an account? Register"):
            st.session_state.page = "register"
            st.rerun()


# ── Register ───────────────────────────────────────────────────────

def page_register():
    st.title("📝 Create Account")
    col, _ = st.columns([1, 1])
    with col:
        with st.form("register_form"):
            email     = st.text_input("Email *")
            username  = st.text_input("Username *")
            full_name = st.text_input("Full Name")
            password  = st.text_input("Password * (min 8 chars)", type="password")
            password2 = st.text_input("Confirm Password", type="password")

            st.divider()
            st.markdown("##### Terms & Privacy")
            st.markdown(
                "By registering you agree to our [Terms of Service](#) and "
                "[Privacy Policy](#). Your data will be processed to provide "
                "AI quality control insights."
            )
            consent = st.checkbox(
                "I consent to the processing of my data as described above. *",
                value=False,
            )
            submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            if password != password2:
                st.error("Passwords do not match.")
            elif not consent:
                st.error("You must accept the terms and privacy policy to register.")
            elif not all([email, username, password]):
                st.error("Please fill in all required fields.")
            else:
                result = api(
                    "POST", "/auth/register",
                    json_body={
                        "email": email,
                        "username": username,
                        "full_name": full_name,
                        "password": password,
                        "consent_given": consent,
                    },
                    auth=False,
                )
                if result:
                    # Fetch the auto-generated OTP for demo display
                    otp_result = api(
                        "POST", "/otp/send",
                        json_body={"email": email},
                        auth=False,
                    )
                    st.session_state.pending_verify_email = email
                    st.session_state.otp_demo_code = otp_result.get("otp_code") if otp_result else None
                    st.session_state.page = "verify_otp"
                    st.rerun()

        if st.button("Already have an account? Login"):
            st.session_state.page = "login"
            st.rerun()


# ── Dashboard ──────────────────────────────────────────────────────

def page_dashboard():
    ensure_ws()
    st.title("🏭 Real-Time Quality Control Dashboard")

    # Connection status
    status_col, refresh_col = st.columns([3, 1])
    with status_col:
        if st.session_state.ws_connected:
            st.success("🟢 Live data connected")
        else:
            st.warning("🟡 Connecting to data stream…")
    with refresh_col:
        if st.button("⟳ Refresh", use_container_width=True):
            st.rerun()

    data = list(st.session_state.ws_data)

    if not data:
        st.info("Waiting for sensor data from the simulator…")
        st.markdown(
            "Start the simulator: `python -m simulator.data_simulator --all-machines`"
        )
        # Show empty placeholder charts
        data = []

    # Machine filter
    machines = sorted({d["machine_id"] for d in data}) if data else []
    selected = st.selectbox("Machine", ["All"] + machines)
    if selected != "All":
        data = [d for d in data if d["machine_id"] == selected]

    if not data:
        return

    # ── Metric cards ──────────────────────────────────────────────
    latest = data[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🌡 Temperature",   f"{latest['temperature']:.1f} °C")
    m2.metric("📊 Pressure",      f"{latest['pressure']:.1f} kPa")
    m3.metric("📳 Vibration",     f"{latest['vibration']:.2f} mm/s")
    m4.metric("🔴 Defect Rate",   f"{latest['defect_rate']*100:.2f}%")
    m5.metric("⚡ Speed",         f"{latest['production_speed']:.1f} u/min")

    # Anomaly alert
    if latest.get("is_anomaly"):
        st.error(
            f"⚠️ **Anomaly detected** | Severity: `{latest.get('severity','?').upper()}`\n\n"
            f"**Root Cause:** {latest.get('root_cause', 'Analyzing…')}\n\n"
            f"**Recommendation:** {latest.get('recommendation', '')}"
        )

    # ── Time-series charts ────────────────────────────────────────
    timestamps = [d["timestamp"] for d in data]
    st.divider()
    col_a, col_b = st.columns(2)

    def line_chart(col, title, y_values, y_label, anomaly_flags=None):
        fig = go.Figure()
        colors = [
            "red" if (anomaly_flags and anomaly_flags[i]) else "#1f77b4"
            for i in range(len(y_values))
        ]
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_values,
            mode="lines+markers",
            marker=dict(color=colors, size=5),
            line=dict(color="#1f77b4", width=2),
            name=y_label,
        ))
        fig.update_layout(
            title=title, xaxis_title="Time", yaxis_title=y_label,
            height=260, margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        col.plotly_chart(fig, use_container_width=True)

    anomalies = [d.get("is_anomaly", False) for d in data]
    line_chart(col_a, "Temperature (°C)", [d["temperature"] for d in data], "°C", anomalies)
    line_chart(col_b, "Pressure (kPa)",   [d["pressure"] for d in data], "kPa", anomalies)
    line_chart(col_a, "Vibration (mm/s)", [d["vibration"] for d in data], "mm/s", anomalies)
    line_chart(col_b, "Defect Rate (%)",  [d["defect_rate"]*100 for d in data], "%", anomalies)

    # Production speed
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=timestamps, y=[d["production_speed"] for d in data],
        fill="tozeroy", mode="lines",
        line=dict(color="#2ca02c", width=2),
    ))
    fig_speed.update_layout(
        title="Production Speed (u/min)", height=200,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_speed, use_container_width=True)

    # Auto-refresh every 3 seconds
    time.sleep(3)
    st.rerun()


# ── AI Chat ────────────────────────────────────────────────────────

def page_chat():
    st.title("💬 AI Quality Control Assistant")

    # Load sessions on first visit
    if not st.session_state.chat_sessions:
        sessions = api("GET", "/chat/sessions") or []
        st.session_state.chat_sessions = sessions

    col_left, col_right = st.columns([1, 3])

    # ── Session sidebar ────────────────────────────────────────────
    with col_left:
        st.markdown("**Conversations**")
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.active_session_id = None
            st.session_state.messages = []
            st.rerun()

        for sess in st.session_state.chat_sessions:
            label = f"💬 {sess['title'][:30]}"
            if st.button(label, key=sess["id"], use_container_width=True):
                st.session_state.active_session_id = sess["id"]
                # Load history
                detail = api("GET", f"/chat/sessions/{sess['id']}")
                if detail:
                    st.session_state.messages = detail.get("messages", [])
                st.rerun()

    # ── Chat window ────────────────────────────────────────────────
    with col_right:
        if st.session_state.active_session_id:
            st.caption(f"Session: `{st.session_state.active_session_id}`")
        else:
            st.caption("New conversation")

        # Render message history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input
        prompt = st.chat_input("Ask about quality issues, root causes, recommendations…")
        if prompt:
            # Optimistic UI
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing…"):
                    result = api(
                        "POST", "/chat/message",
                        json_body={
                            "session_id": st.session_state.active_session_id,
                            "message": prompt,
                        }
                    )
                if result:
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.active_session_id = result["session_id"]

                    # Refresh session list
                    sessions = api("GET", "/chat/sessions") or []
                    st.session_state.chat_sessions = sessions

                    if result.get("sources"):
                        with st.expander("📚 Sources used"):
                            for s in result["sources"]:
                                st.markdown(f"- `{s}`")
                else:
                    st.error("Failed to get a response.")


# ── Profile ────────────────────────────────────────────────────────

def page_profile():
    st.title("👤 Profile")
    u = st.session_state.user
    if not u:
        return

    tab_info, tab_pw, tab_delete = st.tabs(["Update Profile", "Change Password", "Delete Account"])

    with tab_info:
        with st.form("profile_form"):
            full_name = st.text_input("Full Name", value=u.get("full_name") or "")
            username  = st.text_input("Username", value=u.get("username", ""))
            submitted = st.form_submit_button("Save Changes")
        if submitted:
            result = api("PUT", "/users/me", json_body={"full_name": full_name, "username": username})
            if result:
                st.session_state.user = result
                st.success("Profile updated.")

    with tab_pw:
        with st.form("pw_form"):
            current_pw = st.text_input("Current Password", type="password")
            new_pw     = st.text_input("New Password (min 8 chars)", type="password")
            confirm_pw = st.text_input("Confirm New Password", type="password")
            submitted  = st.form_submit_button("Change Password")
        if submitted:
            if new_pw != confirm_pw:
                st.error("New passwords do not match.")
            else:
                result = api(
                    "POST", "/users/me/change-password",
                    json_body={"current_password": current_pw, "new_password": new_pw}
                )
                if result:
                    st.success("Password changed successfully.")

    with tab_delete:
        st.warning("⚠️ This will permanently delete your account and all data.")
        confirm = st.text_input("Type your username to confirm:")
        if st.button("Delete My Account", type="primary"):
            if confirm != u.get("username"):
                st.error("Username does not match.")
            else:
                result = api("DELETE", "/users/me")
                if result is not None:
                    for k in ("token", "user"):
                        st.session_state[k] = None
                    st.session_state.page = "login"
                    st.success("Account deleted.")
                    st.rerun()


# ── Admin ──────────────────────────────────────────────────────────

def page_admin():
    st.title("🛡 Admin Panel — User Management")
    users = api("GET", "/users/") or []
    if not users:
        return

    st.metric("Total Users", len(users))
    st.divider()

    for u in users:
        with st.expander(f"{'🔴' if not u['is_active'] else '🟢'} {u['username']} — {u['email']} ({u['role']})"):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Full Name:** {u.get('full_name') or '—'}")
            c2.markdown(f"**Role:** `{u['role']}`")
            c3.markdown(f"**Joined:** {u['created_at'][:10]}")

            if u["is_active"] and u["id"] != st.session_state.user["id"]:
                if st.button("Deactivate", key=f"deact_{u['id']}"):
                    result = api("PATCH", f"/users/{u['id']}/deactivate")
                    if result:
                        st.warning(f"{u['username']} deactivated.")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════

render_sidebar()

page = st.session_state.page
if page == "login":
    page_login()
elif page == "register":
    page_register()
elif page == "verify_otp":
    page_verify_otp()
elif page == "dashboard":
    page_dashboard()
elif page == "chat":
    page_chat()
elif page == "profile":
    page_profile()
elif page == "admin":
    page_admin()
else:
    page_login()
