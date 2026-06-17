"""
frontend.py : BlinkIT Product Intelligence Dashboard
All visual identity: CSS, colour palette, sidebar, banners,
chart helpers, and reusable UI components.

Import in app.py:
    from components.frontend import (
        apply_styles, apply_chart_style,
        render_sidebar, render_home_banner,
        render_page_banner, render_insight_card,
        render_info_strip, render_stat_row,
        hbar_chart, vbar_chart,
        YELLOW, DARK, SURFACE, CHART_COLORS,
    )
"""

import base64
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# BRAND PALETTE
# Yellow: #F8D000  |  Dark: #1A1A1A  |  Off-white: #FAFAF5
# ─────────────────────────────────────────────────────────────────────────────
YELLOW      = "#F8D000"
YELLOW_DARK = "#D4B000"
YELLOW_SOFT = "#FEF3C7"
DARK        = "#1A1A1A"
DARK2       = "#2D2D2D"
DARK3       = "#3D3D3D"
OFFWHITE    = "#FAFAF5"
SURFACE     = "#FFFFFF"
BORDER      = "#E8E8E0"
MUTED       = "#6B6B6B"
FONT_FAM    = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
GRID_COLOR  = "#F4F4F0"

CHART_COLORS = [
    "#F8D000", "#1A1A1A", "#F59E0B", "#92400E",
    "#065F46", "#1E3A5F", "#7C2D12", "#4C1D95",
]

# Navigation pages — single source of truth
NAV_PAGES = ["Home", "Data Analysis & Insights", "Rating Predictor", "Chat with blinkBOT"]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (call once at the very top of app.py, before anything else)
# ─────────────────────────────────────────────────────────────────────────────
def set_page_config():
    st.set_page_config(
        page_title="BlinkInsighT",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
def apply_styles():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: {FONT_FAM}; }}

/* ── App background ────────────────────────────────── */
.stApp {{ background: {OFFWHITE}; }}

/* ── Hide Streamlit chrome ─────────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; height: 0; }}
[data-testid="collapsedControl"] {{ display: none !important; }}
.block-container {{ padding-top: 1.2rem !important; }}

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {DARK} !important;
    border-right: 1px solid {DARK2};
}}
[data-testid="stSidebar"] * {{ color: #C8C8C0; }}
[data-testid="stSidebar"] hr {{ border-color: {DARK2} !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    font-size: 0.84rem;
    font-weight: 500;
    padding: 5px 0;
    color: #A0A09A !important;
}}
section[data-testid="stSidebar"] {{
    transform: none !important;
    min-width: 256px !important;
    width: 256px !important;
    visibility: visible !important;
}}
section[data-testid="stSidebar"][aria-expanded="false"] {{
    margin-left: 0 !important;
    transform: none !important;
}}

/* ── Metric cards ──────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}}
[data-testid="metric-container"] label {{
    color: {MUTED} !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="metric-container"] [data-testid="metric-value"] {{
    color: {DARK} !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
}}

/* ── Tabs ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0px;
    background: transparent;
    border-bottom: 2px solid {BORDER};
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 0;
    padding: 10px 20px;
    font-size: 0.82rem;
    font-weight: 600;
    color: {MUTED} !important;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {DARK} !important;
    border-bottom: 2px solid {YELLOW} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 24px; }}

/* ── Typography ────────────────────────────────────── */
h1 {{
    color: {DARK} !important;
    font-weight: 900 !important;
    font-size: 1.9rem !important;
    letter-spacing: -0.03em;
}}
h2 {{
    color: {DARK} !important;
    font-weight: 700 !important;
    font-size: 1.25rem !important;
    letter-spacing: -0.02em;
}}
h3 {{
    color: {DARK} !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}}

/* ── Insight card ──────────────────────────────────── */
.insight-card {{
    background: {SURFACE};
    border-radius: 10px;
    padding: 20px 24px;
    border: 1px solid {BORDER};
    margin: 12px 0 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.insight-label {{
    font-size: 0.65rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {MUTED};
    margin-bottom: 10px;
}}
.insight-body {{
    font-size: 0.88rem;
    color: #3D3D3D;
    line-height: 1.8;
}}
.insight-body ul {{ margin: 0; padding-left: 16px; }}
.insight-body li {{ margin-bottom: 5px; }}

/* ── Info strip ────────────────────────────────────── */
.info-strip {{
    background: {YELLOW_SOFT};
    border-left: 3px solid {YELLOW};
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0 20px;
    font-size: 0.87rem;
    color: #78350F;
    line-height: 1.7;
}}

/* ── Nav cards ─────────────────────────────────────── */
.nav-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 16px 0 24px;
}}
.nav-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 22px 18px;
    text-align: center;
    cursor: default;
    transition: box-shadow 0.15s, border-color 0.15s;
}}
.nav-card:hover {{
    box-shadow: 0 4px 16px rgba(248,208,0,0.18);
    border-color: {YELLOW};
}}
.nav-card-icon {{
    width: 36px; height: 36px;
    border-radius: 8px;
    background: {YELLOW_SOFT};
    margin: 0 auto 12px;
    display: flex; align-items: center; justify-content: center;
}}
.nav-card-title {{ font-size: 0.85rem; font-weight: 700; color: {DARK}; }}
.nav-card-sub   {{ font-size: 0.72rem; color: {MUTED}; margin-top: 3px; }}

/* ── Pill tag ───────────────────────────────────────── */
.pill {{
    display: inline-block;
    background: {YELLOW_SOFT};
    color: #78350F;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}}

/* ── Section divider ────────────────────────────────── */
.divider {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 28px 0;
}}

/* ── Stat row ───────────────────────────────────────── */
.stat-row {{
    display: flex; gap: 16px; flex-wrap: wrap;
    margin: 16px 0;
}}
.stat-box {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 18px;
    flex: 1; min-width: 120px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.stat-box-label {{
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: {MUTED};
}}
.stat-box-value {{
    font-size: 1.4rem; font-weight: 800;
    color: {DARK}; margin-top: 2px;
}}
.stat-box-accent {{
    border-left: 3px solid {YELLOW};
}}

/* ── Dataframe ──────────────────────────────────────── */
.dataframe {{ font-size: 0.83rem !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    """
    Renders the sidebar logo, nav radio, and dataset quick-stats.
    Returns the selected page name string.
    """
    with st.sidebar:

        # ── Logo / wordmark ───────────────────────────────────────────────
        st.markdown(f"""
            <div style="padding: 24px 16px 16px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:32px; height:32px; background:{YELLOW};
                                 border-radius:6px; display:flex; align-items:center;
                                 justify-content:center;">
                        <div style="width:14px; height:14px; background:{DARK};
                                     border-radius:2px;"></div>
                    </div>
                    <div>
                        <div style="font-size:1.15rem; font-weight:900;
                                     color:#F8F8F0; letter-spacing:-0.03em;">
                            blink<span style="color:#22C55E;">InsighT</span>
                        </div>
                        <div style="color:#888880; font-size:0.6rem;
                                    letter-spacing:0.1em; margin-top:1px;">
                            know your products · grow your sales
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Navigation ────────────────────────────────────────────────────
        page = st.radio(
            "Navigate",
            NAV_PAGES,
            label_visibility="collapsed",
        )

        st.markdown("---")

       
    

    return page


# ─────────────────────────────────────────────────────────────────────────────
# BANNERS
# ─────────────────────────────────────────────────────────────────────────────
def render_home_banner(image_path: str):
    """
    Full-height hero banner for the Home page.
    Pass the absolute or relative path to a local image/video file.
    Supports .mp4 (video autoplay) and image files (.jpg / .png).
    """
    ext = image_path.rsplit(".", 1)[-1].lower()

    with open(image_path, "rb") as f:
        media_b64 = base64.b64encode(f.read()).decode()

    if ext == "mp4":
        media_tag = f"""
            <video autoplay muted loop playsinline
                style="width:100%; height:100%; object-fit:cover;">
                <source src="data:video/mp4;base64,{media_b64}" type="video/mp4">
            </video>
        """
    else:
        mime = "image/png" if ext == "png" else "image/jpeg"
        media_tag = f"""
            <img src="data:{mime};base64,{media_b64}"
                 style="width:100%; height:100%; object-fit:cover;" />
        """

    st.html(f"""
        <div style="
            position:relative; height:450px;
            border-radius:16px; overflow:hidden; margin-bottom:30px;
        ">
            {media_tag}
            <div style="
                position:absolute; inset:0;
                background:linear-gradient(
                    90deg,
                    rgba(0,0,0,0.75) 0%,
                    rgba(0,0,0,0.35) 60%,
                    transparent 100%
                );
            "></div>
            <div style="
                position:absolute; top:50%; left:50px;
                transform:translateY(-50%); color:white;
            ">
                <h1 style="
                    font-size:48px; font-weight:900; margin:0 0 10px;
                    letter-spacing:-0.03em; line-height:1.05;
                ">
                    Hello!
                </h1>
               
                
            </div>
        </div>
    """)


def render_page_banner(image_path: str, eyebrow: str, title: str, subtitle: str):
    """
    Compact 160 px banner used at the top of inner pages
    (Data Analysis & Insights, ML Models).
    """
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    st.html(f"""
        <div style="
            position:relative; height:160px;
            border-radius:12px; overflow:hidden;
        ">
            <img src="data:{mime};base64,{img_b64}"
                 style="width:100%; height:100%; object-fit:cover;" />
            <div style="
                position:absolute; inset:0;
                background:linear-gradient(
                    90deg,
                    rgba(26,26,26,0.88) 0%,
                    rgba(26,26,26,0.50) 100%
                );
            "></div>
            <div style="
                position:absolute; top:50%; left:36px;
                transform:translateY(-50%); color:white;
            ">
                <div style="
                    color:{YELLOW}; font-size:11px;
                    font-weight:800; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:6px;
                ">
                    {eyebrow}
                </div>
                <h1 style="margin:0; font-size:1.7rem; font-weight:900;">{title}</h1>
                <p style="margin:4px 0 0; font-size:0.82rem;
                           color:rgba(255,255,255,0.65);">{subtitle}</p>
            </div>
        </div>
    """)


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_insight_card(body_html: str, label: str = "Key Observations"):
    """Yellow-accented insight card with a small uppercase label."""
    st.markdown(f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-body">{body_html}</div>
        </div>
    """, unsafe_allow_html=True)


def render_info_strip(html: str):
    """Yellow left-bordered info strip for contextual notes."""
    st.markdown(f'<div class="info-strip">{html}</div>', unsafe_allow_html=True)


def render_stat_row(stats: list[tuple[str, str]], accent_index: int = 0):
    """
    Render a horizontal row of stat boxes.
    stats: list of (label, value) tuples
    accent_index: which box gets the yellow left border
    """
    boxes = ""
    for i, (label, value) in enumerate(stats):
        accent = "stat-box-accent" if i == accent_index else ""
        boxes += f"""
            <div class="stat-box {accent}">
                <div class="stat-box-label">{label}</div>
                <div class="stat-box-value">{value}</div>
            </div>
        """
    st.markdown(f'<div class="stat-row">{boxes}</div>', unsafe_allow_html=True)


def render_nav_cards(cards: list[tuple[str, str]]):
    """
    Render a 4-column grid of nav cards on the Home page.
    cards: list of (title, subtitle) tuples
    """
    html = '<div class="nav-grid">'
    for title, sub in cards:
        html += f"""
            <div class="nav-card">
                <div class="nav-card-icon">
                    <div style="width:16px;height:16px;
                                background:{YELLOW};border-radius:3px;"></div>
                </div>
                <div class="nav-card-title">{title}</div>
                <div class="nav-card-sub">{sub}</div>
            </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def render_pill(text: str) -> str:
    """Return HTML string for an inline pill tag."""
    return f'<span class="pill">{text}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def apply_chart_style(fig, height: int = 400, legend: bool = True):
    """Apply consistent BlinkIT brand styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT_FAM, color="#3D3D3D", size=11.5),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=16, r=16, t=48, b=16),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10.5),
        ) if legend else {},
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, tickfont=dict(size=11),
        ),
    )
    return fig


def hbar_chart(data, y_col: str, x_col: str, title: str,
               fmt_fn, color=None, x_title: str = ""):
    """Horizontal bar chart — ranked lists, top-N, etc."""
    color = color or YELLOW
    fig = go.Figure(go.Bar(
        y=data[y_col],
        x=data[x_col],
        orientation="h",
        marker_color=color,
        text=[fmt_fn(v) for v in data[x_col]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig = apply_chart_style(fig, legend=False)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, weight=700)),
        xaxis_title=x_title,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def vbar_chart(data, x_col: str, y_col: str, title: str,
               fmt_fn, color=None, y_title: str = ""):
    """Vertical bar chart — comparisons across categories."""
    color = color or YELLOW
    fig = go.Figure(go.Bar(
        x=data[x_col],
        y=data[y_col],
        marker_color=color,
        text=[fmt_fn(v) for v in data[y_col]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig = apply_chart_style(fig, legend=False)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, weight=700)),
        yaxis_title=y_title,
    )
    return fig