"""
insights.py  BlinkIT Product Intelligence Dashboard
Dimension-driven analysis: Overall, City, Category, Discount, Profit, Rating.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from components.frontend import (
    apply_chart_style, render_divider,
    YELLOW, DARK, DARK2, SURFACE, BORDER, MUTED,
    CHART_COLORS, FONT_FAM, YELLOW_SOFT,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _side_card(points):
    bullets = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(f"""
        <div style="background:{YELLOW_SOFT};border:1px solid {BORDER};
                    border-radius:10px;padding:16px 14px;">
            <div style="font-weight:800;color:{DARK};margin-bottom:10px;
                        font-size:0.78rem;text-transform:uppercase;
                        letter-spacing:0.06em;">Key Takeaway</div>
            <ul style="margin:0;padding-left:16px;font-size:0.76rem;
                       color:{DARK};line-height:1.65;">
                {bullets}
            </ul>
        </div>
    """, unsafe_allow_html=True)


def _prepare(df):
    df = df.copy()
    df["offer_type"] = df["offer_type"].fillna("No Offer")
    df["revenue"] = df["final_price"] * df["sold_quantity"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _kpi_card(col, label, value, sub="", inverted=False, names=None):
    bg          = YELLOW    if inverted else DARK
    val_color   = DARK      if inverted else YELLOW
    label_color = "#5A5A2A" if inverted else "#888880"
    sub_color   = "#5A5A3A" if inverted else "#6B6B6B"
    pill_bg     = "rgba(0,0,0,0.12)"   if inverted else "rgba(248,208,0,0.10)"
    pill_color  = "rgba(0,0,0,0.70)"   if inverted else YELLOW
    border      = "#CAAA00"            if inverted else DARK2

    pills_html = ""
    if names:
        shown = names[:9]
        rest  = len(names) - 9
        pills = "".join(
            f'<span style="display:inline-block;background:{pill_bg};color:{pill_color};'
            f'border-radius:4px;padding:1px 8px;font-size:0.55rem;font-weight:700;'
            f'letter-spacing:0.02em;margin:2px 2px 0 0;">{n}</span>'
            for n in shown
        )
        if rest > 0:
            pills += (
                f'<span style="display:inline-block;background:{pill_bg};color:{sub_color};'
                f'border-radius:4px;padding:1px 8px;font-size:0.55rem;font-weight:600;'
                f'margin:2px 2px 0 0;">+{rest} more</span>'
            )
        pills_html = f'<div style="margin-top:8px;line-height:1.9;">{pills}</div>'

    min_h = "148px" if names else "110px"
    col.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:18px 18px 16px;
                    border:1px solid {border};
                    box-shadow:0 2px 10px rgba(0,0,0,{"0.05" if inverted else "0.18"});
                    min-height:{min_h};display:flex;flex-direction:column;">
            <div style="font-size:0.56rem;font-weight:800;text-transform:uppercase;
                        letter-spacing:0.13em;color:{label_color};">{label}</div>
            <div style="font-size:1.6rem;font-weight:900;color:{val_color};
                        letter-spacing:-0.03em;line-height:1.1;margin-top:8px;">{value}</div>
            <div style="font-size:0.64rem;color:{sub_color};margin-top:4px;">{sub}</div>
            {pills_html}
        </div>
    """, unsafe_allow_html=True)


def _kpi_row(specs):
    cols = st.columns(len(specs))
    for i, (col, spec) in enumerate(zip(cols, specs)):
        _kpi_card(
            col,
            spec["label"], spec["value"], spec.get("sub", ""),
            inverted=(i % 2 == 1),
            names=spec.get("names"),
        )


def _render_kpis(df):
    city_rev   = df.groupby("city")["revenue"].sum()
    seller_rev = df.groupby("seller")["revenue"].sum()
    on_time    = (df["delivery_status"] == "On-Time").mean() * 100

    max_city   = city_rev.idxmax();   min_city   = city_rev.idxmin()
    max_seller = seller_rev.idxmax(); min_seller = seller_rev.idxmin()

    _kpi_row([
        {"label": "Product Categories",   "value": str(df["category"].nunique()),
         "sub": "distinct product types",
         "names": sorted(df["category"].unique().tolist())},
        {"label": "Cities Served",        "value": str(df["city"].nunique()),
         "sub": "markets across India",
         "names": sorted(df["city"].unique().tolist())},
        {"label": "Active Sellers",       "value": str(df["seller"].nunique()),
         "sub": "partner seller accounts",
         "names": sorted(df["seller"].unique().tolist())},
        {"label": "Total SKUs",           "value": f"{len(df):,}",
         "sub": "products in catalogue"},
        {"label": "Total Brands",         "value": str(df["brand"].nunique()),
         "sub": "unique brands in catalogue"},
    ])
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _kpi_row([
        {"label": "Total Revenue",        "value": f"₹{df['revenue'].sum() / 1e7:.2f} Crs",
         "sub": "gross revenue across all products"},
        {"label": "Total Units Sold",     "value": f"{df['sold_quantity'].sum():,.0f}",
         "sub": "across all cities & categories"},
        {"label": "Avg Product Rating",   "value": f"{df['rating'].mean():.2f} ★",
         "sub": "customer satisfaction / 5.0"},
        {"label": "On-Time Delivery",     "value": f"{on_time:.0f}%",
         "sub": f"{100 - on_time:.0f}% of orders are delayed"},
    ])
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _kpi_row([
        {"label": "Top City based on Revenue",     "value": max_city,
         "sub": f"₹{city_rev[max_city] / 1e5:.2f}L generated"},
        {"label": "Top Seller based on revenue",   "value": max_seller,
         "sub": f"₹{seller_rev[max_seller] / 1e5:.2f}L generated"},
        {"label": "City generating lowest revenue",  "value": min_city,
         "sub": f"₹{city_rev[min_city] / 1e5:.2f}L - expansion opportunity"},
        {"label": "Seller generating lowest revenue","value": min_seller,
         "sub": f"₹{seller_rev[min_seller] / 1e5:.2f}L - needs attention"},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def show_insights(df):
    df = _prepare(df)
    df["date_added"] = pd.to_datetime(df["date_added"])

    # ── Date range filter ─────────────────────────────────────────────────
    min_date = df["date_added"].min().date()
    max_date = df["date_added"].max().date()

    col_filter, col_badge = st.columns([3, 1])
    with col_filter:
        date_range = st.date_input(
            "Filter by date added",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )
    with col_badge:
        st.markdown(
            f'<div style="background:{DARK};border-radius:8px;padding:10px 14px;'
            f'border:1px solid {DARK2};font-size:0.72rem;color:{YELLOW};font-weight:700;">'
            f'📅 {min_date.strftime("%b %Y")} – {max_date.strftime("%b %Y")}'
            f'<span style="display:block;font-weight:400;color:#888;font-size:0.65rem;margin-top:2px;">'
            f'Full dataset: {len(df):,} products</span></div>',
            unsafe_allow_html=True,
        )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        df = df[(df["date_added"].dt.date >= start) & (df["date_added"].dt.date <= end)]

    if df.empty:
        st.warning("No products found in the selected date range. Try a wider range.")
        return

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _render_kpis(df)
    render_divider()

    tabs = st.tabs([
        "Overall Analysis",
        "City Based",
        "Category Based",
        "Discount & Profit",
        "Rating Based",
    ])
    with tabs[0]: _tab_overall(df)
    with tabs[1]: _tab_city(df)
    with tabs[2]: _tab_category(df)
    with tabs[3]: _tab_discount_profit(df)
    with tabs[4]: _tab_rating(df)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 - OVERALL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def _tab_overall(df):

    # ── Monthly Revenue Trend ─────────────────────────────────────────────
    with st.expander("Revenue Generated by month", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            monthly = (
                df.assign(month=df["date_added"].dt.to_period("M").astype(str))
                .groupby("month")
                .agg(revenue=("revenue","sum"), units=("sold_quantity","sum"))
                .reset_index()
                .sort_values("month")
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["month"], y=monthly["revenue"] / 1e5,
                mode="lines+markers",
                name="Revenue (₹L)",
                line=dict(color=YELLOW, width=2.5),
                marker=dict(size=6, color=YELLOW),
                fill="tozeroy",
                fillcolor="rgba(248,208,0,0.12)",
            ))
            fig.add_trace(go.Scatter(
                x=monthly["month"], y=monthly["units"],
                mode="lines",
                name="Units Sold",
                line=dict(color=CHART_COLORS[4], width=1.8, dash="dot"),
                yaxis="y2",
                visible="legendonly",
            ))
            fig = apply_chart_style(fig, height=380, legend=True)
            fig.update_layout(
                title=dict(text="Monthly Revenue Trend (₹ Lakhs)", font=dict(size=13, weight=700)),
                yaxis=dict(title="Revenue (₹L)"),
                yaxis2=dict(title="Units Sold", overlaying="y", side="right"),
                xaxis=dict(tickangle=-45, title="Month", tickfont=dict(size=9)),
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            peak_month = monthly.loc[monthly["revenue"].idxmax(), "month"]
            low_month  = monthly.loc[monthly["revenue"].idxmin(), "month"]
            peak_rev   = monthly["revenue"].max() / 1e5
            low_rev    = monthly["revenue"].min() / 1e5
            avg_rev    = monthly["revenue"].mean() / 1e5
            first_rev  = monthly["revenue"].iloc[0] / 1e5
            last_rev   = monthly["revenue"].iloc[-1] / 1e5
            trend_dir  = "up" if last_rev > first_rev else "down"
            peak_vs_avg = (peak_rev / avg_rev - 1) * 100
            _side_card([
                f"<b>{peak_month}</b> was the best month at ₹{peak_rev:.1f}L - {peak_vs_avg:.0f}% above the ₹{avg_rev:.1f}L monthly average.",
                f"<b>{low_month}</b> was the weakest at ₹{low_rev:.1f}L - {(1 - low_rev/avg_rev)*100:.0f}% below average.",
                f"Revenue is trending <b>{trend_dir}</b> overall - ₹{first_rev:.1f}L in {monthly['month'].iloc[0]} vs ₹{last_rev:.1f}L in {monthly['month'].iloc[-1]}.",
            ])

    # ── Revenue by Category ───────────────────────────────────────────────
    with st.expander("Revenue by Category", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            cat_rev = (
                df.groupby("category")
                .agg(revenue=("revenue","sum"), units=("sold_quantity","sum"))
                .reset_index()
                .sort_values("revenue", ascending=True)
            )
            fig = go.Figure(go.Bar(
                y=cat_rev["category"], x=cat_rev["revenue"] / 1e3,
                orientation="h", marker_color=YELLOW,
                text=[f"₹{v:.0f}K" for v in cat_rev["revenue"] / 1e3],
                textposition="outside", textfont=dict(size=10),
            ))
            fig = apply_chart_style(fig, height=380, legend=False)
            fig.update_layout(
                title=dict(text="Total Revenue by Product Category", font=dict(size=13, weight=700)),
                xaxis_title="Revenue (₹K)", yaxis_title="",
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            top_cat    = cat_rev.sort_values("revenue", ascending=False).iloc[0]["category"]
            bot_cat    = cat_rev.sort_values("revenue", ascending=True).iloc[0]["category"]
            top_rev_k  = cat_rev.sort_values("revenue", ascending=False).iloc[0]["revenue"] / 1e3
            bot_rev_k  = cat_rev.sort_values("revenue", ascending=True).iloc[0]["revenue"] / 1e3
            rev_ratio  = top_rev_k / bot_rev_k if bot_rev_k > 0 else 0
            _side_card([
                f"<b>{top_cat}</b> leads at ₹{top_rev_k:.0f}K - {rev_ratio:.1f}x more than <b>{bot_cat}</b> (₹{bot_rev_k:.0f}K).",
                f"<b>{bot_cat}</b> is the smallest category. Check if it has fewer SKUs listed or just lower demand.",
            ])

    # ── Brand Pareto ──────────────────────────────────────────────────────
    with st.expander("Brand Concentration", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            brand_rev = (
                df.groupby("brand")["revenue"].sum()
                .sort_values(ascending=False).reset_index()
            )
            brand_rev["cum_pct"] = brand_rev["revenue"].cumsum() / brand_rev["revenue"].sum() * 100
            idx_80 = int((brand_rev["cum_pct"] <= 80).sum())

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=brand_rev["brand"], y=brand_rev["revenue"] / 1e3,
                name="Revenue (₹K)", yaxis="y1",
                marker_color=[YELLOW if i <= idx_80 else DARK2 for i in range(len(brand_rev))],
                text=[f"₹{v:.0f}K" for v in brand_rev["revenue"] / 1e3],
                textposition="outside", textfont=dict(size=9),
            ))
            fig.add_trace(go.Scatter(
                x=brand_rev["brand"], y=brand_rev["cum_pct"],
                name="Cumulative %", yaxis="y2",
                mode="lines", line=dict(color="#EF4444", width=2),
            ))
            fig.add_hline(y=80, yref="y2", line_dash="dash", line_color="#EF4444",
                          annotation_text="80%", annotation_font_size=10)
            fig = apply_chart_style(fig, height=420, legend=True)
            fig.update_layout(
                title=dict(text="Brand Revenue - Pareto Analysis", font=dict(size=13, weight=700)),
                yaxis=dict(title="Revenue (₹K)"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                xaxis=dict(tickangle=-40),
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            total_brands   = len(brand_rev)
            top3_pct       = brand_rev["revenue"].iloc[:3].sum() / brand_rev["revenue"].sum() * 100
            top1_brand     = brand_rev["brand"].iloc[0]
            top1_pct       = brand_rev["revenue"].iloc[0] / brand_rev["revenue"].sum() * 100
            _side_card([
                f"<b>{idx_80 + 1} of {total_brands} brands</b> (yellow) account for 80% of revenue.",
                f"Top brand <b>{top1_brand}</b> alone is {top1_pct:.1f}% of total. Top 3 combined: {top3_pct:.1f}%.",
                f"The remaining {total_brands - idx_80 - 1} grey brands share just 20% of revenue between them.",
            ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 - CITY BASED
# ─────────────────────────────────────────────────────────────────────────────
def _tab_city(df):

    # ── Revenue by City ───────────────────────────────────────────────────
    with st.expander("Revenue by City", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            city_rev = (
                df.groupby("city")
                .agg(revenue=("revenue","sum"), units=("sold_quantity","sum"))
                .reset_index().sort_values("revenue", ascending=True)
            )
            fig = go.Figure(go.Bar(
                y=city_rev["city"], x=city_rev["revenue"] / 1e3,
                orientation="h", marker_color=YELLOW,
                text=[f"₹{v:.0f}K" for v in city_rev["revenue"] / 1e3],
                textposition="outside", textfont=dict(size=10),
            ))
            fig = apply_chart_style(fig, height=400, legend=False)
            fig.update_layout(
                title=dict(text="Total Revenue by City", font=dict(size=13, weight=700)),
                xaxis_title="Revenue (₹K)", yaxis_title="",
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            avg_city_rev  = city_rev["revenue"].mean() / 1e3
            below_avg     = city_rev[city_rev["revenue"] / 1e3 < avg_city_rev]["city"].tolist()
            top2_share    = city_rev.sort_values("revenue", ascending=False).head(2)["revenue"].sum() / city_rev["revenue"].sum() * 100
            _side_card([
                f"Average city revenue is ₹{avg_city_rev:.0f}K.",
                f"Below average: {', '.join(below_avg) if below_avg else 'none'} — most room to grow.",
                f"Top 2 cities account for {top2_share:.0f}% of total city revenue.",
            ])

    # ── City × Category Heatmap ───────────────────────────────────────────
    with st.expander("City × Category Heatmap : where are we not selling what we should be?", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            rev_heat = df.groupby(["city","category"])["revenue"].sum().reset_index()
            pivot    = rev_heat.pivot(index="city", columns="category", values="revenue").fillna(0)
            text_mat = [[f"₹{v/1e3:.0f}K" for v in row] for row in pivot.values]

            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                text=text_mat, texttemplate="%{text}",
                textfont=dict(size=10, family=FONT_FAM),
                colorscale=[[0.0,"#1A1A1A"],[0.4,"#3D3D3D"],[0.7,"#F59E0B"],[1.0,"#F8D000"]],
                showscale=True,
                colorbar=dict(title="Revenue", tickfont=dict(size=9)),
            ))
            fig.update_layout(
                title=dict(text="Revenue Heatmap: City × Category", font=dict(size=13, weight=700)),
                height=430, font=dict(family=FONT_FAM), paper_bgcolor=SURFACE,
                margin=dict(l=16, r=16, t=48, b=16),
                xaxis=dict(tickangle=-20), yaxis=dict(title="City"),
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            top2_cities   = city_rev.sort_values("revenue", ascending=False).head(2)["city"].tolist()
            top2_pivot    = pivot.loc[[c for c in top2_cities if c in pivot.index]]
            weakest_city  = top2_pivot.stack().idxmin()
            weakest_val   = top2_pivot.stack().min() / 1e3
            overall_min   = pivot.stack().idxmin()
            _side_card([
                f"In your top cities, the weakest combo is <b>{weakest_city[0]} × {weakest_city[1]}</b> at ₹{weakest_val:.0f}K - biggest gap to close.",
                f"Overall lowest cell: <b>{overall_min[0]} × {overall_min[1]}</b>. Dark row = whole city underperforms. Dark column = category issue everywhere.",
            ])

    # ── Delivery Performance by City ──────────────────────────────────────
    with st.expander("Delivery Performance by City", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            del_city = (
                df.groupby("city")
                .agg(
                    on_time_pct=("delivery_status", lambda x: (x=="On-Time").mean() * 100),
                    avg_time=("delivery_time_min","mean"),
                )
                .reset_index().sort_values("on_time_pct", ascending=True)
            )
            fig = go.Figure(go.Bar(
                y=del_city["city"], x=del_city["on_time_pct"],
                orientation="h",
                marker_color=[
                    "#EF4444" if v < 70 else YELLOW if v < 85 else CHART_COLORS[4]
                    for v in del_city["on_time_pct"]
                ],
                text=[f"{v:.1f}%" for v in del_city["on_time_pct"]],
                textposition="outside", textfont=dict(size=10),
            ))
            fig.add_vline(x=80, line_dash="dash", line_color=MUTED,
                          annotation_text="80% target", annotation_font_size=10)
            fig = apply_chart_style(fig, height=400, legend=False)
            fig.update_layout(
                title=dict(text="On-Time Delivery % by City", font=dict(size=13, weight=700)),
                xaxis_title="On-Time Delivery %", xaxis_range=[0, 115],
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            bad_cities   = del_city[del_city["on_time_pct"] < 70]["city"].tolist()
            best_city_ot = del_city.sort_values("on_time_pct", ascending=False).iloc[0]["city"]
            best_ot_pct  = del_city.sort_values("on_time_pct", ascending=False).iloc[0]["on_time_pct"]
            worst_city   = del_city.sort_values("on_time_pct").iloc[0]["city"]
            worst_ot_pct = del_city.sort_values("on_time_pct").iloc[0]["on_time_pct"]
            _side_card([
                f"<b>{best_city_ot}</b> leads at {best_ot_pct:.1f}% on-time. <b>{worst_city}</b> is worst at {worst_ot_pct:.1f}%.",
                f"Below 70% cities: {', '.join(bad_cities) if bad_cities else 'none - all above 70%, good state'}.",
            ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 - CATEGORY BASED
# ─────────────────────────────────────────────────────────────────────────────
def _tab_category(df):

    # ── Units Sold vs Stock ───────────────────────────────────────────────
    # with st.expander("Units Sold vs Current Stock", expanded=False):
    #     col_chart, col_insight = st.columns([3, 1])
    #     with col_chart:
    #         cat_bal = (
    #             df.groupby("category")
    #             .agg(total_sold=("sold_quantity","sum"), total_stock=("stock","sum"))
    #             .reset_index().sort_values("total_sold", ascending=True)
    #         )
    #         fig = go.Figure()
    #         fig.add_trace(go.Bar(
    #             y=cat_bal["category"], x=cat_bal["total_sold"],
    #             orientation="h", name="Units Sold",
    #             marker_color=YELLOW, opacity=0.9,
    #         ))
    #         fig.add_trace(go.Bar(
    #             y=cat_bal["category"], x=cat_bal["total_stock"],
    #             orientation="h", name="Current Stock",
    #             marker_color=DARK2, opacity=0.85,
    #         ))
    #         fig = apply_chart_style(fig, height=400, legend=True)
    #         fig.update_layout(
    #             title=dict(text="Units Sold vs Current Stock by Category", font=dict(size=13, weight=700)),
    #             barmode="group", xaxis_title="Units", yaxis_title="",
    #         )
    #         st.plotly_chart(fig, width='stretch')
    #     with col_insight:
    #         cat_bal["sold_to_stock"] = cat_bal["total_sold"] / (cat_bal["total_stock"] + 1)
    #         at_risk_cat  = cat_bal.sort_values("sold_to_stock", ascending=False).iloc[0]["category"]
    #         at_risk_rat  = cat_bal.sort_values("sold_to_stock", ascending=False).iloc[0]["sold_to_stock"]
    #         overstock_cat = cat_bal.sort_values("sold_to_stock").iloc[0]["category"]
    #         overstock_rat = cat_bal.sort_values("sold_to_stock").iloc[0]["sold_to_stock"]
    #         _side_card([
    #             f"<b>{at_risk_cat}</b> sells {at_risk_rat:.1f}x its stock - highest stockout risk, reorder now.",
    #             f"<b>{overstock_cat}</b> has sold only {overstock_rat:.1f}x its stock - most overstocked, cash sitting idle.",
    #         ])

    # ── Demand–Supply Quadrant ────────────────────────────────────────────
    with st.expander("Demand–Supply Quadrant", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            quad = (
                df.groupby("category")
                .agg(avg_demand=("demand_index","mean"), avg_stock=("stock","mean"))
                .reset_index()
            )
            med_d = quad["avg_demand"].median()
            med_s = quad["avg_stock"].median()

            def _qlabel(row):
                hi_d = row["avg_demand"] >= med_d
                hi_s = row["avg_stock"]  >= med_s
                if hi_d and not hi_s: return "Reorder Now"
                if hi_d and hi_s:     return "Healthy"
                if not hi_d and hi_s: return "Overstocked"
                return "Phase Out"

            quad["quadrant"] = quad.apply(_qlabel, axis=1)
            color_map = {
                "Reorder Now": "#EF4444",
                "Healthy":     CHART_COLORS[4],
                "Overstocked": "#F59E0B",
                "Phase Out":   MUTED,
            }
            fig = px.scatter(
                quad, x="avg_demand", y="avg_stock",
                color="quadrant",
                color_discrete_map=color_map,
                text="category",
                labels={"avg_demand":"Avg Demand Index","avg_stock":"Avg Stock Level"},
            )
            fig.update_traces(marker=dict(size=18, line=dict(width=1.5, color=DARK)))
            fig.add_vline(x=med_d, line_dash="dash", line_color=MUTED, line_width=1)
            fig.add_hline(y=med_s, line_dash="dash", line_color=MUTED, line_width=1)
            fig = apply_chart_style(fig, height=460, legend=True)
            fig.update_traces(textposition="top center", textfont=dict(size=11))
            fig.update_layout(title=dict(text="Demand–Supply Quadrant by Category", font=dict(size=13, weight=700)))
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            reorder_cats   = quad[quad["quadrant"] == "Reorder Now"]["category"].tolist()
            overstock_cats = quad[quad["quadrant"] == "Overstocked"]["category"].tolist()
            healthy_cats   = quad[quad["quadrant"] == "Healthy"]["category"].tolist()
            _side_card([
                f"<b>Reorder Now</b> ({len(reorder_cats)}): {', '.join(reorder_cats) if reorder_cats else 'none'}.",
                f"<b>Overstocked</b> ({len(overstock_cats)}): {', '.join(overstock_cats) if overstock_cats else 'none'}.",
                f"<b>Healthy</b> ({len(healthy_cats)}): {', '.join(healthy_cats) if healthy_cats else 'none'}.",
            ])



# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 - DISCOUNT & PROFIT
# ─────────────────────────────────────────────────────────────────────────────
def _tab_discount_profit(df):

    # ── Discount vs Margin by Category ───────────────────────────────────
    with st.expander("Discount vs Profit Margin by Category", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            ml = (
                df.groupby("category")
                .agg(
                    avg_discount=("discount_pct","mean"),
                    avg_margin=("profit_margin_pct","mean"),
                    total_revenue=("revenue","sum"),
                )
                .reset_index()
            )
            med_disc = ml["avg_discount"].median()
            med_marg = ml["avg_margin"].median()

            fig = px.scatter(
                ml, x="avg_discount", y="avg_margin",
                size="total_revenue", color="category",
                text="category", size_max=58,
                color_discrete_sequence=CHART_COLORS,
                labels={"avg_discount":"Avg Discount (%)","avg_margin":"Avg Profit Margin (%)"},
            )
            fig.add_vline(x=med_disc, line_dash="dash", line_color=MUTED,
                          annotation_text="Median Discount", annotation_font_size=9)
            fig.add_hline(y=med_marg, line_dash="dash", line_color=MUTED,
                          annotation_text="Median Margin", annotation_font_size=9,
                          annotation_position="bottom right")
            fig.add_annotation(
                x=ml["avg_discount"].max() - 0.5, y=ml["avg_margin"].min() + 0.3,
                text="Margin Leakage Zone", showarrow=False,
                font=dict(size=10, color="#EF4444"), xanchor="right",
            )
            fig = apply_chart_style(fig, height=460, legend=True)
            fig.update_traces(textposition="top center", textfont=dict(size=10))
            fig.update_layout(
                title=dict(text="Discount vs Margin by Category (bubble size = revenue)", font=dict(size=13, weight=700)),
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            danger = ml[(ml["avg_discount"] > med_disc) & (ml["avg_margin"] < med_marg)]
            safe   = ml[(ml["avg_discount"] < med_disc) & (ml["avg_margin"] > med_marg)]
            danger_cats = danger.sort_values("total_revenue", ascending=False)["category"].tolist()
            safe_cats   = safe["category"].tolist()
            _side_card([
                f"Leakage zone (high discount + low margin): {', '.join(danger_cats) if danger_cats else 'none'}.",
                f"Safe zone (low discount + high margin): {', '.join(safe_cats) if safe_cats else 'none'}.",
            ])

    # ── Category × Offer Type Heatmap ────────────────────────────────────
    with st.expander("Category × Offer Type Heatmap", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            cat_offer = (
                df.groupby(["category", "offer_type"])["sold_quantity"]
                .mean().reset_index()
            )
            pivot_co = cat_offer.pivot(index="category", columns="offer_type", values="sold_quantity").fillna(0)
            text_co  = [[f"{v:.0f}" for v in row] for row in pivot_co.values]

            fig = go.Figure(go.Heatmap(
                z=pivot_co.values,
                x=pivot_co.columns.tolist(),
                y=pivot_co.index.tolist(),
                text=text_co, texttemplate="%{text}",
                textfont=dict(size=11, family=FONT_FAM),
                colorscale=[[0.0,"#1A1A1A"],[0.4,"#3D3D3D"],[0.7,"#F59E0B"],[1.0,"#F8D000"]],
                showscale=True,
                colorbar=dict(title="Avg Units Sold", tickfont=dict(size=9)),
            ))
            fig.update_layout(
                title=dict(text="Avg Units Sold: Category × Offer Type", font=dict(size=13, weight=700)),
                height=420, font=dict(family=FONT_FAM), paper_bgcolor=SURFACE,
                margin=dict(l=16, r=16, t=48, b=16),
                xaxis=dict(title="Offer Type", tickangle=-20),
                yaxis=dict(title="Category"),
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            best_cell  = cat_offer.sort_values("sold_quantity", ascending=False).iloc[0]
            worst_cell = cat_offer.sort_values("sold_quantity").iloc[0]
            best_offer_overall = (
                cat_offer.groupby("offer_type")["sold_quantity"].mean()
                .idxmax()
            )
            _side_card([
                f"Best combo: <b>{best_cell['category']} × {best_cell['offer_type']}</b> at {best_cell['sold_quantity']:.0f} avg units.",
                f"Weakest combo: <b>{worst_cell['category']} × {worst_cell['offer_type']}</b> at {worst_cell['sold_quantity']:.0f} avg units.",
                f"Across all categories, <b>{best_offer_overall}</b> is the most effective offer type.",
            ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 - RATING BASED
# ─────────────────────────────────────────────────────────────────────────────
def _tab_rating(df):

    # ── Rating by Category ────────────────────────────────────────────────
    with st.expander("Rating by Category", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            rat_cat = (
                df.groupby("category")
                .agg(avg_rating=("rating","mean"), count=("rating","count"))
                .reset_index()
                .sort_values("avg_rating", ascending=True)
            )
            colors = [
                "#EF4444" if r < 3.5 else YELLOW if r < 4.0 else CHART_COLORS[4]
                for r in rat_cat["avg_rating"]
            ]
            fig = go.Figure(go.Bar(
                y=rat_cat["category"], x=rat_cat["avg_rating"],
                orientation="h", marker_color=colors,
                text=[f"★ {v:.2f}  ({c:,} reviews)" for v, c in zip(rat_cat["avg_rating"], rat_cat["count"])],
                textposition="outside", textfont=dict(size=10),
            ))
            fig.add_vline(x=4.0, line_dash="dash", line_color=MUTED,
                          annotation_text="4.0 target", annotation_font_size=10)
            fig = apply_chart_style(fig, height=380, legend=False)
            fig.update_layout(
                title=dict(text="Average Customer Rating by Category", font=dict(size=13, weight=700)),
                xaxis_title="Avg Rating (/ 5.0)", xaxis_range=[0, 5.5],
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            low_rated    = rat_cat[rat_cat["avg_rating"] < 3.5]["category"].tolist()
            top_rated_cat = rat_cat.sort_values("avg_rating", ascending=False).iloc[0]["category"]
            top_rated_val = rat_cat.sort_values("avg_rating", ascending=False).iloc[0]["avg_rating"]
            worst_cat    = rat_cat.sort_values("avg_rating").iloc[0]["category"]
            worst_val    = rat_cat.sort_values("avg_rating").iloc[0]["avg_rating"]
            overall_avg  = rat_cat["avg_rating"].mean()
            _side_card([
                f"<b>{top_rated_cat}</b> is highest rated at ★{top_rated_val:.2f}. <b>{worst_cat}</b> is lowest at ★{worst_val:.2f} (avg ★{overall_avg:.2f}).",
                f"Below ★3.5: {', '.join(low_rated) if low_rated else 'none - all above 3.5, solid state'}.",
            ])

    # ── Delivery Speed vs Rating ───────────────────────────────────────────
    with st.expander("Delivery Speed vs Rating", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            city_del = (
                df.groupby("city")
                .agg(
                    avg_delivery=("delivery_time_min","mean"),
                    avg_rating=("rating","mean"),
                    delayed_pct=("delivery_status", lambda x: (x=="Delayed").mean() * 100),
                )
                .reset_index()
            )
            slope, intercept, r_val, _, _ = stats.linregress(
                city_del["avg_delivery"], city_del["avg_rating"]
            )
            x_line = np.linspace(city_del["avg_delivery"].min(), city_del["avg_delivery"].max(), 100)
            y_line = slope * x_line + intercept

            fig = px.scatter(
                city_del, x="avg_delivery", y="avg_rating",
                size="delayed_pct", color="delayed_pct",
                text="city", size_max=50,
                color_continuous_scale=[[0.0, CHART_COLORS[4]], [0.5, YELLOW], [1.0, "#EF4444"]],
                labels={"avg_delivery":"Avg Delivery Time (min)","avg_rating":"Avg Rating","delayed_pct":"Delay %"},
            )
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=DARK, width=1.5, dash="dash"),
                name=f"Trend  r={r_val:.3f}", showlegend=True,
            ))
            fig = apply_chart_style(fig, height=440, legend=True)
            fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            fig.update_layout(
                title=dict(text="City: Delivery Speed vs Customer Rating  (bubble size = Delay %)", font=dict(size=13, weight=700)),
                coloraxis_colorbar=dict(title="Delay %", tickfont=dict(size=9)),
            )
            st.plotly_chart(fig, width='stretch')
        with col_insight:
            direction    = "slower delivery does hurt ratings" if r_val < -0.1 else "delivery speed alone isn't the main driver here"
            fastest_city = city_del.sort_values("avg_delivery").iloc[0]["city"]
            fastest_time = city_del.sort_values("avg_delivery").iloc[0]["avg_delivery"]
            slowest_city = city_del.sort_values("avg_delivery").iloc[-1]["city"]
            slowest_time = city_del.sort_values("avg_delivery").iloc[-1]["avg_delivery"]
            _side_card([
                f"Correlation r = {r_val:.3f} - {direction}.",
                f"<b>{fastest_city}</b> is fastest at {fastest_time:.0f} min avg. <b>{slowest_city}</b> is slowest at {slowest_time:.0f} min.",
            ])

    # ── Seller Quality Scorecard ──────────────────────────────────────────
    with st.expander("Rating by Seller", expanded=False):
        col_chart, col_insight = st.columns([3, 1])
        with col_chart:
            seller_raw = (
                df.groupby("seller")
                .agg(
                    avg_rating=("rating","mean"),
                    on_time_pct=("delivery_status", lambda x: (x=="On-Time").mean() * 100),
                    avg_margin=("profit_margin_pct","mean"),
                    total_revenue=("revenue","sum"),
                )
                .reset_index()
            )
            metrics = ["avg_rating","on_time_pct","avg_margin","total_revenue"]
            scaler  = MinMaxScaler()
            scaled  = seller_raw.copy()
            scaled[metrics] = scaler.fit_transform(seller_raw[metrics])
            scaled["score"] = scaled[metrics].mean(axis=1) * 100

            seller_sorted = seller_raw.copy()
            seller_sorted["score"] = scaled["score"].values
            seller_sorted = seller_sorted.sort_values("avg_rating", ascending=True)
            bar_colors = [
                "#EF4444" if r < 3.5 else YELLOW if r < 4.0 else CHART_COLORS[4]
                for r in seller_sorted["avg_rating"]
            ]
            fig = go.Figure(go.Bar(
                y=seller_sorted["seller"], x=seller_sorted["avg_rating"],
                orientation="h", marker_color=bar_colors,
                text=[f"★{r:.2f}  •  {ot:.0f}% on-time" for r, ot in zip(seller_sorted["avg_rating"], seller_sorted["on_time_pct"])],
                textposition="outside", textfont=dict(size=10),
            ))
            fig.add_vline(x=4.0, line_dash="dash", line_color=MUTED,
                          annotation_text="4.0 target", annotation_font_size=10)
            fig = apply_chart_style(fig, height=360, legend=False)
            fig.update_layout(
                title=dict(text="Avg Customer Rating by Seller", font=dict(size=13, weight=700)),
                xaxis_title="Avg Rating (/ 5.0)", xaxis_range=[0, 5.5],
            )
            st.plotly_chart(fig, width='stretch')

            lb = seller_raw.copy()
            lb.insert(1, "Score", scaled["score"].round(1))
            lb = lb.sort_values("Score", ascending=False).reset_index(drop=True)
            lb["avg_rating"]    = lb["avg_rating"].map("{:.2f}".format)
            lb["on_time_pct"]   = lb["on_time_pct"].map("{:.1f}%".format)
            lb["avg_margin"]    = lb["avg_margin"].map("{:.1f}%".format)
            lb["total_revenue"] = lb["total_revenue"].map("₹{:,.0f}".format)
            lb.columns = ["Seller","Score","Avg Rating","On-Time %","Avg Margin","Revenue"]

        with col_insight:
            top_scorer    = lb.iloc[0]["Seller"]
            top_score_val = lb.iloc[0]["Score"]
            bot_scorer    = lb.iloc[-1]["Seller"]
            bot_score_val = lb.iloc[-1]["Score"]
            top_scorer_rating = lb.iloc[0]["Avg Rating"]
            _side_card([
                f"<b>{top_scorer}</b> tops the scorecard at {top_score_val:.1f}/100 with avg rating {top_scorer_rating}.",
                f"<b>{bot_scorer}</b> is last at {bot_score_val:.1f}/100 - check which dimension is pulling them down.",
            ])
