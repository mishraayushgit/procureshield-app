"""
ProcureShield AI
Fresh Streamlit SaaS application for connected procurement fraud intelligence.

Run:
    streamlit run procureshield_ai.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from textwrap import dedent
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="ProcureShield AI",
    page_icon="PS",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_NAME = "ProcureShield"
APP_SUBTITLE = "AI PLATFORM"

C = {
    "bg": "#07111f",
    "bg2": "#0b1728",
    "sidebar": "#0d1a31",
    "sidebar2": "#142447",
    "panel": "rgba(17, 31, 55, 0.82)",
    "panel2": "rgba(22, 39, 70, 0.70)",
    "line": "rgba(132, 179, 207, 0.18)",
    "muted": "#8fa4bd",
    "text": "#eef7ff",
    "soft": "#c4d4e6",
    "teal": "#12d6c5",
    "teal2": "#28f2d2",
    "blue": "#57a6ff",
    "purple": "#8b7cff",
    "amber": "#ffc857",
    "red": "#ff5d6c",
    "green": "#43e39f",
}

PAGES = [
    ("MAIN", "Dashboard", "DB"),
    ("MAIN", "Transaction Feed", "TF"),
    ("MAIN", "Vendor Intelligence", "VI"),
    ("ANALYTICS", "Risk Reports", "RR"),
    ("ANALYTICS", "AI Monitoring Center", "AI"),
    ("SETTINGS", "Audit Logs", "AL"),
    ("SETTINGS", "Configuration", "CF"),
]

PAYMENTS = ["UPI", "NEFT", "RTGS", "Card", "Wire", "Cheque"]
DEPARTMENTS = ["IT", "Finance", "Operations", "Legal", "Facilities", "Marketing", "Supply Chain"]
CATEGORIES = ["Software", "Consulting", "Logistics", "Hardware", "Office", "Travel", "Security"]


def html_fragment(markup: str) -> str:
    """Keep Streamlit markdown from treating indented HTML as code blocks."""
    return "\n".join(line.strip() for line in dedent(markup).strip().splitlines() if line.strip())


def render_html(markup: str, target=st) -> None:
    target.markdown(html_fragment(markup), unsafe_allow_html=True)


def money(value: float) -> str:
    return f"INR {value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def risk_label(score: float) -> str:
    if score >= 76:
        return "Critical"
    if score >= 58:
        return "High"
    if score >= 36:
        return "Elevated"
    return "Clear"


def risk_color(score: float) -> str:
    if score >= 76:
        return C["red"]
    if score >= 58:
        return C["amber"]
    if score >= 36:
        return C["blue"]
    return C["green"]


@st.cache_data(show_spinner=False)
def generate_procurement_data(seed: int = 12, rows: int = 560) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    vendor_ids = [f"VND-{1000 + i}" for i in range(1, 34)]
    vendor_names = [
        "Astra Logic Systems",
        "Northstar Cloudworks",
        "Meridian Infosec",
        "BluePeak Facilities",
        "Kairo Supply Network",
        "Veridian Consulting",
        "Zenith Hardware Co",
        "Helio Payments",
        "Quantum OfficeWorks",
        "Trident Logistics",
        "OrbitEdge Software",
        "Nexa Audit Partners",
        "Copperline Services",
        "Silvergate Ops",
        "Ardent Security Labs",
        "Tessera Procurement",
        "Cobalt Travel Desk",
        "CivicGrid Utilities",
        "Altura Legal Advisory",
        "Saffron DataWorks",
        "Axion Managed IT",
        "Prism Compliance",
        "Vantage Systems",
        "Nimbus Support",
        "Keystone Materials",
        "Crestline Analytics",
        "Pioneer LeaseWorks",
        "SignalPoint AI",
        "Vertex Engineering",
        "Mosaic Enterprise",
        "NovaShield Services",
        "LumenPay Network",
        "Evergreen Sourcing",
    ]
    vendors = pd.DataFrame(
        {
            "vendor_id": vendor_ids,
            "vendor_name": vendor_names,
            "country": rng.choice(["India", "Singapore", "UAE", "Germany", "US"], len(vendor_ids), p=[0.64, 0.1, 0.1, 0.08, 0.08]),
            "onboarded": [datetime(2020, 1, 1) + timedelta(days=int(x)) for x in rng.integers(1, 1800, len(vendor_ids))],
            "compliance_score": rng.integers(58, 98, len(vendor_ids)),
            "base_trust": rng.integers(48, 96, len(vendor_ids)),
            "network_density": rng.uniform(0.08, 0.88, len(vendor_ids)),
            "bank_changes": rng.integers(0, 5, len(vendor_ids)),
        }
    )

    suspicious = {vendor_ids[i] for i in [7, 15, 21, 27, 31]}
    start = datetime.now() - timedelta(days=120)
    records: List[Dict[str, object]] = []

    for idx in range(rows):
        vendor = rng.choice(vendor_ids, p=np.linspace(1.8, 0.7, len(vendor_ids)) / np.linspace(1.8, 0.7, len(vendor_ids)).sum())
        v_ix = vendor_ids.index(vendor)
        department = rng.choice(DEPARTMENTS)
        category = rng.choice(CATEGORIES)
        payment = rng.choice(PAYMENTS, p=[0.23, 0.28, 0.18, 0.14, 0.08, 0.09])
        base_amount = rng.lognormal(mean=11.45, sigma=0.65)
        if department in ["IT", "Supply Chain"]:
            base_amount *= 1.32
        if payment in ["Wire", "UPI"]:
            base_amount *= 1.08
        if vendor in suspicious:
            base_amount *= rng.uniform(1.22, 2.45)
        amount = float(np.clip(base_amount, 12000, 1800000))
        created = start + timedelta(hours=int(rng.integers(0, 120 * 24)), minutes=int(rng.integers(0, 60)))
        hour = created.hour
        weekend = created.weekday() >= 5
        split = amount > 390000 and amount < 500000 and rng.random() < 0.55
        late = hour < 7 or hour > 21
        vendor_row = vendors.loc[vendors["vendor_id"] == vendor].iloc[0]
        rule_score = 18
        rule_score += max(0, (amount - 180000) / 11500)
        rule_score += 15 if payment in ["UPI", "Wire"] else 0
        rule_score += 13 if late else 0
        rule_score += 8 if weekend else 0
        rule_score += 15 if split else 0
        rule_score += (100 - vendor_row["base_trust"]) * 0.45
        rule_score += vendor_row["bank_changes"] * 4
        rule_score += 12 if vendor in suspicious else 0
        risk = float(np.clip(rule_score + rng.normal(0, 8), 4, 99))
        fraud = int(risk > 72 or (vendor in suspicious and risk > 58 and rng.random() < 0.5))
        records.append(
            {
                "transaction_id": f"TXN-{700000 + idx}",
                "timestamp": created,
                "vendor_id": vendor,
                "vendor_name": vendor_names[v_ix],
                "department": department,
                "category": category,
                "payment_method": payment,
                "amount": round(amount, 2),
                "invoice_age": int(rng.integers(1, 66)),
                "approver": rng.choice(["A. Rao", "M. Kapoor", "D. Shah", "S. Iyer", "K. Mehta", "R. Sen"]),
                "risk_score": round(risk, 1),
                "fraud_flag": fraud,
                "late_hour": late,
                "weekend": weekend,
                "split_pattern": split,
            }
        )

    df = pd.DataFrame(records).sort_values("timestamp", ascending=False).reset_index(drop=True)

    features = df[["amount", "invoice_age", "risk_score"]].copy()
    features["upi"] = (df["payment_method"] == "UPI").astype(int)
    features["wire"] = (df["payment_method"] == "Wire").astype(int)
    features["late_hour"] = df["late_hour"].astype(int)
    features["weekend"] = df["weekend"].astype(int)
    features["split_pattern"] = df["split_pattern"].astype(int)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    iso = IsolationForest(n_estimators=80, contamination=0.08, random_state=seed)
    iso.fit(X)
    anomaly = -iso.score_samples(X)
    df["anomaly_score"] = np.round(MinMaxScaler(feature_range=(0, 100)).fit_transform(anomaly.reshape(-1, 1)).ravel(), 1)

    clf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=seed, class_weight="balanced")
    clf.fit(X, df["fraud_flag"])
    df["ai_confidence"] = np.round(clf.predict_proba(X)[:, 1] * 100, 1)
    df["risk_label"] = df["risk_score"].apply(risk_label)
    df["status"] = np.select(
        [df["risk_score"] >= 76, df["risk_score"] >= 58, df["risk_score"] >= 36],
        ["Blocked", "Review", "Watch",],
        default="Approved",
    )

    vendor_stats = (
        df.groupby(["vendor_id", "vendor_name"], as_index=False)
        .agg(
            total_spend=("amount", "sum"),
            txns=("transaction_id", "count"),
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            frauds=("fraud_flag", "sum"),
            anomaly=("anomaly_score", "mean"),
        )
        .merge(vendors, on=["vendor_id", "vendor_name"], how="left")
    )
    vendor_stats["trust_score"] = np.clip(
        vendor_stats["base_trust"] - vendor_stats["avg_risk"] * 0.34 - vendor_stats["frauds"] * 2.8 + vendor_stats["compliance_score"] * 0.18,
        3,
        99,
    ).round(1)
    vendor_stats["risk_tier"] = vendor_stats["avg_risk"].apply(risk_label)
    return df, vendor_stats


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {{
            --bg: {C["bg"]};
            --panel: {C["panel"]};
            --line: {C["line"]};
            --text: {C["text"]};
            --muted: {C["muted"]};
            --teal: {C["teal"]};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, rgba(18,214,197,0.16), transparent 34%),
                radial-gradient(circle at 84% 4%, rgba(87,166,255,0.12), transparent 31%),
                linear-gradient(135deg, #060c16 0%, #081324 52%, #0b1728 100%) !important;
            color: var(--text);
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        [data-testid="stSidebar"] {{
            min-width: 340px !important;
            max-width: 340px !important;
            width: 340px !important;
            background:
                linear-gradient(180deg, rgba(15,29,55,0.98), rgba(19,35,68,0.98) 58%, rgba(10,20,39,0.98)) !important;
            border-right: 1px solid rgba(100, 154, 193, 0.20);
            box-shadow: 24px 0 80px rgba(0,0,0,0.30);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding: 28px 18px 24px 18px;
        }}

        .main .block-container {{
            max-width: 1580px;
            padding: 34px 42px 56px 42px;
        }}

        ::-webkit-scrollbar {{ width: 7px; height: 7px; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(18,214,197,0.55); border-radius: 999px; }}
        ::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.04); }}

        .brand-wrap {{
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 18px 14px 24px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.09);
            margin-bottom: 18px;
        }}
        .brand-mark {{
            width: 52px;
            height: 52px;
            border-radius: 16px;
            display: grid;
            place-items: center;
            color: #eaffff;
            font-weight: 900;
            letter-spacing: -0.06em;
            background: linear-gradient(135deg, #13dfc5, #2875ff);
            box-shadow: 0 0 34px rgba(18,214,197,0.34), inset 0 1px 0 rgba(255,255,255,0.42);
        }}
        .brand-title {{ font-size: 19px; font-weight: 900; color: #f6fbff; line-height: 1.05; }}
        .brand-sub {{ font-size: 11px; font-weight: 800; color: #b6c7dc; letter-spacing: 0.18em; margin-top: 4px; }}

        .nav-section {{
            color: #d3e3f5;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 0.18em;
            margin: 20px 8px 9px 8px;
        }}
        .nav-item {{
            display: flex;
            align-items: center;
            gap: 13px;
            text-decoration: none !important;
            min-height: 50px;
            padding: 0 16px;
            border-radius: 13px;
            color: #c9d6e8 !important;
            margin: 5px 0;
            font-size: 15px;
            font-weight: 700;
            border: 1px solid transparent;
            transition: all 170ms ease;
        }}
        .nav-item:hover {{
            color: #ffffff !important;
            background: rgba(255,255,255,0.065);
            border-color: rgba(255,255,255,0.08);
        }}
        .nav-item.active {{
            color: #8dfff2 !important;
            background: linear-gradient(135deg, rgba(18,214,197,0.19), rgba(87,166,255,0.12));
            border-color: rgba(18,214,197,0.26);
            box-shadow: 0 0 22px rgba(18,214,197,0.16), inset 0 1px 0 rgba(255,255,255,0.08);
        }}
        .nav-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #8091ab;
            box-shadow: 0 0 0 4px rgba(128,145,171,0.08);
        }}
        .active .nav-dot {{
            background: var(--teal);
            box-shadow: 0 0 18px rgba(18,214,197,0.85), 0 0 0 5px rgba(18,214,197,0.10);
        }}
        .nav-code {{
            margin-left: auto;
            color: rgba(225,239,255,0.44);
            font-size: 11px;
            font-weight: 900;
        }}

        [data-testid="stSidebar"] .stButton {{
            margin: 5px 0 !important;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            justify-content: flex-start !important;
            min-height: 50px !important;
            width: 100% !important;
            border-radius: 13px !important;
            padding: 0 16px !important;
            font-size: 15px !important;
            font-weight: 800 !important;
            text-align: left !important;
            box-shadow: none !important;
            transition: all 170ms ease !important;
            background: rgba(255,255,255,0.025) !important;
            border: 1px solid transparent !important;
            color: #c9d6e8 !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            color: #ffffff !important;
            background: rgba(255,255,255,0.075) !important;
            border-color: rgba(255,255,255,0.10) !important;
            transform: translateY(-1px);
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            color: #8dfff2 !important;
            background: linear-gradient(135deg, rgba(18,214,197,0.19), rgba(87,166,255,0.12)) !important;
            border-color: rgba(18,214,197,0.30) !important;
            box-shadow: 0 0 22px rgba(18,214,197,0.16), inset 0 1px 0 rgba(255,255,255,0.08) !important;
        }}

        .portfolio-card, .side-action {{
            margin-top: 22px;
            padding: 20px;
            border-radius: 15px;
            background: rgba(255,255,255,0.055);
            border: 1px solid rgba(255,255,255,0.11);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.07);
        }}
        .portfolio-title {{
            color: #d8e6f9;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 0.13em;
            margin-bottom: 14px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            color: #dce9fb;
            padding: 9px 0;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            font-size: 14px;
        }}
        .stat-row:last-child {{ border-bottom: 0; }}
        .side-action {{
            background: linear-gradient(135deg, #10d9c2, #1f85ff);
            color: white;
            font-weight: 900;
            text-align: center;
            letter-spacing: 0.02em;
            box-shadow: 0 14px 36px rgba(18,214,197,0.25);
        }}

        .topbar {{
            display: grid;
            grid-template-columns: minmax(240px, 1fr) minmax(260px, 480px) auto auto;
            align-items: center;
            gap: 14px;
            padding: 14px 16px;
            margin-bottom: 30px;
            border-radius: 18px;
            background: rgba(17,31,55,0.68);
            border: 1px solid rgba(132,179,207,0.17);
            box-shadow: 0 18px 55px rgba(0,0,0,0.24), inset 0 1px 0 rgba(255,255,255,0.06);
            backdrop-filter: blur(20px);
        }}
        .page-kicker {{
            color: var(--teal);
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 0.17em;
        }}
        .page-title {{
            color: #f4faff;
            font-size: 25px;
            font-weight: 900;
            line-height: 1.1;
            margin-top: 3px;
        }}
        .search-pill, .status-pill, .time-pill {{
            min-height: 44px;
            display: flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid rgba(132,179,207,0.16);
            background: rgba(5,12,25,0.42);
            color: #c9d9ec;
            padding: 0 16px;
            font-weight: 700;
            font-size: 13px;
            white-space: nowrap;
        }}
        .search-pill {{ color: #8197b2; }}
        .status-light {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--teal);
            box-shadow: 0 0 18px rgba(18,214,197,0.82);
            margin-right: 9px;
        }}

        .section-title {{
            color: #eff7ff;
            font-size: 19px;
            font-weight: 900;
            letter-spacing: -0.01em;
            margin: 4px 0 6px 0;
        }}
        .section-subtitle {{
            color: #8fa4bd;
            font-size: 13px;
            line-height: 1.48;
            font-weight: 650;
            margin: 0 0 20px 0;
        }}
        .micro-label {{
            color: #8ca2bc;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}
        .panel {{
            background: var(--panel);
            border: 1px solid rgba(132,179,207,0.18);
            border-radius: 22px;
            padding: 26px;
            box-shadow: 0 22px 64px rgba(0,0,0,0.23), inset 0 1px 0 rgba(255,255,255,0.055);
            backdrop-filter: blur(22px);
            margin-bottom: 26px;
        }}
        .panel.anomaly {{
            padding: 30px;
            background:
                radial-gradient(circle at 10% 0%, rgba(18,214,197,0.18), transparent 34%),
                radial-gradient(circle at 88% 20%, rgba(139,124,255,0.15), transparent 32%),
                rgba(17,31,55,0.88);
            border-color: rgba(18,214,197,0.24);
            box-shadow: 0 28px 86px rgba(0,0,0,0.30), 0 0 38px rgba(18,214,197,0.08), inset 0 1px 0 rgba(255,255,255,0.07);
        }}
        .hero {{
            overflow: hidden;
            position: relative;
            min-height: 312px;
            background:
                radial-gradient(circle at 18% 0%, rgba(18,214,197,0.25), transparent 32%),
                radial-gradient(circle at 100% 0%, rgba(139,124,255,0.18), transparent 38%),
                linear-gradient(145deg, rgba(15,29,55,0.95), rgba(8,18,35,0.91));
        }}
        .hero:after {{
            content: "";
            position: absolute;
            inset: auto -80px -100px auto;
            width: 330px;
            height: 330px;
            background: radial-gradient(circle, rgba(18,214,197,0.22), transparent 64%);
            filter: blur(2px);
        }}
        .hero-grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(220px, 0.95fr) minmax(260px, 1.05fr);
            gap: 18px;
            align-items: center;
        }}
        .hero-score {{
            font-size: 82px;
            font-weight: 950;
            letter-spacing: -0.07em;
            color: #f6fbff;
            line-height: 0.95;
            margin: 13px 0 7px 0;
        }}
        .hero-sub {{
            color: #a9bbd2;
            font-size: 14px;
            font-weight: 650;
            line-height: 1.55;
            max-width: 500px;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 11px;
            font-size: 12px;
            font-weight: 900;
            background: rgba(18,214,197,0.11);
            border: 1px solid rgba(18,214,197,0.24);
            color: #9dfff4;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }}
        .kpi {{
            position: relative;
            min-height: 126px;
            padding: 19px;
            border-radius: 18px;
            background: rgba(17,31,55,0.76);
            border: 1px solid rgba(132,179,207,0.17);
            box-shadow: 0 16px 44px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.05);
            overflow: hidden;
        }}
        .kpi:after {{
            content: "";
            position: absolute;
            right: -38px;
            top: -42px;
            width: 118px;
            height: 118px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(18,214,197,0.18), transparent 62%);
        }}
        .kpi-label {{
            color: #8fa4bd;
            font-size: 12px;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}
        .kpi-value {{
            color: #f5fbff;
            font-size: 30px;
            font-weight: 920;
            margin-top: 13px;
            letter-spacing: -0.04em;
        }}
        .kpi-delta {{
            color: #aebed2;
            font-size: 12px;
            font-weight: 750;
            margin-top: 8px;
        }}

        .metric-row {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 10px;
            padding: 13px 0;
            border-bottom: 1px solid rgba(132,179,207,0.11);
            color: #dce9f8;
            font-size: 14px;
            font-weight: 700;
        }}
        .metric-row:last-child {{ border-bottom: 0; }}
        .metric-row span:last-child {{ color: #f8fcff; font-weight: 900; }}

        .alert {{
            padding: 16px;
            border-radius: 15px;
            margin-bottom: 13px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(132,179,207,0.15);
        }}
        .alert-head {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: #f3f9ff;
            font-weight: 900;
            font-size: 13px;
        }}
        .alert-copy {{
            color: #9eafc5;
            font-size: 12px;
            line-height: 1.45;
            margin-top: 5px;
        }}

        .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
            border-radius: 13px !important;
            border: 1px solid rgba(132,179,207,0.18) !important;
            background: rgba(5,12,25,0.62) !important;
            color: #eff7ff !important;
            min-height: 44px;
        }}
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {{
            color: #c7d7ea !important;
            font-weight: 800 !important;
        }}
        .stButton > button {{
            min-height: 46px;
            border-radius: 14px !important;
            border: 1px solid rgba(18,214,197,0.30) !important;
            color: #efffff !important;
            background: linear-gradient(135deg, rgba(18,214,197,0.88), rgba(31,133,255,0.92)) !important;
            font-weight: 900 !important;
            box-shadow: 0 15px 32px rgba(18,214,197,0.16);
        }}
        .stButton > button:hover {{
            border-color: rgba(18,214,197,0.70) !important;
            box-shadow: 0 18px 42px rgba(18,214,197,0.24);
            transform: translateY(-1px);
        }}

        [data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(132,179,207,0.16);
        }}

        .briefing-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin-top: 4px;
        }}
        .briefing-card {{
            padding: 18px;
            border-radius: 17px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(132,179,207,0.14);
        }}
        .briefing-title {{
            color: #eaf5ff;
            font-size: 13px;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .briefing-copy {{
            color: #a9bad0;
            font-size: 13px;
            line-height: 1.55;
            font-weight: 650;
        }}

        @media (max-width: 1100px) {{
            [data-testid="stSidebar"] {{
                min-width: 286px !important;
                max-width: 286px !important;
                width: 286px !important;
            }}
            .topbar {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .hero-grid {{ grid-template-columns: 1fr; }}
            .briefing-grid {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 760px) {{
            .main .block-container {{ padding: 18px 16px 36px 16px; }}
            .kpi-grid {{ grid-template-columns: 1fr; }}
            .hero-score {{ font-size: 60px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_active_page() -> str:
    st.session_state.setdefault("active_page", "Dashboard")
    page = st.session_state.active_page
    valid = {name for _, name, _ in PAGES}
    if page not in valid:
        st.session_state.active_page = "Dashboard"
        return "Dashboard"
    return page


def score_transaction(txn: pd.Series | Dict[str, object], vendor: pd.Series | Dict[str, object]) -> Dict[str, object]:
    amount = float(txn["amount"])
    payment = str(txn["payment_method"])
    late_hour = bool(txn.get("late_hour", datetime.now().hour < 7 or datetime.now().hour > 21))
    weekend = bool(txn.get("weekend", datetime.now().weekday() >= 5))
    split_pattern = bool(txn.get("split_pattern", 390000 <= amount <= 500000))
    score = 14
    score += max(0, (amount - 155000) / 12000)
    score += 17 if payment in ["UPI", "Wire"] else 0
    score += 13 if late_hour else 0
    score += 8 if weekend else 0
    score += 17 if split_pattern else 0
    score += (100 - float(vendor["trust_score"])) * 0.5
    score += float(vendor["bank_changes"]) * 4
    risk = round(float(np.clip(score, 3, 99)), 1)
    anomaly = round(float(np.clip(score + (amount / 100000), 5, 99)), 1)
    confidence = round(float(np.clip(score * 0.92 + 7, 8, 99)), 1)
    return {
        "risk_score": risk,
        "anomaly_score": anomaly,
        "ai_confidence": confidence,
        "risk_label": risk_label(risk),
        "status": "Blocked" if risk >= 76 else "Review" if risk >= 58 else "Watch" if risk >= 36 else "Approved",
        "late_hour": late_hour,
        "weekend": weekend,
        "split_pattern": split_pattern,
        "fraud_flag": int(risk >= 76),
    }


def set_global_state(df: pd.DataFrame, vendors: pd.DataFrame, txn: pd.Series | Dict[str, object]) -> None:
    vendor = vendors.loc[vendors["vendor_id"] == txn["vendor_id"]].iloc[0]
    txn_dict = dict(txn)
    txn_dict.update(score_transaction(txn_dict, vendor))
    txn_dict["vendor_id"] = str(txn_dict["vendor_id"])
    txn_dict["vendor_name"] = str(vendor["vendor_name"])
    st.session_state.current_transaction = txn_dict
    st.session_state.current_vendor = vendor.to_dict()
    st.session_state.current_scores = {
        "risk_score": float(txn_dict["risk_score"]),
        "ai_confidence": float(txn_dict["ai_confidence"]),
        "anomaly_score": float(txn_dict["anomaly_score"]),
        "trust_score": float(vendor["trust_score"]),
        "compliance_score": float(vendor["compliance_score"]),
    }
    st.session_state.current_alerts = build_alerts(txn_dict, vendor)
    st.session_state.current_filters = {
        "vendor_id": str(txn_dict["vendor_id"]),
        "amount": float(txn_dict["amount"]),
        "department": str(txn_dict["department"]),
        "category": str(txn_dict["category"]),
        "payment_method": str(txn_dict["payment_method"]),
        "risk_label": str(txn_dict["risk_label"]),
        "status": str(txn_dict["status"]),
    }


def sync_dashboard_widget_state(force: bool = False) -> None:
    txn = st.session_state.current_transaction
    defaults = {
        "dash_vendor_id": str(txn["vendor_id"]),
        "dash_amount": float(txn["amount"]),
        "dash_payment": str(txn["payment_method"]),
        "dash_department": str(txn["department"]),
        "dash_category": str(txn["category"]),
    }
    for key, value in defaults.items():
        if force or key not in st.session_state:
            st.session_state[key] = value


def initialize_state(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    st.session_state.setdefault("active_page", "Dashboard")
    if "current_transaction" not in st.session_state:
        set_global_state(df, vendors, df.sort_values("timestamp", ascending=False).iloc[0])
    for key, default in {
        "feed_search": "",
        "payment_override": None,
        "amount_override": None,
        "current_filters": {},
    }.items():
        st.session_state.setdefault(key, default)
    sync_dashboard_widget_state(force=False)


def normalize_context_from_widgets(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    """Let Dashboard widget state override stale initialized context on every rerun."""
    if "dash_vendor_id" not in st.session_state:
        return
    vendor_id = str(st.session_state.dash_vendor_id)
    if vendor_id not in set(vendors["vendor_id"]):
        return
    sync_master_context(
        df,
        vendors,
        vendor_id=vendor_id,
        amount=float(st.session_state.get("dash_amount", st.session_state.current_transaction["amount"])),
        payment=str(st.session_state.get("dash_payment", st.session_state.current_transaction["payment_method"])),
        department=str(st.session_state.get("dash_department", st.session_state.current_transaction["department"])),
        category=str(st.session_state.get("dash_category", st.session_state.current_transaction["category"])),
    )


def ensure_context_integrity(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    filters = st.session_state.current_filters
    txn_vendor = str(txn["vendor_id"])
    broken = (
        str(vendor.get("vendor_id")) != txn_vendor
        or str(filters.get("vendor_id")) != txn_vendor
        or str(filters.get("payment_method")) != str(txn["payment_method"])
        or str(filters.get("department")) != str(txn["department"])
        or str(filters.get("category")) != str(txn["category"])
    )
    if broken:
        set_global_state(df, vendors, txn)


def build_alerts(txn: pd.Series | Dict[str, object], vendor: pd.Series | Dict[str, object]) -> List[Dict[str, str]]:
    risk = float(txn["risk_score"])
    amount = float(txn["amount"])
    alerts: List[Dict[str, str]] = []
    if risk >= 76:
        alerts.append({"severity": "Critical", "title": "Auto-hold recommended", "copy": "The transaction exceeds the critical risk boundary and should move to manual approval."})
    elif risk >= 58:
        alerts.append({"severity": "High", "title": "Enhanced review required", "copy": "The model detected enough weak signals to require procurement and finance review."})
    if str(txn["payment_method"]) in ["UPI", "Wire"]:
        alerts.append({"severity": "High", "title": "Payment rail pressure", "copy": f"{txn['payment_method']} has elevated fraud correlation for this vendor cohort."})
    if bool(txn.get("split_pattern", False)) or 390000 <= amount <= 500000:
        alerts.append({"severity": "High", "title": "Split invoice signature", "copy": "Amount pattern sits near a common approval threshold and resembles split billing."})
    if float(vendor["trust_score"]) < 45:
        alerts.append({"severity": "Critical", "title": "Vendor trust deterioration", "copy": "Vendor trust score has fallen below the acceptable operating band."})
    if not alerts:
        alerts.append({"severity": "Clear", "title": "No severe exception", "copy": "Monitoring remains active with no critical procurement anomaly for this selection."})
    return alerts[:4]


def contextual_transactions(df: pd.DataFrame, include_current: bool = True) -> pd.DataFrame:
    txn = st.session_state.current_transaction
    current = pd.DataFrame([txn]) if include_current else pd.DataFrame()
    related = df[df["vendor_id"] == txn["vendor_id"]].copy()
    if include_current:
        related = related[related["transaction_id"] != txn["transaction_id"]]
        related = pd.concat([current, related], ignore_index=True)
    related["timestamp"] = pd.to_datetime(related["timestamp"])
    return related.sort_values("timestamp", ascending=False).reset_index(drop=True)


def contextual_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    filters = st.session_state.current_filters
    vendor_id = filters.get("vendor_id")
    department = filters.get("department")
    category = filters.get("category")
    payment = filters.get("payment_method")
    scoped = df[
        (df["vendor_id"] == vendor_id)
        | (df["department"] == department)
        | (df["category"] == category)
        | (df["payment_method"] == payment)
    ].copy()
    current = pd.DataFrame([st.session_state.current_transaction])
    scoped = scoped[scoped["transaction_id"] != st.session_state.current_transaction["transaction_id"]]
    scoped = pd.concat([current, scoped], ignore_index=True)
    scoped["timestamp"] = pd.to_datetime(scoped["timestamp"])
    return scoped.sort_values("timestamp", ascending=False).reset_index(drop=True)


def sync_master_context(
    df: pd.DataFrame,
    vendors: pd.DataFrame,
    vendor_id: str,
    amount: float,
    payment: str,
    department: str,
    category: str,
) -> bool:
    txn = st.session_state.current_transaction
    changed = (
        str(txn["vendor_id"]) != vendor_id
        or abs(float(txn["amount"]) - float(amount)) > 0.01
        or str(txn["payment_method"]) != payment
        or str(txn["department"]) != department
        or str(txn["category"]) != category
    )
    if not changed:
        return False

    base = df[df["vendor_id"] == vendor_id].sort_values("timestamp", ascending=False).head(1)
    new_txn = base.iloc[0].copy() if len(base) else df.iloc[0].copy()
    vendor = vendors.loc[vendors["vendor_id"] == vendor_id].iloc[0]
    new_txn["transaction_id"] = f"SIM-{datetime.now().strftime('%H%M%S')}"
    new_txn["timestamp"] = datetime.now()
    new_txn["amount"] = float(amount)
    new_txn["payment_method"] = payment
    new_txn["department"] = department
    new_txn["category"] = category
    new_txn["vendor_id"] = vendor_id
    new_txn["vendor_name"] = vendor["vendor_name"]
    new_txn["late_hour"] = datetime.now().hour < 7 or datetime.now().hour > 21
    new_txn["weekend"] = datetime.now().weekday() >= 5
    new_txn["split_pattern"] = 390000 <= float(amount) <= 500000
    set_global_state(df, vendors, new_txn)
    return True


def render_sidebar(df: pd.DataFrame) -> str:
    page = get_active_page()
    frauds = int(df["fraud_flag"].sum())
    review = int((df["status"] == "Review").sum())
    rate = frauds / len(df) * 100
    render_html(
        f"""
    <div class="brand-wrap">
        <div class="brand-mark">PS</div>
        <div>
            <div class="brand-title">{APP_NAME}</div>
            <div class="brand-sub">{APP_SUBTITLE}</div>
        </div>
    </div>
    """,
        st.sidebar,
    )
    active_section = None
    for section, name, code in PAGES:
        if section != active_section:
            render_html(f'<div class="nav-section">{section}</div>', st.sidebar)
            active_section = section
        active = name == page
        label = f"{'●' if active else '•'}  {name}    {code}"
        if st.sidebar.button(label, key=f"nav_{name}", type="primary" if active else "secondary", use_container_width=True):
            st.session_state.active_page = name

    render_html(
        f"""
    <div class="portfolio-card">
        <div class="portfolio-title">PORTFOLIO STATS</div>
        <div class="stat-row"><span>Total TXNs</span><strong>{len(df):,}</strong></div>
        <div class="stat-row"><span>Fraud Flagged</span><strong>{frauds:,}</strong></div>
        <div class="stat-row"><span>Under Review</span><strong>{review:,}</strong></div>
        <div class="stat-row"><span>Fraud Rate</span><strong>{rate:.1f}%</strong></div>
    </div>
    <div class="side-action">Refresh Intelligence</div>
    """,
        st.sidebar,
    )
    return get_active_page()


def render_topbar(page: str) -> None:
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    txn = st.session_state.current_transaction
    render_html(
        f"""
        <div class="topbar">
            <div>
                <div class="page-kicker">PROCUREMENT COMMAND SYSTEM</div>
                <div class="page-title">{page}</div>
            </div>
            <div class="search-pill">Search vendors, invoices, anomalies...</div>
            <div class="status-pill"><span class="status-light"></span>AI monitoring live</div>
            <div class="time-pill">{now} IST | {txn["transaction_id"]}</div>
        </div>
        """
    )


def panel_open(title: str | None = None, subtitle: str | None = None, extra: str = "") -> None:
    title_html = f'<div class="section-title">{title}</div>' if title else ""
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    render_html(f'<div class="panel {extra}">{title_html}{subtitle_html}')


def panel_close() -> None:
    render_html("</div>")


def kpi_grid(items: List[Tuple[str, str, str]]) -> None:
    html = '<div class="kpi-grid">'
    for label, value, delta in items:
        html += f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta}</div>
        </div>
        """
    html += "</div>"
    render_html(html)


def plot_theme(fig: go.Figure, height: int = 380, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    title_text = None
    if title:
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><span style='font-size:12px;color:#8fa4bd'>{subtitle}</span>"
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c7d7ea", family="Inter", size=13),
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=18, color="#eef7ff")) if title_text else None,
        margin=dict(l=52, r=28, t=76 if title_text else 34, b=54),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#aebfd3"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor="#0b1728", bordercolor="rgba(18,214,197,0.35)", font=dict(color="#eef7ff", family="Inter", size=12)),
        xaxis=dict(
            gridcolor="rgba(132,179,207,0.10)",
            zerolinecolor="rgba(132,179,207,0.10)",
            linecolor="rgba(132,179,207,0.22)",
            tickfont=dict(size=12, color="#9fb1c7"),
            title_font=dict(size=13, color="#c6d8ec"),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="rgba(132,179,207,0.10)",
            zerolinecolor="rgba(132,179,207,0.10)",
            linecolor="rgba(132,179,207,0.22)",
            tickfont=dict(size=12, color="#9fb1c7"),
            title_font=dict(size=13, color="#c6d8ec"),
            automargin=True,
        ),
    )
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


def gauge(score: float, title: str = "AI Risk Score") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 42, "color": "#f6fbff"}},
            title={"text": title, "font": {"size": 15, "color": "#b9c9dc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.25)"},
                "bar": {"color": risk_color(score), "thickness": 0.25},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 36], "color": "rgba(67,227,159,0.20)"},
                    {"range": [36, 58], "color": "rgba(87,166,255,0.18)"},
                    {"range": [58, 76], "color": "rgba(255,200,87,0.22)"},
                    {"range": [76, 100], "color": "rgba(255,93,108,0.24)"},
                ],
            },
        )
    )
    return plot_theme(fig, 246)


def alerts_html(alerts: List[Dict[str, str]]) -> None:
    html = ""
    for alert in alerts:
        color = {"Critical": C["red"], "High": C["amber"], "Elevated": C["blue"], "Clear": C["green"]}.get(alert["severity"], C["blue"])
        html += f"""
        <div class="alert" style="border-left: 3px solid {color};">
            <div class="alert-head"><span>{alert["title"]}</span><span style="color:{color};">{alert["severity"]}</span></div>
            <div class="alert-copy">{alert["copy"]}</div>
        </div>
        """
    render_html(html)


def summary_briefing(title: str, scoped: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    scores = st.session_state.current_scores
    avg_risk = float(scoped["risk_score"].mean()) if len(scoped) else float(scores["risk_score"])
    spend = float(scoped["amount"].sum()) if len(scoped) else float(txn["amount"])
    top_dept = scoped.groupby("department")["amount"].sum().idxmax() if len(scoped) else str(txn["department"])
    top_payment = scoped.groupby("payment_method")["risk_score"].mean().idxmax() if len(scoped) else str(txn["payment_method"])
    risk = float(scores["risk_score"])
    trust = float(scores["trust_score"])
    anomaly = float(scores["anomaly_score"])
    confidence = float(scores["ai_confidence"])
    render_html(
        f"""
        <div class="briefing-grid">
            <div class="briefing-card">
                <div class="briefing-title">Executive AI Summary</div>
                <div class="briefing-copy">{title} is focused on {vendor["vendor_name"]} ({vendor["vendor_id"]}). Current transaction risk is {risk:.1f}/100 with {confidence:.1f}% AI confidence across {len(scoped):,} context-linked records.</div>
            </div>
            <div class="briefing-card">
                <div class="briefing-title">Fraud Explanation</div>
                <div class="briefing-copy">The active score is driven by {txn["payment_method"]} payment behavior, {money(float(txn["amount"]))} exposure, vendor trust at {trust:.1f}/100, and threshold-sensitive invoice patterns.</div>
            </div>
            <div class="briefing-card">
                <div class="briefing-title">Vendor Trust Interpretation</div>
                <div class="briefing-copy">Trust is currently {risk_label(100 - trust)} pressure: compliance is {float(vendor["compliance_score"]):.0f}/100, bank changes are {int(vendor["bank_changes"])}, and historical flags total {int(vendor["frauds"])}.</div>
            </div>
            <div class="briefing-card">
                <div class="briefing-title">Procurement Risk Summary</div>
                <div class="briefing-copy">Context-linked spend is {money(spend)} with average contextual risk of {avg_risk:.1f}/100. Highest exposed department is {top_dept}; highest risk payment behavior is {top_payment}.</div>
            </div>
            <div class="briefing-card">
                <div class="briefing-title">Anomaly Explanation</div>
                <div class="briefing-copy">Anomaly intensity is {anomaly:.1f}/100, combining amount outlier pressure, payment rail risk, timing signals, and the vendor's behavioral baseline.</div>
            </div>
            <div class="briefing-card">
                <div class="briefing-title">Department Exposure Summary</div>
                <div class="briefing-copy">{txn["department"]} is the active department context. Related charts, tables, alerts, and monitoring panels are filtered or prioritized around this selection.</div>
            </div>
        </div>
        """
    )


def master_transaction_form(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    panel_open("Live Transaction Analysis", "This is the master context. Changing any field synchronizes every page, KPI, alert, heatmap, and summary.")
    vendor_ids = list(vendors.sort_values("vendor_id")["vendor_id"])
    selected_vendor = st.selectbox(
        "Vendor",
        vendor_ids,
        index=vendor_ids.index(str(txn["vendor_id"])) if str(txn["vendor_id"]) in vendor_ids else 0,
        format_func=lambda x: f"{x} | {vendors.loc[vendors['vendor_id'] == x, 'vendor_name'].iloc[0]}",
        key="dash_vendor_id",
    )
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount", min_value=1000.0, max_value=5000000.0, value=float(txn["amount"]), step=10000.0, key="dash_amount")
    with c2:
        payment = st.selectbox("Payment", PAYMENTS, index=PAYMENTS.index(str(txn["payment_method"])) if str(txn["payment_method"]) in PAYMENTS else 0, key="dash_payment")
    c3, c4 = st.columns(2)
    with c3:
        department = st.selectbox("Department", DEPARTMENTS, index=DEPARTMENTS.index(str(txn["department"])) if str(txn["department"]) in DEPARTMENTS else 0, key="dash_department")
    with c4:
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(str(txn["category"])) if str(txn["category"]) in CATEGORIES else 0, key="dash_category")

    auto_synced = sync_master_context(df, vendors, selected_vendor, amount, payment, department, category)
    if auto_synced:
        st.rerun()

    if st.button("Analyze Transaction", use_container_width=True):
        sync_master_context(df, vendors, selected_vendor, amount, payment, department, category)
        st.rerun()
    panel_close()


def dashboard_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    scores = st.session_state.current_scores
    alerts = st.session_state.current_alerts
    scoped = contextual_portfolio(df)
    vendor_txns = contextual_transactions(df)

    kpi_grid(
        [
            ("Selected Vendor", str(txn["vendor_id"]), str(vendor["vendor_name"])),
            ("Transaction Amount", money(float(txn["amount"])), f'{txn["payment_method"]} rail monitored'),
            ("Vendor Trust", f'{scores["trust_score"]:.0f}/100', f'Compliance {scores["compliance_score"]:.0f}/100'),
            ("AI Confidence", pct(scores["ai_confidence"]), "Synchronized across all pages"),
        ]
    )

    left, right = st.columns([1.56, 1], gap="large")
    with left:
        panel_open(extra="hero")
        render_html(
            f"""
            <div class="hero-grid">
                <div>
                    <div class="badge">LIVE MASTER CONTROL</div>
                    <div class="hero-score" style="color:{risk_color(scores["risk_score"])};">{scores["risk_score"]:.0f}</div>
                    <div class="micro-label">{risk_label(scores["risk_score"])} PROCUREMENT RISK</div>
                    <div class="hero-sub">
                        Current analysis is locked to {txn["transaction_id"]} for {vendor["vendor_name"]}.
                        Every page uses this same vendor, amount, payment rail, model score, and alert set.
                    </div>
                </div>
                <div>
            """
        )
        st.plotly_chart(gauge(scores["risk_score"]), use_container_width=True, config={"displayModeBar": False})
        render_html("</div></div>")
        panel_close()

    with right:
        master_transaction_form(df, vendors)

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        panel_open("Fraud Trend", "Risk movement for records connected to the active vendor, department, category, or payment rail.")
        trend = scoped.set_index("timestamp").resample("W")["risk_score"].mean().reset_index()
        fig = px.area(trend, x="timestamp", y="risk_score", color_discrete_sequence=[C["teal"]])
        fig.add_scatter(x=trend["timestamp"], y=trend["risk_score"], mode="lines", line=dict(color=C["blue"], width=2), name="risk")
        fig.update_xaxes(title_text="Week")
        fig.update_yaxes(title_text="Average risk score")
        st.plotly_chart(plot_theme(fig, 390, "Contextual Fraud Trend", f"{txn['vendor_id']} plus matching department/category/payment context"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with c2:
        panel_open("Recent Alerts", "Live alert set generated from the master transaction context.")
        alerts_html(alerts)
        panel_close()

    row1a, row1b = st.columns(2, gap="large")
    with row1a:
        panel_open("Payment Behavior", "Average risk by payment method inside the synchronized context.")
        pay = scoped.groupby("payment_method", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=True)
        fig = px.bar(pay, x="risk_score", y="payment_method", orientation="h", color="risk_score", color_continuous_scale=["#43e39f", "#ffc857", "#ff5d6c"])
        fig.update_xaxes(title_text="Average risk score")
        fig.update_yaxes(title_text="Payment method")
        st.plotly_chart(plot_theme(fig, 380, "Payment Rail Risk", f"Active rail: {txn['payment_method']}"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with row1b:
        panel_open("Department Exposure", "Spend and risk concentration for departments connected to the active context.")
        dept = scoped.groupby("department", as_index=False).agg(amount=("amount", "sum"), risk_score=("risk_score", "mean")).sort_values("amount", ascending=False)
        fig = px.bar(dept, x="department", y="amount", color="risk_score", color_continuous_scale=["#43e39f", "#ffc857", "#ff5d6c"])
        fig.update_xaxes(title_text="Department")
        fig.update_yaxes(title_text="Context-linked spend")
        st.plotly_chart(plot_theme(fig, 380, "Department Exposure", f"Active department: {txn['department']}"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    row2a, row2b = st.columns(2, gap="large")
    with row2a:
        panel_open("Vendor Trust Visualization", "Selected vendor benchmarked against the highest-risk vendor cohort.")
        top = pd.concat([
            vendors[vendors["vendor_id"] == txn["vendor_id"]],
            vendors.sort_values("avg_risk", ascending=False).head(14),
        ]).drop_duplicates("vendor_id")
        fig = px.scatter(top, x="trust_score", y="avg_risk", size="total_spend", color="risk_tier", hover_name="vendor_name", color_discrete_map={"Critical": C["red"], "High": C["amber"], "Elevated": C["blue"], "Clear": C["green"]})
        fig.add_scatter(x=[vendor["trust_score"]], y=[vendor["avg_risk"]], mode="markers+text", text=["Current"], textposition="top center", marker=dict(size=18, color=C["teal"], line=dict(width=2, color="#ffffff")), name="Selected vendor")
        fig.update_xaxes(title_text="Vendor trust score")
        fig.update_yaxes(title_text="Average risk score")
        st.plotly_chart(plot_theme(fig, 380, "Vendor Trust vs Risk", "Current vendor is highlighted against peer risk tier"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with row2b:
        panel_open("Transaction Scatter", "Amount-to-risk relationship for synchronized context records.")
        sample = scoped.sample(min(180, len(scoped)), random_state=4) if len(scoped) > 1 else scoped
        fig = px.scatter(sample, x="amount", y="risk_score", color="status", size="anomaly_score", color_discrete_map={"Blocked": C["red"], "Review": C["amber"], "Watch": C["blue"], "Approved": C["green"]})
        fig.update_xaxes(title_text="Transaction amount")
        fig.update_yaxes(title_text="Risk score")
        st.plotly_chart(plot_theme(fig, 380, "Transaction Risk Scatter", f"{len(sample):,} context-linked observations"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("AI Anomaly Detection", "Dedicated full-width anomaly view for the active vendor context. This section is intentionally spacious because it carries the core AI signal.", "anomaly")
    anomaly = vendor_txns.sort_values("timestamp").tail(80)
    fig = px.scatter(
        anomaly,
        x="timestamp",
        y="anomaly_score",
        size="amount",
        color="risk_score",
        hover_name="transaction_id",
        color_continuous_scale=["#43e39f", "#ffc857", "#ff5d6c"],
    )
    fig.add_hline(y=scores["anomaly_score"], line_dash="dash", line_color=C["teal"], annotation_text="Current anomaly score")
    fig.update_xaxes(title_text="Transaction timeline")
    fig.update_yaxes(title_text="Anomaly intensity")
    st.plotly_chart(plot_theme(fig, 520, "Vendor Anomaly Timeline", f"{txn['vendor_id']} | current score {scores['anomaly_score']:.1f}/100"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    panel_open("Executive Intelligence Briefing", "Summary report generated from the same synchronized vendor, transaction, score, alert, and filter state.")
    summary_briefing("Dashboard briefing", scoped)
    panel_close()

    panel_open("Live Activity Feed", "Current transaction is pinned above recent records from the selected vendor.")
    feed = vendor_txns.head(10)[["timestamp", "transaction_id", "vendor_id", "vendor_name", "amount", "payment_method", "risk_score", "status"]].copy()
    st.dataframe(feed, use_container_width=True, hide_index=True)
    panel_close()


def transaction_feed_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    panel_open("Searchable Live Feed", "The active vendor is pinned first; filters begin from the global master context.")
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        search = st.text_input("Search", value=st.session_state.feed_search, placeholder="Vendor, transaction, department")
        st.session_state.feed_search = search
    with c2:
        status = st.selectbox("Status", ["All", "Blocked", "Review", "Watch", "Approved"])
    with c3:
        default_payment = ["All"] + PAYMENTS
        payment = st.selectbox("Payment", default_payment, index=default_payment.index(str(txn["payment_method"])) if str(txn["payment_method"]) in default_payment else 0)
    with c4:
        min_risk = st.slider("Min Risk", 0, 100, 0)
    st.session_state.current_filters.update({"feed_status": status, "feed_payment": payment, "feed_min_risk": min_risk, "feed_search": search})
    current = pd.DataFrame([txn])
    filtered = pd.concat([current, df[df["transaction_id"] != txn["transaction_id"]]], ignore_index=True)
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    if search:
        needle = search.lower()
        filtered = filtered[
            filtered["vendor_name"].str.lower().str.contains(needle)
            | filtered["vendor_id"].str.lower().str.contains(needle)
            | filtered["transaction_id"].str.lower().str.contains(needle)
            | filtered["department"].str.lower().str.contains(needle)
        ]
    if status != "All":
        filtered = filtered[filtered["status"] == status]
    if payment != "All":
        filtered = filtered[filtered["payment_method"] == payment]
    filtered = filtered[filtered["risk_score"] >= min_risk]
    selected_related = filtered[filtered["vendor_id"] == txn["vendor_id"]].copy()
    filtered = pd.concat([selected_related, filtered[filtered["vendor_id"] != txn["vendor_id"]]]).head(120)
    view = filtered[["transaction_id", "timestamp", "vendor_id", "vendor_name", "department", "category", "payment_method", "amount", "risk_score", "ai_confidence", "status"]].copy()
    view["synced"] = np.where(view["vendor_id"] == txn["vendor_id"], "Current vendor", "")
    st.dataframe(view, use_container_width=True, hide_index=True)
    panel_close()

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        panel_open("Risk Tag Mix", "Status composition after current filters and vendor prioritization.")
        mix = filtered["status"].value_counts().rename_axis("status").reset_index(name="count")
        fig = px.pie(mix, names="status", values="count", hole=0.62, color="status", color_discrete_map={"Blocked": C["red"], "Review": C["amber"], "Watch": C["blue"], "Approved": C["green"]})
        st.plotly_chart(plot_theme(fig, 380, "Filtered Risk Tags", f"{len(filtered):,} visible records"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with c2:
        panel_open("Synchronized Selection", "This is the same master context used by all pages.")
        render_html(
            f"""
            <div class="metric-row"><span>Current TXN</span><span>{txn["transaction_id"]}</span></div>
            <div class="metric-row"><span>Vendor</span><span>{txn["vendor_id"]}</span></div>
            <div class="metric-row"><span>Amount</span><span>{money(float(txn["amount"]))}</span></div>
            <div class="metric-row"><span>Payment</span><span>{txn["payment_method"]}</span></div>
            <div class="metric-row"><span>Risk</span><span style="color:{risk_color(float(txn["risk_score"]))};">{txn["risk_score"]}/100</span></div>
            """
        )
        panel_close()

    panel_open("Transaction Feed Summary Report", "Briefing generated from filtered feed records and the active master context.")
    summary_briefing("Transaction feed briefing", filtered if len(filtered) else contextual_transactions(df))
    panel_close()


def vendor_intelligence_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    vdf = contextual_transactions(df).sort_values("timestamp")
    scores = st.session_state.current_scores

    kpi_grid(
        [
            ("Vendor Profile", str(vendor["vendor_id"]), str(vendor["country"])),
            ("Trust Score", f'{scores["trust_score"]:.0f}/100', f'{vendor["txns"]:.0f} transactions analyzed'),
            ("Total Exposure", money(float(vendor["total_spend"])), "Portfolio-linked spend"),
            ("Fraud History", f'{int(vendor["frauds"])} flags', f'Max risk {vendor["max_risk"]:.0f}/100'),
        ]
    )

    c1, c2 = st.columns([0.9, 1.1], gap="large")
    with c1:
        panel_open("Vendor Profile", "Live profile for the globally selected vendor.")
        render_html(
            f"""
            <div class="metric-row"><span>Name</span><span>{vendor["vendor_name"]}</span></div>
            <div class="metric-row"><span>Vendor ID</span><span>{vendor["vendor_id"]}</span></div>
            <div class="metric-row"><span>Onboarded</span><span>{pd.to_datetime(vendor["onboarded"]).strftime("%d %b %Y")}</span></div>
            <div class="metric-row"><span>Compliance</span><span>{vendor["compliance_score"]:.0f}/100</span></div>
            <div class="metric-row"><span>Bank Changes</span><span>{int(vendor["bank_changes"])}</span></div>
            <div class="metric-row"><span>Network Density</span><span>{float(vendor["network_density"]):.2f}</span></div>
            """
        )
        panel_close()
    with c2:
        panel_open("Behavioral Analysis", "Risk, anomaly, and confidence trend for this vendor only.")
        fig = px.line(vdf, x="timestamp", y=["risk_score", "anomaly_score", "ai_confidence"], color_discrete_sequence=[C["red"], C["amber"], C["teal"]])
        fig.update_xaxes(title_text="Transaction timeline")
        fig.update_yaxes(title_text="Score")
        st.plotly_chart(plot_theme(fig, 420, "Vendor Behavior Timeline", f"{txn['vendor_id']} synchronized intelligence"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    r1a, r1b = st.columns(2, gap="large")
    with r1a:
        panel_open("Department Exposure", "Spend concentration for the selected vendor.")
        dept = vdf.groupby("department", as_index=False)["amount"].sum().sort_values("amount")
        fig = px.bar(dept, x="amount", y="department", orientation="h", color_discrete_sequence=[C["blue"]])
        fig.update_xaxes(title_text="Vendor spend")
        fig.update_yaxes(title_text="Department")
        st.plotly_chart(plot_theme(fig, 380, "Department Exposure", f"Active department: {txn['department']}"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with r1b:
        panel_open("Payment Behavior", "Vendor payment behavior mapped against risk.")
        pay = vdf.groupby("payment_method", as_index=False).agg(amount=("amount", "sum"), risk=("risk_score", "mean"))
        fig = px.scatter(pay, x="amount", y="risk", size="amount", color="payment_method", color_discrete_sequence=[C["teal"], C["blue"], C["purple"], C["amber"], C["red"], C["green"]])
        fig.update_xaxes(title_text="Payment-method spend")
        fig.update_yaxes(title_text="Average risk")
        st.plotly_chart(plot_theme(fig, 380, "Payment Behavior", f"Active rail: {txn['payment_method']}"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    r2a, r2b = st.columns(2, gap="large")
    with r2a:
        panel_open("Compliance Heatmap", "Department and category risk concentration for the selected vendor.")
        heat = vdf.pivot_table(index="department", columns="category", values="risk_score", aggfunc="mean").fillna(0)
        fig = px.imshow(heat, color_continuous_scale=["#0e755f", "#ffc857", "#ff5d6c"], aspect="auto")
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Department")
        st.plotly_chart(plot_theme(fig, 380, "Compliance Heatmap", "Higher intensity indicates higher average fraud risk"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with r2b:
        panel_open("Fraud History", "Historical vendor exceptions by status.")
        history = vdf.groupby("status", as_index=False).agg(count=("transaction_id", "count"), avg_risk=("risk_score", "mean"))
        fig = px.bar(history, x="status", y="count", color="avg_risk", color_continuous_scale=["#43e39f", "#ffc857", "#ff5d6c"])
        fig.update_xaxes(title_text="Status")
        fig.update_yaxes(title_text="Transaction count")
        st.plotly_chart(plot_theme(fig, 380, "Fraud History", f"{int(vendor['frauds'])} historical fraud flags"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Vendor Anomaly Investigation", "Full-width AI anomaly view for this vendor's transaction timeline.", "anomaly")
    fig = px.area(vdf, x="timestamp", y="anomaly_score", color_discrete_sequence=[C["teal"]])
    fig.add_scatter(x=vdf["timestamp"], y=vdf["risk_score"], mode="lines", line=dict(color=C["red"], width=2), name="Risk score")
    fig.add_hline(y=scores["anomaly_score"], line_dash="dash", line_color=C["amber"], annotation_text="Current anomaly")
    fig.update_xaxes(title_text="Transaction timeline")
    fig.update_yaxes(title_text="AI score")
    st.plotly_chart(plot_theme(fig, 500, "Vendor Anomaly Detection", f"{txn['vendor_id']} anomaly and risk behavior"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    panel_open("Risk Prediction Panel", "Predictive control guidance based on the synchronized vendor context.")
    alerts_html(st.session_state.current_alerts)
    panel_close()

    panel_open("Vendor Intelligence Briefing", "Enterprise summary report for the selected vendor.")
    summary_briefing("Vendor intelligence briefing", vdf)
    panel_close()


def risk_reports_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    selected_vendor_df = contextual_transactions(df)
    scoped = contextual_portfolio(df)
    kpi_grid(
        [
            ("Portfolio Spend", money(float(df["amount"].sum())), "Rolling synthetic procurement lake"),
            ("Current Vendor Spend", money(float(selected_vendor_df["amount"].sum())), str(txn["vendor_id"])),
            ("High Risk Vendors", str(int((vendors["avg_risk"] >= 58).sum())), "Risk tier high or critical"),
            ("Compliance Drift", pct(float((100 - vendors["compliance_score"]).mean())), "Average control gap"),
        ]
    )
    r1a, r1b = st.columns(2, gap="large")
    with r1a:
        panel_open("Department Risk Heatmap", "Contextual heatmap filtered by active vendor, department, category, and payment rail.")
        heat = scoped.pivot_table(index="department", columns="category", values="risk_score", aggfunc="mean").fillna(0)
        fig = px.imshow(heat, color_continuous_scale=["#1e4f72", "#12d6c5", "#ffc857", "#ff5d6c"], aspect="auto")
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Department")
        st.plotly_chart(plot_theme(fig, 410, "Department Risk Heatmap", f"Active context: {txn['vendor_id']} / {txn['department']}"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with r1b:
        panel_open("Fraud Distribution", "Status distribution for the same synchronized context.")
        dist = scoped["status"].value_counts().rename_axis("status").reset_index(name="count")
        fig = px.pie(dist, names="status", values="count", hole=0.58, color="status", color_discrete_map={"Blocked": C["red"], "Review": C["amber"], "Watch": C["blue"], "Approved": C["green"]})
        st.plotly_chart(plot_theme(fig, 410, "Fraud Distribution", f"{len(scoped):,} context-linked records"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    r2a, r2b = st.columns(2, gap="large")
    with r2a:
        panel_open("Hourly Fraud Analysis", "Average risk by transaction hour inside the active context.")
        hourly = scoped.assign(hour=scoped["timestamp"].dt.hour).groupby("hour", as_index=False)["risk_score"].mean()
        fig = px.bar(hourly, x="hour", y="risk_score", color="risk_score", color_continuous_scale=["#43e39f", "#ffc857", "#ff5d6c"])
        fig.update_xaxes(title_text="Hour of day")
        fig.update_yaxes(title_text="Average risk")
        st.plotly_chart(plot_theme(fig, 380, "Hourly Fraud Analysis", "Late-hour spikes are easier to isolate"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with r2b:
        panel_open("Category Exposure", "Spend concentration by category for synchronized context.")
        cat = scoped.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        fig = px.bar(cat, x="category", y="amount", color_discrete_sequence=[C["teal"]])
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Spend")
        st.plotly_chart(plot_theme(fig, 380, "Category Exposure", f"Active category: {txn['category']}"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Predictive Anomaly Report", "Full-width AI anomaly section for risk reporting and compliance review.", "anomaly")
    trend = scoped.set_index("timestamp").resample("D").agg(risk_score=("risk_score", "mean"), anomaly_score=("anomaly_score", "mean"), amount=("amount", "sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend["timestamp"], y=trend["amount"], name="Spend", marker_color="rgba(87,166,255,0.35)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=trend["timestamp"], y=trend["risk_score"], mode="lines", name="Risk", line=dict(color=C["red"], width=3)))
    fig.add_trace(go.Scatter(x=trend["timestamp"], y=trend["anomaly_score"], mode="lines", name="Anomaly", line=dict(color=C["teal"], width=3)))
    fig.update_layout(yaxis2=dict(title="Spend", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#8fa4bd")))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="AI score")
    st.plotly_chart(plot_theme(fig, 520, "Predictive Risk and Spend Signal", "Daily anomaly, risk, and procurement spend in one executive view"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    panel_open("High-Risk Vendors and Predictive Alerts", "Selected vendor is included with the highest-risk peer vendors.")
    view = pd.concat([vendors[vendors["vendor_id"] == txn["vendor_id"]], vendors.sort_values("avg_risk", ascending=False).head(12)]).drop_duplicates("vendor_id")
    view = view[["vendor_id", "vendor_name", "total_spend", "txns", "avg_risk", "trust_score", "frauds", "risk_tier"]]
    st.dataframe(view, use_container_width=True, hide_index=True)
    panel_close()

    panel_open("Risk Reports Summary", "Enterprise intelligence briefing for risk and compliance leaders.")
    summary_briefing("Risk reports briefing", scoped)
    panel_close()


def ai_monitoring_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    scores = st.session_state.current_scores
    txn = st.session_state.current_transaction
    scoped = contextual_portfolio(df)
    vendor_txns = contextual_transactions(df)
    active = int((scoped["risk_score"] >= 58).sum())
    kpi_grid(
        [
            ("Model Confidence", pct(scores["ai_confidence"]), "Live selected transaction"),
            ("Anomaly Engine", pct(scores["anomaly_score"]), "Isolation forest signal"),
            ("Active Detections", f"{active:,}", "High and critical events"),
            ("System Health", "99.97%", "Realtime scan cluster"),
        ]
    )

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        panel_open("AI Health Indicators", "Operational model health for the synchronized procurement context.")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[92, 88, 96, 84, scores["ai_confidence"], 91], theta=["Inference", "Data Freshness", "Policy Rules", "Drift", "Confidence", "Latency"], fill="toself", line=dict(color=C["teal"])))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(132,179,207,0.14)")))
        st.plotly_chart(plot_theme(fig, 410, "AI Health Radar", "Confidence, latency, drift, and policy engine status"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with c2:
        panel_open("Alert Severity", "Alert mix generated from context-linked records.")
        severity = scoped.assign(severity=scoped["risk_score"].apply(risk_label)).groupby("severity", as_index=False).size()
        fig = px.bar(severity, x="severity", y="size", color="severity", color_discrete_map={"Critical": C["red"], "High": C["amber"], "Elevated": C["blue"], "Clear": C["green"]})
        fig.update_xaxes(title_text="Severity")
        fig.update_yaxes(title_text="Detection count")
        st.plotly_chart(plot_theme(fig, 410, "Alert Severity", f"{active:,} active detections"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Anomaly Engine Command View", "Full-width AI operations view for live scanning and anomaly intensity.", "anomaly")
    fig = px.scatter(
        vendor_txns.sort_values("timestamp").tail(100),
        x="timestamp",
        y="anomaly_score",
        size="amount",
        color="status",
        hover_name="transaction_id",
        color_discrete_map={"Blocked": C["red"], "Review": C["amber"], "Watch": C["blue"], "Approved": C["green"]},
    )
    fig.add_hline(y=scores["anomaly_score"], line_dash="dash", line_color=C["teal"], annotation_text="Current scan")
    fig.update_xaxes(title_text="Scan timeline")
    fig.update_yaxes(title_text="Anomaly intensity")
    st.plotly_chart(plot_theme(fig, 520, "Live Anomaly Scanning", f"{txn['vendor_id']} monitored with {scores['ai_confidence']:.1f}% confidence"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    c3, c4 = st.columns([1, 1], gap="large")
    with c3:
        panel_open("Live Scanning Feed", "Current transaction pinned to recent selected-vendor scans.")
        scan = vendor_txns.head(12)[["timestamp", "transaction_id", "vendor_id", "payment_method", "risk_score", "anomaly_score", "status"]].copy()
        st.dataframe(scan, use_container_width=True, hide_index=True)
        panel_close()
    with c4:
        panel_open("System Diagnostics", "AI system diagnostics for the active master context.")
        render_html(
            f"""
            <div class="metric-row"><span>Feature Store</span><span style="color:{C["green"]};">Online</span></div>
            <div class="metric-row"><span>Rules Engine</span><span style="color:{C["green"]};">Synced</span></div>
            <div class="metric-row"><span>Procurement Graph</span><span style="color:{C["green"]};">Stable</span></div>
            <div class="metric-row"><span>Current Vendor Lock</span><span>{txn["vendor_id"]}</span></div>
            <div class="metric-row"><span>Review Queue SLA</span><span style="color:{C["amber"]};">14 min</span></div>
            <div class="metric-row"><span>Policy Drift</span><span>2.8%</span></div>
            """
        )
        alerts_html(st.session_state.current_alerts)
        panel_close()

    panel_open("AI Monitoring Briefing", "Model explanation and operations summary for the selected vendor context.")
    summary_briefing("AI monitoring briefing", scoped)
    panel_close()


def audit_logs_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    events = [
        ("Transaction analyzed", txn["transaction_id"], "Model scored live transaction and synchronized session state."),
        ("Vendor graph refreshed", txn["vendor_id"], f'{vendor["vendor_name"]} relationship, spend, and compliance context loaded.'),
        ("Risk report updated", risk_label(float(txn["risk_score"])), "Heatmaps and predictive alerts now reflect the selected transaction context."),
        ("AI monitoring event", pct(float(txn["ai_confidence"])), "Confidence and anomaly diagnostics recalculated for active selection."),
        ("Control recommendation", str(txn["status"]), "Workflow policy mapped to current score and payment rail."),
    ]
    rows = []
    base = datetime.now()
    for i, (event, entity, detail) in enumerate(events):
        rows.append({"time": base - timedelta(minutes=i * 4), "event": event, "entity": entity, "detail": detail, "actor": "ProcureShield AI"})
    panel_open("Synchronized Audit Trail")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    panel_close()

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        panel_open("Related Vendor History")
        related = contextual_transactions(df).head(14)[["timestamp", "transaction_id", "department", "amount", "risk_score", "status"]]
        st.dataframe(related, use_container_width=True, hide_index=True)
        panel_close()
    with c2:
        panel_open("Current Alerts")
        alerts_html(st.session_state.current_alerts)
        panel_close()


def configuration_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    scores = st.session_state.current_scores
    panel_open("Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("Critical risk threshold", 60, 95, 76)
        st.slider("High risk threshold", 40, 80, 58)
    with c2:
        st.selectbox("Default payment scrutiny", ["Balanced", "Strict", "Maximum"], index=1)
        st.selectbox("Model profile", ["ProcureShield RF+IF v3", "Fast Review v2", "Conservative Audit v1"])
    with c3:
        st.selectbox("Notification channel", ["SOC Console", "Email + Teams", "Webhook"])
        st.slider("SLA minutes", 5, 90, 15)
    render_html(
        f"""
        <div class="metric-row"><span>Session vendor</span><span>{st.session_state.current_vendor["vendor_id"]}</span></div>
        <div class="metric-row"><span>Session risk score</span><span style="color:{risk_color(scores["risk_score"])};">{scores["risk_score"]:.1f}/100</span></div>
        <div class="metric-row"><span>Global alert count</span><span>{len(st.session_state.current_alerts)}</span></div>
        """
    )
    panel_close()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        panel_open("Rule Weights")
        rules = pd.DataFrame(
            {
                "control": ["Payment Rail", "Split Invoice", "Late Hour", "Vendor Trust", "Bank Change", "Amount Outlier"],
                "weight": [17, 17, 13, 20, 11, 22],
                "enabled": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            }
        )
        st.dataframe(rules, use_container_width=True, hide_index=True)
        panel_close()
    with c2:
        panel_open("Deployment Readiness")
        render_html(
            f"""
            <div class="metric-row"><span>Streamlit Cloud</span><span style="color:{C["green"]};">Ready</span></div>
            <div class="metric-row"><span>Replit Runtime</span><span style="color:{C["green"]};">Ready</span></div>
            <div class="metric-row"><span>Cache Strategy</span><span>@st.cache_data</span></div>
            <div class="metric-row"><span>Charts</span><span>Plotly optimized</span></div>
            <div class="metric-row"><span>State Model</span><span>Global session sync</span></div>
            """
        )
        panel_close()


def footer() -> None:
    render_html(
        """
        <div style="margin-top:22px; padding:18px 4px; color:#6f86a2; font-size:12px; font-weight:700;">
            ProcureShield AI | Connected procurement fraud intelligence | Demo data generated locally
        </div>
        """
    )


def main() -> None:
    inject_css()
    df, vendors = generate_procurement_data()
    initialize_state(df, vendors)
    normalize_context_from_widgets(df, vendors)
    ensure_context_integrity(df, vendors)
    page = render_sidebar(df)
    render_topbar(page)

    if page == "Dashboard":
        dashboard_page(df, vendors)
    elif page == "Transaction Feed":
        transaction_feed_page(df, vendors)
    elif page == "Vendor Intelligence":
        vendor_intelligence_page(df, vendors)
    elif page == "Risk Reports":
        risk_reports_page(df, vendors)
    elif page == "AI Monitoring Center":
        ai_monitoring_page(df, vendors)
    elif page == "Audit Logs":
        audit_logs_page(df, vendors)
    elif page == "Configuration":
        configuration_page(df, vendors)
    footer()


if __name__ == "__main__":
    main()


