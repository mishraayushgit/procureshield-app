"""
ProcureShield Procurement Governance
Streamlit application for procurement governance and vendor risk controls.

Run:
    streamlit run procureshield_ai.py
"""

from __future__ import annotations

import base64
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
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
    page_title="ProcureShield Procurement Governance",
    page_icon="PS",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_NAME = "ProcureShield"
APP_SUBTITLE = "PROCUREMENT OPERATIONS"
APP_DIR = Path(__file__).resolve().parent

C = {
    "bg": "#f5f7fa",
    "bg2": "#ffffff",
    "sidebar": "#ffffff",
    "sidebar2": "#f8fafc",
    "panel": "#ffffff",
    "panel2": "#ffffff",
    "line": "#d1d5db",
    "muted": "#6b7280",
    "text": "#111827",
    "soft": "#4b5563",
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
    ("ANALYTICS", "Risk & Compliance Monitoring", "RC"),
    ("SETTINGS", "Audit Logs", "AL"),
    ("SETTINGS", "Configuration", "CF"),
]

PAYMENTS = ["UPI", "NEFT", "RTGS", "Card", "Wire", "Cheque"]
DEPARTMENTS = ["IT", "Finance", "Operations", "Legal", "Facilities", "Marketing", "Supply Chain"]
CATEGORIES = ["Software", "Consulting", "Logistics", "Hardware", "Office", "Travel", "Security"]


def html_fragment(markup: str) -> str:
    """Keep Streamlit markdown from treating indented HTML as code blocks."""
    return "\n".join(line.strip() for line in dedent(markup).strip().splitlines() if line.strip())


def platform_logo_data_uri() -> str:
    """Use the supplied transparent logo when present, with a restrained fallback."""
    for filename in ("project_logo.png", "project_logo.svg", "logo.png", "logo.svg"):
        logo = APP_DIR / filename
        if logo.exists():
            mime = mimetypes.guess_type(logo.name)[0] or "image/png"
            payload = base64.b64encode(logo.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{payload}"
    fallback = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <path fill="#155f9d" d="M32 4 55 13v16c0 14-9 25-23 31C18 54 9 43 9 29V13z"/>
      <path fill="#fff" d="m20 31 8 8 17-18 4 4-21 22-12-12z"/>
    </svg>
    """
    return "data:image/svg+xml;base64," + base64.b64encode(fallback.encode("utf-8")).decode("ascii")


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
            background: #f5f7fa !important;
            color: var(--text);
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        [data-testid="stSidebar"] {{
            min-width: 340px !important;
            max-width: 340px !important;
            width: 340px !important;
            background: #ffffff !important;
            border-right: 1px solid rgba(100, 154, 193, 0.20);
            box-shadow: none;
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
        ::-webkit-scrollbar-track {{ background: #f3f4f6; }}

        .brand-wrap {{
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 18px 14px 24px 14px;
            border-bottom: 1px solid #d1d5db;
            margin-bottom: 18px;
        }}
        .brand-mark {{
            width: 52px;
            height: 52px;
            border-radius: 16px;
            display: grid;
            place-items: center;
            color: #111827;
            font-weight: 900;
            letter-spacing: -0.06em;
            background: linear-gradient(135deg, #13dfc5, #2875ff);
            box-shadow: none;
        }}
        .brand-title {{ font-size: 19px; font-weight: 900; color: #000000; line-height: 1.05; }}
        .brand-sub {{ font-size: 11px; font-weight: 800; color: #6b7280; letter-spacing: 0.18em; margin-top: 4px; }}

        .nav-section {{
            color: #6b7280;
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
            color: #374151 !important;
            margin: 5px 0;
            font-size: 15px;
            font-weight: 700;
            border: 1px solid transparent;
            transition: all 170ms ease;
        }}
        .nav-item:hover {{
            color: #111827 !important;
            background: #f3f4f6;
            border-color: #d1d5db;
        }}
        .nav-item.active {{
            color: #0f4c81 !important;
            background: linear-gradient(135deg, rgba(18,214,197,0.19), rgba(87,166,255,0.12));
            border-color: rgba(18,214,197,0.26);
            box-shadow: none;
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
            color: #6b7280;
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
            background: #ffffff !important;
            border: 1px solid transparent !important;
            color: #374151 !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            color: #111827 !important;
            background: #f3f4f6 !important;
            border-color: #d1d5db !important;
            transform: translateY(-1px);
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            color: #0f4c81 !important;
            background: linear-gradient(135deg, rgba(18,214,197,0.19), rgba(87,166,255,0.12)) !important;
            border-color: rgba(18,214,197,0.30) !important;
            box-shadow: none !important;
        }}

        .portfolio-card, .side-action {{
            margin-top: 22px;
            padding: 20px;
            border-radius: 15px;
            background: #ffffff;
            border: 1px solid #d1d5db;
            box-shadow: none;
        }}
        .portfolio-title {{
            color: #374151;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 0.13em;
            margin-bottom: 14px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            color: #374151;
            padding: 9px 0;
            border-bottom: 1px solid #d1d5db;
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
            background: #ffffff;
            border: 1px solid rgba(132,179,207,0.17);
            box-shadow: 0 1px 3px rgba(16,24,40,0.06);
            backdrop-filter: blur(20px);
        }}
        .page-kicker {{
            color: var(--teal);
            font-size: 11px;
            font-weight: 900;
            letter-spacing: 0.17em;
        }}
        .page-title {{
            color: #000000;
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
            background: #f8fafc;
            color: #4b5563;
            padding: 0 16px;
            font-weight: 700;
            font-size: 13px;
            white-space: nowrap;
        }}
        .search-pill {{ color: #4b5563; }}
        .status-light {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--teal);
            box-shadow: 0 0 18px rgba(18,214,197,0.82);
            margin-right: 9px;
        }}

        .section-title {{
            color: #000000;
            font-size: 19px;
            font-weight: 900;
            letter-spacing: -0.01em;
            margin: 4px 0 6px 0;
        }}
        .section-subtitle {{
            color: #4b5563;
            font-size: 13px;
            line-height: 1.48;
            font-weight: 650;
            margin: 0 0 20px 0;
        }}
        .micro-label {{
            color: #6b7280;
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
            box-shadow: 0 1px 3px rgba(16,24,40,0.06);
            backdrop-filter: blur(22px);
            margin-bottom: 26px;
        }}
        .panel.anomaly {{
            padding: 30px;
            background: #ffffff;
            border-color: rgba(18,214,197,0.24);
            box-shadow: 0 1px 3px rgba(16,24,40,0.06);
        }}
        .hero {{
            overflow: hidden;
            position: relative;
            min-height: 312px;
            background: #ffffff;
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
            color: #111827;
            line-height: 0.95;
            margin: 13px 0 7px 0;
        }}
        .hero-sub {{
            color: #4b5563;
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
            color: #0f4c81;
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
            background: #ffffff;
            border: 1px solid rgba(132,179,207,0.17);
            box-shadow: 0 1px 3px rgba(16,24,40,0.06);
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
            color: #374151;
            font-size: 12px;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}
        .kpi-value {{
            color: #111827;
            font-size: 30px;
            font-weight: 920;
            margin-top: 13px;
            letter-spacing: -0.04em;
        }}
        .kpi-delta {{
            color: #6b7280;
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
            color: #374151;
            font-size: 14px;
            font-weight: 700;
        }}
        .metric-row:last-child {{ border-bottom: 0; }}
        .metric-row span:last-child {{ color: #111827; font-weight: 800; }}

        .alert {{
            padding: 16px;
            border-radius: 15px;
            margin-bottom: 13px;
            background: #ffffff;
            border: 1px solid rgba(132,179,207,0.15);
        }}
        .alert-head {{
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: #111827;
            font-weight: 900;
            font-size: 13px;
        }}
        .alert-copy {{
            color: #4b5563;
            font-size: 12px;
            line-height: 1.45;
            margin-top: 5px;
        }}

        .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
            border-radius: 13px !important;
            border: 1px solid #d1d5db !important;
            background: #ffffff !important;
            color: #111827 !important;
            min-height: 44px;
        }}
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {{
            color: #374151 !important;
            font-weight: 800 !important;
        }}
        .stButton > button {{
            min-height: 46px;
            border-radius: 14px !important;
            border: 1px solid rgba(18,214,197,0.30) !important;
            color: #ffffff !important;
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
            background: #ffffff;
            border: 1px solid rgba(132,179,207,0.14);
        }}
        .briefing-title {{
            color: #000000;
            font-size: 13px;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .briefing-copy {{
            color: #4b5563;
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

        /* Enterprise procurement theme */
        :root {{
            --bg: #f5f7fa;
            --panel: #ffffff;
            --line: #d9e0e8;
            --text: #1f2937;
            --muted: #667085;
            --teal: #1769aa;
        }}
        .stApp {{ background: #f5f7fa !important; color: #1f2937; }}
        [data-testid="stSidebar"] {{
            background: #ffffff !important;
            border-right: 1px solid #d9e0e8;
            box-shadow: none;
        }}
        .brand-wrap {{ border-bottom-color: #e5e9ef; }}
        .brand-mark {{
            width:58px; height:58px; padding:7px; border-radius:10px;
            display:flex; align-items:center; justify-content:center;
            background:#ffffff; border:1px solid #d9e0e8; box-shadow:0 1px 3px rgba(16,24,40,.08);
        }}
        .brand-mark img {{ width:100%; height:100%; object-fit:contain; display:block; }}
        .brand-title, .section-title, .page-title {{ color: #172b4d; }}
        .brand-sub, .nav-section, .section-subtitle, .micro-label {{ color: #667085; }}
        [data-testid="stSidebar"] .stButton > button {{
            border-radius: 6px !important;
            color: #344054 !important;
            background: transparent !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            color: #145a91 !important;
            background: #eef6fd !important;
            border-color: #d2e6f6 !important;
            transform: none;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            color: #145a91 !important;
            background: #e8f3fc !important;
            border-color: #c8e1f4 !important;
            box-shadow: none !important;
        }}
        .portfolio-card {{
            border-radius: 6px;
            background: #f8fafc;
            border: 1px solid #e1e7ef;
            box-shadow: none;
        }}
        .portfolio-title, .stat-row {{ color: #475467; }}
        .stat-row {{ border-bottom-color: #e5e9ef; }}
        .side-action {{ display: none; }}
        .topbar {{
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid #d9e0e8;
            box-shadow: 0 1px 3px rgba(16,24,40,0.06);
            backdrop-filter: none;
        }}
        .page-kicker {{ color: #1769aa; }}
        .search-pill, .status-pill, .time-pill {{
            border-radius: 6px;
            border-color: #d9e0e8;
            background: #f8fafc;
            color: #667085;
        }}
        .status-light {{ background: #2e7d32; box-shadow: none; }}
        .panel {{
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid #d9e0e8;
            box-shadow: 0 1px 3px rgba(16,24,40,0.06);
            backdrop-filter: none;
            padding: 20px;
            margin-bottom: 18px;
        }}
        .kpi {{ border-radius: 8px; background: #ffffff; border-color: #d9e0e8; box-shadow: none; }}
        .kpi:after {{ display: none; }}
        .kpi-label, .kpi-delta {{ color: #667085; }}
        .kpi-value {{ color: #172b4d; }}
        .panel.anomaly {{ background: #ffffff; border-color: #d9e0e8; box-shadow: 0 1px 3px rgba(16,24,40,0.06); }}
        .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
            border-radius: 5px !important;
            border-color: #cfd8e3 !important;
            background: #ffffff !important;
            color: #1f2937 !important;
            min-height: 40px;
        }}
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {{
            color: #344054 !important;
            font-size: 13px !important;
            font-weight: 700 !important;
        }}
        .stDateInput input, .stTextInput input, .stNumberInput input {{
            background:#ffffff !important; color:#1f2937 !important; border-color:#cfd8e3 !important;
        }}
        .stSelectbox [data-baseweb="select"] > div {{
            background:#ffffff !important; color:#1f2937 !important; border-color:#cfd8e3 !important;
        }}
        .stSelectbox svg {{ fill:#667085 !important; }}
        .stNumberInput button {{ background:#f8fafc !important; color:#344054 !important; border-color:#cfd8e3 !important; }}
        .stButton > button {{
            border-radius: 5px !important;
            border: 1px solid #1769aa !important;
            color: #111827 !important;
            background: #1769aa !important;
            box-shadow: none;
        }}
        .stButton > button:hover {{ background: #145a91 !important; box-shadow: none; transform: none; }}
        .workflow-head {{
            display:flex; justify-content:space-between; align-items:center; gap:18px;
            padding:20px 22px; margin-bottom:18px; border:1px solid #d9e0e8;
            border-radius:8px; background:#ffffff; box-shadow:0 1px 3px rgba(16,24,40,.06);
        }}
        .workflow-title {{ color:#172b4d; font-size:22px; font-weight:800; }}
        .workflow-sub {{ color:#667085; font-size:13px; margin-top:5px; }}
        .workflow-meta {{ display:flex; gap:10px; align-items:center; color:#667085; font-size:12px; font-weight:700; }}
        .chip {{
            display:inline-flex; align-items:center; gap:6px; padding:4px 9px;
            border-radius:999px; font-size:11px; font-weight:800; border:1px solid transparent;
        }}
        .chip.pending {{ color:#9a6700; background:#fff8e1; border-color:#f4d88b; }}
        .chip.good {{ color:#256029; background:#edf7ed; border-color:#b7d9b9; }}
        .chip.warn {{ color:#9a3412; background:#fff3e8; border-color:#f3c49e; }}
        .chip.info {{ color:#145a91; background:#eef6fd; border-color:#c8e1f4; }}
        .card-title {{ color:#172b4d; font-size:14px; font-weight:800; margin-bottom:3px; }}
        .card-note {{ color:#667085; font-size:12px; line-height:1.45; margin-bottom:14px; }}
        .context-strip {{
            display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:10px;
            padding:12px; margin-bottom:14px; background:#f8fafc; border:1px solid #e1e7ef; border-radius:6px;
        }}
        .context-label {{ color:#667085; font-size:10px; font-weight:800; letter-spacing:.08em; }}
        .context-value {{ color:#1f2937; font-size:13px; font-weight:800; margin-top:3px; }}
        .check-row {{ display:flex; justify-content:space-between; gap:10px; padding:10px 0; border-bottom:1px solid #edf0f4; color:#475467; font-size:13px; }}
        .check-row:last-child {{ border-bottom:0; }}
        .check-row strong {{ color:#172b4d; }}
        .attachment-box {{ padding:16px; text-align:center; color:#667085; font-size:12px; border:1px dashed #aebdcd; border-radius:6px; background:#fbfcfd; }}
        .contract-link {{ color:#1769aa; font-weight:700; text-decoration:none; }}
        .sidebar-score {{ display:flex; align-items:baseline; gap:5px; padding:6px 0 14px; color:#172b4d; font-size:36px; font-weight:800; }}
        .sidebar-score span {{ color:#667085; font-size:13px; font-weight:700; }}
        .metric-row {{ color:#475467; border-bottom-color:#edf0f4; }}
        .metric-row span:last-child {{ color:#172b4d; }}
        .alert {{ background:#fbfcfd; border-color:#e1e7ef; border-radius:6px; }}
        .alert-head {{ color:#172b4d; }}
        .alert-copy {{ color:#667085; }}
        .briefing-card {{ background:#fbfcfd; border-color:#e1e7ef; border-radius:6px; }}
        .briefing-title {{ color:#172b4d; }}
        .briefing-copy {{ color:#475467; }}
        [data-testid="stDataFrame"] {{ border-radius:6px; border-color:#d9e0e8; }}
        [data-testid="stFileUploader"] section {{ background:#fbfcfd; border-color:#aebdcd; }}
        .executive-grid {{
            display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:10px; margin:0 0 18px;
        }}
        .executive-card {{
            min-height:84px; padding:13px 12px; border:1px solid #d9e0e8; border-radius:6px;
            background:#ffffff; box-shadow:0 1px 2px rgba(16,24,40,.04);
        }}
        .executive-label {{ color:#667085; font-size:10px; font-weight:800; letter-spacing:.07em; text-transform:uppercase; line-height:1.35; }}
        .executive-value {{ color:#172b4d; font-size:16px; font-weight:800; margin-top:10px; }}
        @media (max-width: 1250px) {{ .executive-grid {{ grid-template-columns:repeat(4,minmax(0,1fr)); }} }}
        @media (max-width: 760px) {{ .executive-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}

        /* WCAG AA high-contrast layer */
        :root {{
            --text: #111827;
            --heading: #000000;
            --label: #374151;
            --secondary: #4b5563;
            --metadata: #6b7280;
            --line: #d1d5db;
        }}
        .stApp, .main, .main p, .main li, .main span, .main div {{
            color: var(--text);
        }}
        .panel, .kpi, .workflow-head, .executive-card, .briefing-card, .portfolio-card,
        .context-strip, .attachment-box, .topbar {{
            border-color: var(--line);
        }}
        .brand-title, .page-title, .section-title, .workflow-title, .card-title,
        .briefing-title, .sidebar-score, .kpi-value, .executive-value {{
            color: var(--heading) !important;
        }}
        .section-subtitle, .workflow-sub, .card-note, .briefing-copy, .alert-copy,
        .attachment-box, .sidebar-score span {{
            color: var(--secondary) !important;
        }}
        .brand-sub, .nav-section, .micro-label, .kpi-delta, .context-label,
        .workflow-meta, .executive-label, .search-pill, .status-pill, .time-pill {{
            color: var(--metadata) !important;
        }}
        .kpi-label, .context-value, .check-row, .stat-row, .portfolio-title,
        .alert-head {{
            color: var(--label) !important;
        }}
        .metric-row {{
            color: var(--label) !important;
            border-bottom-color: var(--line) !important;
            font-weight:600 !important;
        }}
        .metric-row span:first-child {{ color:var(--label) !important; font-weight:600 !important; }}
        .metric-row span:last-child, .metric-row strong {{ color:var(--text) !important; font-weight:800 !important; }}
        .check-row span:first-child {{ color:var(--label) !important; font-weight:600 !important; }}
        .check-row strong {{ color:var(--text) !important; font-weight:800 !important; }}
        [data-testid="stSidebar"], [data-testid="stSidebar"] * {{
            color:var(--text);
        }}
        [data-testid="stSidebar"] .stButton > button {{
            color:var(--label) !important;
            font-weight:700 !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            color:#0f4c81 !important;
        }}
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label,
        .stSlider label, .stFileUploader label {{
            color:var(--label) !important;
            font-weight:700 !important;
        }}
        .stTextInput input, .stNumberInput input, .stDateInput input,
        .stSelectbox [data-baseweb="select"], .stSelectbox [data-baseweb="select"] *,
        [data-baseweb="popover"] *, [role="option"] {{
            color:var(--text) !important;
        }}
        [data-testid="stFileUploader"] *, [data-testid="stFileUploaderDropzone"] * {{
            color:var(--label) !important;
        }}
        [data-testid="stDataFrame"], [data-testid="stDataFrame"] * {{
            color:var(--text) !important;
            --gdg-text-dark: #111827;
            --gdg-text-medium: #374151;
            --gdg-text-light: #4b5563;
            --gdg-border-color: #d1d5db;
            --gdg-header-text: #111827;
            --gdg-header-background: #f3f4f6;
            --gdg-bg-cell: #ffffff;
            --gdg-bg-header: #f3f4f6;
        }}
        [data-testid="stDataFrame"] {{ border-color:var(--line) !important; }}
        [data-testid="stDataFrame"] button {{ color:var(--label) !important; }}

        /* White enterprise dropdowns, comboboxes, autocomplete and multiselect menus */
        .stSelectbox [data-baseweb="select"],
        .stMultiSelect [data-baseweb="select"],
        [role="combobox"] {{
            background:#ffffff !important; color:#111827 !important;
            border-color:#d1d5db !important; border-radius:6px !important;
            transition:border-color 150ms ease, box-shadow 150ms ease, background-color 150ms ease !important;
        }}
        .stSelectbox [data-baseweb="select"]:focus-within,
        .stMultiSelect [data-baseweb="select"]:focus-within,
        [role="combobox"]:focus-within {{
            border-color:#1769aa !important;
            box-shadow:0 0 0 3px rgba(23,105,170,.14) !important;
        }}
        .stSelectbox [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] *,
        [role="combobox"] * {{
            color:#111827 !important;
        }}
        [data-baseweb="popover"]:not([data-baseweb="tooltip"]),
        [data-baseweb="popover"]:not([data-baseweb="tooltip"]) > div,
        [data-baseweb="menu"],
        [role="listbox"],
        [role="menu"] {{
            background:#ffffff !important; color:#111827 !important;
            border-color:#d1d5db !important;
        }}
        [data-baseweb="popover"]:not([data-baseweb="tooltip"]) {{
            border:1px solid #d1d5db !important; border-radius:8px !important;
            box-shadow:0 10px 24px rgba(17,24,39,.12) !important;
            overflow:hidden !important;
        }}
        [role="option"], [data-baseweb="menu"] li, [role="menuitem"] {{
            background:#ffffff !important; color:#111827 !important;
            transition:background-color 140ms ease, color 140ms ease !important;
        }}
        [role="option"] *, [data-baseweb="menu"] li *, [role="menuitem"] * {{
            color:#111827 !important;
        }}
        [role="option"]:hover, [role="option"]:focus, [role="option"][aria-highlighted="true"],
        [role="option"][data-highlighted="true"],
        [data-baseweb="menu"] li:hover, [role="menuitem"]:hover {{
            background:#f3f4f6 !important; color:#111827 !important;
        }}
        [role="option"][aria-selected="true"],
        [data-baseweb="menu"] li[aria-selected="true"],
        [role="menuitem"][aria-selected="true"] {{
            background:#e5e7eb !important; color:#111827 !important; font-weight:700 !important;
        }}
        [data-baseweb="tag"] {{
            background:#e5e7eb !important; border:1px solid #d1d5db !important;
            color:#111827 !important;
        }}
        [data-baseweb="tag"] * {{ color:#111827 !important; }}
        [data-baseweb="popover"]:not([data-baseweb="tooltip"]) input {{
            background:#ffffff !important; color:#111827 !important; border-color:#d1d5db !important;
        }}
        [data-baseweb="popover"]:not([data-baseweb="tooltip"]) small,
        [data-baseweb="popover"]:not([data-baseweb="tooltip"]) [class*="caption"] {{
            color:#4b5563 !important;
        }}

        /* Compact enterprise navigation rail */
        section[data-testid="stSidebar"] {{
            min-width:92px !important; max-width:92px !important; width:92px !important;
            flex:0 0 92px !important; overflow:visible !important; transform:none !important;
        }}
        [data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"] {{ display:none !important; }}
        [data-testid="stSidebarContent"] {{
            min-width:92px !important; max-width:92px !important; width:92px !important;
            box-sizing:border-box !important; overflow:visible !important;
        }}
        [data-testid="stSidebarUserContent"] {{
            min-width:68px !important; max-width:68px !important; width:68px !important;
            overflow:visible !important;
        }}
        section[data-testid="stSidebar"] > div:first-child {{
            padding:18px 12px !important; overflow:visible !important;
        }}
        [data-testid="stSidebar"] .brand-wrap {{
            justify-content:center; padding:4px 0 18px; margin-bottom:14px;
        }}
        [data-testid="stSidebar"] .brand-mark {{ width:54px; height:54px; padding:7px; }}
        [data-testid="stSidebar"] .brand-title,
        [data-testid="stSidebar"] .brand-sub,
        [data-testid="stSidebar"] .nav-section,
        [data-testid="stSidebar"] .portfolio-card,
        [data-testid="stSidebar"] .side-action {{ display:none !important; }}
        [data-testid="stSidebar"] .stButton {{ margin:7px 0 !important; }}
        [data-testid="stSidebar"] .stButton > button {{
            min-height:52px !important; width:68px !important; padding:0 !important;
            justify-content:center !important; border-radius:8px !important;
            font-size:22px !important; line-height:1 !important;
            border:1px solid transparent !important;
            transition:background-color 210ms ease, border-color 210ms ease, color 210ms ease !important;
        }}
        [data-testid="stSidebar"] .stButton > button p {{
            font-size:22px !important; line-height:1 !important; margin:0 !important;
            transform:scale(1); transition:transform 210ms cubic-bezier(.2,.8,.2,1) !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            color:#0f4c81 !important; background:#e8f3fc !important; border-color:#bfdbef !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] p {{
            transform:scale(1.18);
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            color:#0f4c81 !important; background:#eef6fd !important; border-color:#bfdbef !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_"] {{
            position:relative !important; overflow:visible !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_"]:before {{
            content:""; position:absolute; left:-12px; top:50%; width:3px; height:0;
            border-radius:0 4px 4px 0; background:#1769aa; transform:translateY(-50%);
            transition:height 210ms ease; z-index:2;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_"]:has(button[kind="primary"]):before {{
            height:30px;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_"]:after {{
            position:absolute; left:calc(100% + 14px); top:50%; transform:translate(-8px,-50%);
            width:max-content; max-width:250px; padding:12px 14px; border-radius:11px;
            color:#ffffff; background:#111827; box-shadow:0 10px 24px rgba(15,23,42,.20);
            font-size:12px; font-weight:500; line-height:1.45; text-align:left; white-space:pre-line;
            opacity:0; visibility:hidden; pointer-events:none; z-index:9999;
            transition:opacity 210ms ease, transform 210ms cubic-bezier(.2,.8,.2,1), visibility 210ms ease;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_"]:hover:after {{
            opacity:1; visibility:visible; transform:translate(0,-50%);
        }}
        [data-testid="stSidebar"] .st-key-nav_Dashboard:after {{
            content:"Dashboard\A Executive procurement controls and operating status";
        }}
        [data-testid="stSidebar"] .st-key-nav_Transaction-Feed:after {{
            content:"Transaction Feed\A Monitor procurement transactions in real time";
        }}
        [data-testid="stSidebar"] .st-key-nav_Vendor-Intelligence:after {{
            content:"Vendor Intelligence\A Analyze vendor risk, compliance and performance";
        }}
        [data-testid="stSidebar"] .st-key-nav_Risk-Reports:after {{
            content:"Risk Reports\A View control alerts and risk analytics";
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_Risk-"]:after {{
            content:"Risk & Compliance Monitoring\A Monitor compliance controls and active exceptions";
        }}
        [data-testid="stSidebar"] .st-key-nav_Risk-Reports:after {{
            content:"Risk Reports\A View control alerts and risk analytics";
        }}
        [data-testid="stSidebar"] .st-key-nav_Audit-Logs:after {{
            content:"Audit Logs\A Audit trail and activity monitoring";
        }}
        [data-testid="stSidebar"] .st-key-nav_Configuration:after {{
            content:"Configuration\A Configure procurement control policies";
        }}

        /* Live transaction risk meter */
        .risk-meter-wrap {{
            display:flex; flex-direction:column; align-items:center; gap:10px;
            padding:12px 0 18px; margin:0 0 12px; border-bottom:1px solid #d1d5db;
        }}
        .risk-meter {{
            --score:0; --risk-color:#2e7d32;
            width:154px; height:154px; border-radius:50%;
            display:grid; place-items:center;
            background:conic-gradient(var(--risk-color) calc(var(--score) * 1%), #e5e7eb 0);
            position:relative;
        }}
        .risk-meter:after {{
            content:""; position:absolute; inset:14px; border-radius:50%; background:#ffffff;
        }}
        .risk-meter-content {{ position:relative; z-index:1; text-align:center; }}
        .risk-meter-label {{ color:#6b7280; font-size:10px; font-weight:800; letter-spacing:.10em; text-transform:uppercase; }}
        .risk-meter-score {{ color:#000000; font-size:42px; font-weight:900; line-height:1; margin-top:5px; }}
        .risk-meter-category {{ color:#374151; font-size:13px; font-weight:800; }}
        .analysis-results-title {{
            color:#000000; font-size:11px; font-weight:900; letter-spacing:.11em;
            text-transform:uppercase; margin:8px 0 4px;
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
        <div class="brand-mark"><img src="{platform_logo_data_uri()}" alt="{APP_NAME} logo"></div>
        <div>
            <div class="brand-title">{APP_NAME}</div>
            <div class="brand-sub">{APP_SUBTITLE}</div>
        </div>
    </div>
    """,
        st.sidebar,
    )
    nav_icons = {
        "Dashboard": "\u25a6",
        "Transaction Feed": "\u2261",
        "Vendor Intelligence": "\u25c7",
        "Risk Reports": "\u25b3",
        "Risk & Compliance Monitoring": "\u2713",
        "Audit Logs": "\u25a4",
        "Configuration": "\u2699",
    }
    for section, name, code in PAGES:
        active = name == page
        if st.sidebar.button(nav_icons[name], key=f"nav_{name}", type="primary" if active else "secondary", use_container_width=True):
            st.session_state.active_page = name

    render_html(
        f"""
    <div class="portfolio-card">
        <div class="portfolio-title">CONTROL PORTFOLIO</div>
        <div class="stat-row"><span>Transactions Assessed</span><strong>{len(df):,}</strong></div>
        <div class="stat-row"><span>Control Exceptions</span><strong>{frauds:,}</strong></div>
        <div class="stat-row"><span>Pending Review</span><strong>{review:,}</strong></div>
        <div class="stat-row"><span>Exception Rate</span><strong>{rate:.1f}%</strong></div>
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
            <div class="page-kicker">PROCUREMENT OPERATIONS PLATFORM</div>
                <div class="page-title">{page}</div>
            </div>
            <div class="search-pill">Search vendors, invoices, purchase orders...</div>
            <div class="status-pill"><span class="status-light"></span>Risk & compliance monitoring</div>
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


def executive_control_grid(scores: Dict[str, float], txn: Dict[str, object]) -> None:
    risk = float(scores["risk_score"])
    health = "Stable" if float(scores["trust_score"]) >= 55 else "Watchlist"
    payment = "Elevated" if str(txn["payment_method"]) in ["UPI", "Wire"] else "Controlled"
    render_html(
        f"""
        <div class="executive-grid">
            <div class="executive-card"><div class="executive-label">Compliance Status</div><div class="executive-value"><span class="chip good">Compliant</span></div></div>
            <div class="executive-card"><div class="executive-label">Vendor Health</div><div class="executive-value"><span class="chip {'good' if health == 'Stable' else 'warn'}">{health}</span></div></div>
            <div class="executive-card"><div class="executive-label">Audit Readiness</div><div class="executive-value">92%</div></div>
            <div class="executive-card"><div class="executive-label">Approval Progress</div><div class="executive-value">2 of 4</div></div>
            <div class="executive-card"><div class="executive-label">Contract Alignment</div><div class="executive-value"><span class="chip good">Matched</span></div></div>
            <div class="executive-card"><div class="executive-label">Payment Risk Exposure</div><div class="executive-value"><span class="chip {'warn' if payment == 'Elevated' else 'good'}">{payment}</span></div></div>
            <div class="executive-card"><div class="executive-label">Procurement Control Score</div><div class="executive-value">{100 - risk:.0f}/100</div></div>
        </div>
        """
    )


def plot_theme(fig: go.Figure, height: int = 380, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    title_text = None
    if title:
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><span style='font-size:12px;color:#667085'>{subtitle}</span>"
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#344054", family="Inter", size=13),
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=18, color="#172b4d")) if title_text else None,
        margin=dict(l=52, r=28, t=76 if title_text else 34, b=54),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#475467"),
            bgcolor="rgba(255,255,255,0)",
        ),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cfd8e3", font=dict(color="#172b4d", family="Inter", size=12)),
        xaxis=dict(
            gridcolor="#edf0f4",
            zerolinecolor="#edf0f4",
            linecolor="#d9e0e8",
            tickfont=dict(size=12, color="#667085"),
            title_font=dict(size=13, color="#475467"),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="#edf0f4",
            zerolinecolor="#edf0f4",
            linecolor="#d9e0e8",
            tickfont=dict(size=12, color="#667085"),
            title_font=dict(size=13, color="#475467"),
            automargin=True,
        ),
    )
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


def gauge(score: float, title: str = "Control Risk Score") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 42, "color": "#111827"}},
            title={"text": title, "font": {"size": 15, "color": "#374151"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                "bar": {"color": risk_color(score), "thickness": 0.25},
                "bgcolor": "#f3f4f6",
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
                <div class="briefing-title">Procurement Transaction Context</div>
                <div class="briefing-copy">{title} is focused on {vendor["vendor_name"]} ({vendor["vendor_id"]}). Current transaction risk is {risk:.1f}/100 with {confidence:.1f}% control confidence across {len(scoped):,} context-linked records.</div>
            </div>
            <div class="briefing-card">
                <div class="briefing-title">Control Exception Rationale</div>
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
                <div class="briefing-title">Exception Review Rationale</div>
                <div class="briefing-copy">Exception intensity is {anomaly:.1f}/100, combining amount tolerance pressure, payment-route risk, timing controls, and the vendor's operating baseline.</div>
            </div>
            <div class="briefing-card">
                <div class="briefing-title">Department Exposure Summary</div>
                <div class="briefing-copy">{txn["department"]} is the active department context. Related charts, tables, alerts, and monitoring panels are filtered or prioritized around this selection.</div>
            </div>
        </div>
        """
    )


def procurement_intake_workspace(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    vendor_ids = list(vendors.sort_values("vendor_id")["vendor_id"])

    panel_open()
    render_html(
        """
        <div class="card-title">Vendor Information</div>
        <div class="card-note">Select the supplier record and confirm the purchasing ownership for this transaction.</div>
        """
    )
    selected_vendor = st.selectbox(
        "Search vendor name or ID",
        vendor_ids,
        index=vendor_ids.index(str(txn["vendor_id"])) if str(txn["vendor_id"]) in vendor_ids else 0,
        format_func=lambda x: f"{x} | {vendors.loc[vendors['vendor_id'] == x, 'vendor_name'].iloc[0]}",
        key="dash_vendor_id",
    )
    render_html(
        f"""
        <div class="context-strip">
            <div><div class="context-label">VENDOR ID</div><div class="context-value">{txn["vendor_id"]}</div></div>
            <div><div class="context-label">VENDOR NAME</div><div class="context-value">{vendor["vendor_name"]}</div></div>
            <div><div class="context-label">ONBOARDING STATUS</div><div class="context-value"><span class="chip good">Active supplier</span></div></div>
        </div>
        """
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        department = st.selectbox("Department", DEPARTMENTS, index=DEPARTMENTS.index(str(txn["department"])) if str(txn["department"]) in DEPARTMENTS else 0, key="dash_department")
    with c2:
        st.text_input("Cost Center", value="CC-IT-042", key="intake_cost_center")
    with c3:
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(str(txn["category"])) if str(txn["category"]) in CATEGORIES else 0, key="dash_category")
    panel_close()

    panel_open()
    render_html(
        """
        <div class="card-title">Invoice Information</div>
        <div class="card-note">Capture invoice identifiers, document date, payment route, and the transaction value submitted for review.</div>
        """
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Invoice Number", value=f"INV-{str(txn['transaction_id'])[-6:]}", key="intake_invoice")
    with c2:
        st.date_input("Invoice Date", value=pd.to_datetime(txn["timestamp"]).date(), key="intake_invoice_date")
    with c3:
        payment = st.selectbox("Payment Method", PAYMENTS, index=PAYMENTS.index(str(txn["payment_method"])) if str(txn["payment_method"]) in PAYMENTS else 0, key="dash_payment")
    c4, c5 = st.columns(2)
    with c4:
        st.selectbox("Currency", ["INR", "USD", "EUR", "GBP", "SGD"], key="intake_currency")
    with c5:
        amount = st.number_input("Transaction Amount", min_value=1000.0, max_value=5000000.0, value=float(txn["amount"]), step=10000.0, key="dash_amount")
    panel_close()

    panel_open()
    render_html(
        """
        <div class="card-title">Purchase Order Details</div>
        <div class="card-note">Associate this invoice with its purchase order and governing contract before validation.</div>
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Purchase Order Number", value=f"PO-2026-{str(txn['transaction_id'])[-5:]}", key="intake_po")
    with c2:
        st.text_input("Contract Reference", value="CTR-IT-2026-0184", key="intake_contract")
    render_html('<a class="contract-link" href="#">View linked master services agreement</a>')
    panel_close()

    panel_open()
    render_html(
        """
        <div class="card-title">Compliance Validation</div>
        <div class="card-note">Automated control checks are refreshed when the procurement transaction context changes.</div>
        <div class="check-row"><span>Vendor onboarding controls</span><span class="chip good">Passed</span></div>
        <div class="check-row"><span>Tax registration verification</span><span class="chip good">Verified</span></div>
        <div class="check-row"><span>Contract coverage</span><span class="chip good">Matched</span></div>
        <div class="check-row"><span>Invoice amount tolerance</span><span class="chip warn">Review required</span></div>
        """
    )
    panel_close()

    panel_open()
    render_html(
        """
        <div class="card-title">Approval Workflow</div>
        <div class="card-note">Assign the accountable approval authority and provide the supporting transaction documents.</div>
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Approval Authority", ["M. Kapoor | Finance Director", "A. Rao | Procurement Lead", "S. Iyer | Department Head"], key="intake_authority")
    with c2:
        st.selectbox("Approval Status", ["Pending Review", "Approved", "Escalated"], key="intake_approval_status")
    st.file_uploader("Document attachments", accept_multiple_files=True, key="intake_documents")
    render_html('<div class="attachment-box">Attach purchase order, invoice PDF, contract addendum, or supporting approval documents.</div>')
    panel_close()

    auto_synced = sync_master_context(df, vendors, selected_vendor, amount, payment, department, category)
    if auto_synced:
        st.rerun()

    if st.button("Validate Procurement Record", use_container_width=True):
        sync_master_context(df, vendors, selected_vendor, amount, payment, department, category)
        st.rerun()


def transaction_analysis_sidebar() -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    scores = st.session_state.current_scores
    risk = float(scores["risk_score"])
    if risk >= 76:
        meter_color, risk_category = "#c62828", "High Risk"
    elif risk >= 36:
        meter_color, risk_category = "#d97706", "Medium Risk"
    else:
        meter_color, risk_category = "#2e7d32", "Low Risk"
    compliance = "Compliant" if float(scores["compliance_score"]) >= 70 else "Review Required"
    duplicate = "No duplicate detected" if not bool(txn.get("split_pattern", False)) else "Potential split invoice"

    panel_open("Live Transaction Analysis", "Procurement transaction context and control validation summary.")
    render_html(
        f"""
        <div class="risk-meter-wrap">
            <div class="risk-meter" style="--score:{risk:.0f}; --risk-color:{meter_color};">
                <div class="risk-meter-content">
                    <div class="risk-meter-label">Risk Score</div>
                    <div class="risk-meter-score">{risk:.0f}</div>
                </div>
            </div>
            <div class="risk-meter-category" style="color:{meter_color};">{risk_category}</div>
        </div>
        <div class="analysis-results-title">Analysis Results</div>
        <div class="check-row"><span>Vendor Risk Score</span><span class="chip {'warn' if risk >= 58 else 'good'}">{risk_label(risk)}</span></div>
        <div class="check-row"><span>Compliance Status</span><span class="chip good">{compliance}</span></div>
        <div class="check-row"><span>Contract Match Status</span><span class="chip good">Matched</span></div>
        <div class="check-row"><span>Duplicate Invoice Check</span><span class="chip {'warn' if bool(txn.get("split_pattern", False)) else 'good'}">{duplicate}</span></div>
        <div class="check-row"><span>Payment Risk Indicator</span><span class="chip {'warn' if str(txn["payment_method"]) in ["UPI", "Wire"] else 'good'}">{txn["payment_method"]} monitored</span></div>
        <div class="check-row"><span>Approval Chain Status</span><span class="chip pending">Pending finance review</span></div>
        """
    )
    panel_close()
    panel_open("Transaction Reference", "Linked record identifiers for audit traceability.")
    render_html(
        f"""
        <div class="check-row"><span>Transaction ID</span><strong>{txn["transaction_id"]}</strong></div>
        <div class="check-row"><span>Vendor ID</span><strong>{vendor["vendor_id"]}</strong></div>
        <div class="check-row"><span>Contract Reference</span><a class="contract-link" href="#">CTR-IT-2026-0184</a></div>
        <div class="check-row"><span>Review SLA</span><strong>14 minutes</strong></div>
        """
    )
    panel_close()


def dashboard_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    scores = st.session_state.current_scores
    alerts = st.session_state.current_alerts
    scoped = contextual_portfolio(df)
    vendor_txns = contextual_transactions(df)

    render_html(
        f"""
        <div class="workflow-head">
            <div>
                <div class="workflow-title">Procurement Transaction Intake</div>
                <div class="workflow-sub">Validate vendor, invoice, purchase order, compliance, and approval information before posting.</div>
            </div>
            <div class="workflow-meta">
                <span class="chip pending">Pending Review</span>
                <span>Last updated {datetime.now().strftime("%d %b %Y, %H:%M")} IST</span>
            </div>
        </div>
        """
    )
    kpi_grid(
        [
            ("Selected Vendor", str(txn["vendor_id"]), str(vendor["vendor_name"])),
            ("Transaction Amount", money(float(txn["amount"])), f'{txn["payment_method"]} payment method'),
            ("Vendor Compliance", f'{scores["compliance_score"]:.0f}/100', "Supplier master controls"),
            ("Approval Status", "Pending Review", "Finance and procurement workflow"),
        ]
    )
    executive_control_grid(scores, txn)

    left, right = st.columns([1.72, 0.92], gap="large")
    with left:
        procurement_intake_workspace(df, vendors)
    with right:
        transaction_analysis_sidebar()

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        panel_open("Control Risk Trend", "Risk movement for records connected to the active vendor, department, category, or payment route.")
        trend = scoped.set_index("timestamp").resample("W")["risk_score"].mean().reset_index()
        fig = px.area(trend, x="timestamp", y="risk_score", color_discrete_sequence=[C["teal"]])
        fig.add_scatter(x=trend["timestamp"], y=trend["risk_score"], mode="lines", line=dict(color=C["blue"], width=2), name="risk")
        fig.update_xaxes(title_text="Week")
        fig.update_yaxes(title_text="Average risk score")
        st.plotly_chart(plot_theme(fig, 390, "Procurement Control Risk Trend", f"{txn['vendor_id']} plus matching department, category, and payment context"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with c2:
        panel_open("Active Control Exceptions", "Current review items generated from the procurement transaction context.")
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
        panel_open("Transaction Risk Distribution", "Amount-to-risk relationship for synchronized procurement records.")
        sample = scoped.sample(min(180, len(scoped)), random_state=4) if len(scoped) > 1 else scoped
        fig = px.scatter(sample, x="amount", y="risk_score", color="status", size="anomaly_score", color_discrete_map={"Blocked": C["red"], "Review": C["amber"], "Watch": C["blue"], "Approved": C["green"]})
        fig.update_xaxes(title_text="Transaction amount")
        fig.update_yaxes(title_text="Risk score")
        st.plotly_chart(plot_theme(fig, 380, "Transaction Risk Distribution", f"{len(sample):,} context-linked observations"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Compliance Exception Analysis", "Full-width review of amount outliers, payment signals, and vendor exceptions in the active procurement context.", "anomaly")
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
    fig.add_hline(y=scores["anomaly_score"], line_dash="dash", line_color=C["teal"], annotation_text="Current exception score")
    fig.update_xaxes(title_text="Transaction timeline")
    fig.update_yaxes(title_text="Exception intensity")
    st.plotly_chart(plot_theme(fig, 520, "Vendor Exception Timeline", f"{txn['vendor_id']} | current score {scores['anomaly_score']:.1f}/100"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    panel_open("Procurement Review Briefing", "Summary generated from the synchronized vendor, transaction, score, alert, and filter state.")
    summary_briefing("Dashboard briefing", scoped)
    panel_close()

    panel_open("Live Activity Feed", "Current transaction is pinned above recent records from the selected vendor.")
    feed = vendor_txns.head(10)[["timestamp", "transaction_id", "vendor_id", "vendor_name", "amount", "payment_method", "risk_score", "status"]].copy()
    st.dataframe(feed, use_container_width=True, hide_index=True)
    panel_close()


def transaction_feed_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    panel_open("Searchable Transaction Register", "The active vendor is pinned first; filters begin from the procurement transaction context.")
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
        panel_open("Synchronized Selection", "This is the same procurement transaction context used by all pages.")
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

    panel_open("Transaction Register Summary", "Briefing generated from filtered records and the active procurement transaction context.")
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
            ("Control Exception History", f'{int(vendor["frauds"])} flags', f'Max risk {vendor["max_risk"]:.0f}/100'),
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
        panel_open("Control Exception History", "Historical vendor exceptions by status.")
        history = vdf.groupby("status", as_index=False).agg(count=("transaction_id", "count"), avg_risk=("risk_score", "mean"))
        fig = px.bar(history, x="status", y="count", color="avg_risk", color_continuous_scale=["#43e39f", "#ffc857", "#ff5d6c"])
        fig.update_xaxes(title_text="Status")
        fig.update_yaxes(title_text="Transaction count")
        st.plotly_chart(plot_theme(fig, 380, "Control Exception History", f"{int(vendor['frauds'])} historical exception flags"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Vendor Exception Investigation", "Full-width exception review for this vendor's transaction timeline.", "anomaly")
    fig = px.area(vdf, x="timestamp", y="anomaly_score", color_discrete_sequence=[C["teal"]])
    fig.add_scatter(x=vdf["timestamp"], y=vdf["risk_score"], mode="lines", line=dict(color=C["red"], width=2), name="Risk score")
    fig.add_hline(y=scores["anomaly_score"], line_dash="dash", line_color=C["amber"], annotation_text="Current anomaly")
    fig.update_xaxes(title_text="Transaction timeline")
    fig.update_yaxes(title_text="Control score")
    st.plotly_chart(plot_theme(fig, 500, "Vendor Exception Review", f"{txn['vendor_id']} exception and risk behavior"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    panel_open("Control Recommendation Panel", "Control guidance based on the synchronized vendor context.")
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
        panel_open("Control Status Distribution", "Status distribution for the same synchronized context.")
        dist = scoped["status"].value_counts().rename_axis("status").reset_index(name="count")
        fig = px.pie(dist, names="status", values="count", hole=0.58, color="status", color_discrete_map={"Blocked": C["red"], "Review": C["amber"], "Watch": C["blue"], "Approved": C["green"]})
        st.plotly_chart(plot_theme(fig, 410, "Control Status Distribution", f"{len(scoped):,} context-linked records"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    r2a, r2b = st.columns(2, gap="large")
    with r2a:
        panel_open("Hourly Control Exposure", "Average risk by transaction hour inside the active context.")
        hourly = scoped.assign(hour=scoped["timestamp"].dt.hour).groupby("hour", as_index=False)["risk_score"].mean()
        fig = px.bar(hourly, x="hour", y="risk_score", color="risk_score", color_continuous_scale=["#43e39f", "#ffc857", "#ff5d6c"])
        fig.update_xaxes(title_text="Hour of day")
        fig.update_yaxes(title_text="Average risk")
        st.plotly_chart(plot_theme(fig, 380, "Hourly Control Exposure", "Late-hour risk patterns are easier to isolate"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with r2b:
        panel_open("Category Exposure", "Spend concentration by category for synchronized context.")
        cat = scoped.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        fig = px.bar(cat, x="category", y="amount", color_discrete_sequence=[C["teal"]])
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Spend")
        st.plotly_chart(plot_theme(fig, 380, "Category Exposure", f"Active category: {txn['category']}"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Control Exception Report", "Full-width exception section for risk reporting and compliance review.", "anomaly")
    trend = scoped.set_index("timestamp").resample("D").agg(risk_score=("risk_score", "mean"), anomaly_score=("anomaly_score", "mean"), amount=("amount", "sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend["timestamp"], y=trend["amount"], name="Spend", marker_color="rgba(87,166,255,0.35)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=trend["timestamp"], y=trend["risk_score"], mode="lines", name="Risk", line=dict(color=C["red"], width=3)))
    fig.add_trace(go.Scatter(x=trend["timestamp"], y=trend["anomaly_score"], mode="lines", name="Anomaly", line=dict(color=C["teal"], width=3)))
    fig.update_layout(yaxis2=dict(title="Spend", overlaying="y", side="right", gridcolor="rgba(255,255,255,0)", tickfont=dict(color="#6b7280")))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Control score")
    st.plotly_chart(plot_theme(fig, 520, "Risk and Spend Control Signal", "Daily exceptions, risk, and procurement spend in one executive view"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    panel_open("High-Risk Vendors and Control Alerts", "Selected vendor is included with the highest-risk peer vendors.")
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
            ("Control Confidence", pct(scores["ai_confidence"]), "Selected transaction assessment"),
            ("Exception Intensity", pct(scores["anomaly_score"]), "Current control signal"),
            ("Active Review Items", f"{active:,}", "High and critical events"),
            ("Control Service Health", "99.97%", "Operational availability"),
        ]
    )

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        panel_open("Control Service Indicators", "Operational control health for the synchronized procurement context.")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[92, 88, 96, 84, scores["ai_confidence"], 91], theta=["Inference", "Data Freshness", "Policy Rules", "Drift", "Confidence", "Latency"], fill="toself", line=dict(color=C["teal"])))
        fig.update_layout(polar=dict(bgcolor="rgba(255,255,255,0)", radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(132,179,207,0.14)")))
        st.plotly_chart(plot_theme(fig, 410, "Control Service Health", "Confidence, latency, policy, and data freshness status"), use_container_width=True, config={"displayModeBar": False})
        panel_close()
    with c2:
        panel_open("Alert Severity", "Alert mix generated from context-linked records.")
        severity = scoped.assign(severity=scoped["risk_score"].apply(risk_label)).groupby("severity", as_index=False).size()
        fig = px.bar(severity, x="severity", y="size", color="severity", color_discrete_map={"Critical": C["red"], "High": C["amber"], "Elevated": C["blue"], "Clear": C["green"]})
        fig.update_xaxes(title_text="Severity")
        fig.update_yaxes(title_text="Detection count")
        st.plotly_chart(plot_theme(fig, 410, "Alert Severity", f"{active:,} active detections"), use_container_width=True, config={"displayModeBar": False})
        panel_close()

    panel_open("Control Monitoring Overview", "Full-width operations view for active review signals and exception intensity.", "anomaly")
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
    fig.update_yaxes(title_text="Exception intensity")
    st.plotly_chart(plot_theme(fig, 520, "Live Control Monitoring", f"{txn['vendor_id']} monitored with {scores['ai_confidence']:.1f}% control confidence"), use_container_width=True, config={"displayModeBar": False})
    panel_close()

    c3, c4 = st.columns([1, 1], gap="large")
    with c3:
        panel_open("Live Scanning Feed", "Current transaction pinned to recent selected-vendor scans.")
        scan = vendor_txns.head(12)[["timestamp", "transaction_id", "vendor_id", "payment_method", "risk_score", "anomaly_score", "status"]].copy()
        st.dataframe(scan, use_container_width=True, hide_index=True)
        panel_close()
    with c4:
        panel_open("Control Service Diagnostics", "Operational diagnostics for the active procurement context.")
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

    panel_open("Risk & Compliance Briefing", "Control explanation and operations summary for the selected vendor context.")
    summary_briefing("Risk and compliance briefing", scoped)
    panel_close()


def audit_logs_page(df: pd.DataFrame, vendors: pd.DataFrame) -> None:
    txn = st.session_state.current_transaction
    vendor = st.session_state.current_vendor
    events = [
        ("Transaction assessed", txn["transaction_id"], "Control service scored the transaction and synchronized session state."),
        ("Vendor graph refreshed", txn["vendor_id"], f'{vendor["vendor_name"]} relationship, spend, and compliance context loaded.'),
        ("Risk report updated", risk_label(float(txn["risk_score"])), "Heatmaps and predictive alerts now reflect the selected transaction context."),
        ("Control monitoring event", pct(float(txn["ai_confidence"])), "Confidence and exception diagnostics recalculated for active selection."),
        ("Control recommendation", str(txn["status"]), "Workflow policy mapped to current score and payment rail."),
    ]
    rows = []
    base = datetime.now()
    for i, (event, entity, detail) in enumerate(events):
        rows.append({"time": base - timedelta(minutes=i * 4), "event": event, "entity": entity, "detail": detail, "actor": "ProcureShield Controls"})
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
        st.selectbox("Control profile", ["ProcureShield Standard Controls", "Expedited Review", "Conservative Audit"])
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
            ProcureShield | Procurement operations and compliance controls | Demo data generated locally
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
    elif page == "Risk & Compliance Monitoring":
        ai_monitoring_page(df, vendors)
    elif page == "Audit Logs":
        audit_logs_page(df, vendors)
    elif page == "Configuration":
        configuration_page(df, vendors)
    footer()


if __name__ == "__main__":
    main()
