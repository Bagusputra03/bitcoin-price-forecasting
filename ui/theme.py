import streamlit as st

APP_NAME = "Bitcoin Forecast Intelligence"
APP_TAGLINE = "Daily Bitcoin Price Forecasting"

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --btc:#F2A900; --btc-soft:#FFC84A; --black:#0B0B0C; --charcoal:#151515;
  --paper:#F7F5EF; --card:#FFFFFF; --ink:#171717; --muted:#6F6B63; --line:#E8E2D7;
  --green:#168A58; --red:#C84A41; --warm:#FFF8E6;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:linear-gradient(180deg,#FAF9F5 0%,#F6F3EC 100%);color:var(--ink)}
#MainMenu,footer,header{visibility:hidden}
.block-container{max-width:1460px;padding-top:1.35rem;padding-bottom:4rem}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#080808 0%,#111111 58%,#17130B 100%);border-right:1px solid rgba(242,169,0,.18);min-width:270px;max-width:270px}
[data-testid="stSidebar"]>div:first-child{width:270px}
[data-testid="stSidebar"] *{color:#F5F3EE}
[data-testid="stSidebar"] div[role="radiogroup"] label{border-radius:12px;margin:.16rem 0;padding:.54rem .68rem;transition:.18s;min-height:42px;border:1px solid transparent}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(242,169,0,.09);border-color:rgba(242,169,0,.14)}
[data-testid="stSidebar"] div[role="radiogroup"] label p{font-size:13px;white-space:normal;line-height:1.3;font-weight:600}
.brand-shell{padding:12px 3px 17px;display:flex;align-items:center;gap:12px}
.brand-icon{width:43px;height:43px;border-radius:14px;background:linear-gradient(145deg,#FFCA55,#F2A900);color:#111;display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:800;box-shadow:0 12px 32px rgba(242,169,0,.22)}
.brand-title{font-size:16px;font-weight:800;line-height:1.12;color:#fff}.brand-sub{font-size:9px;color:#F2A900;letter-spacing:.13em;margin-top:4px;font-weight:700}
.nav-caption{font-size:9px!important;letter-spacing:.13em;color:#8D877D!important;font-weight:800;margin:10px 4px 6px}
.signed-card{padding:12px 13px;border:1px solid rgba(242,169,0,.16);border-radius:14px;background:rgba(242,169,0,.05);display:flex;flex-direction:column;gap:3px}.signed-card small{font-size:8px;color:#A29B8E}.signed-card b{font-size:13px}.signed-card span{font-size:10px;color:#A29B8E}

/* Hero */
.hero{background:linear-gradient(128deg,#0B0B0C 0%,#171717 62%,#3B2A08 115%);padding:31px 35px;border-radius:24px;color:white;box-shadow:0 20px 48px rgba(18,16,10,.13);margin-bottom:22px;position:relative;overflow:hidden;border:1px solid rgba(242,169,0,.14)}
.hero:after{content:'₿';position:absolute;right:34px;top:-25px;font-size:180px;font-weight:800;color:rgba(242,169,0,.07);transform:rotate(9deg)}
.eyebrow{font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:#F2A900;font-weight:800}.hero h1{font-size:31px;line-height:1.14;margin:9px 0 10px;font-weight:800;max-width:900px}.hero p{font-size:13px;line-height:1.65;color:#CBC6BC;max-width:880px;margin:0}

/* Cards */
.metric-card{background:rgba(255,255,255,.96);border:1px solid var(--line);border-top:3px solid rgba(242,169,0,.72);border-radius:17px;padding:17px 19px;box-shadow:0 9px 26px rgba(37,31,19,.05);min-height:111px}.metric-label{font-size:10px;color:#777166;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.metric-value{font-size:25px;font-weight:800;color:#151515;margin:8px 0 2px;word-break:break-word}.metric-hint{font-size:10px;color:#9A9387}
.panel{background:rgba(255,255,255,.97);border:1px solid var(--line);border-radius:19px;padding:21px;box-shadow:0 8px 26px rgba(37,31,19,.045);margin:8px 0 16px}
.soft-badge,.live-badge,.gold-badge{display:inline-block;background:#FFF5D8;color:#7B5700;border:1px solid #F3D27B;padding:6px 10px;border-radius:999px;font-size:9px;font-weight:800;margin-right:6px;letter-spacing:.04em}.live-badge{background:#ECF8F1;color:#126B45;border-color:#BFE3D0}.gold-badge{background:#F2A900;color:#111;border-color:#F2A900}
.flow-step{background:#fff;border:1px solid var(--line);border-radius:15px;padding:15px 10px;box-shadow:0 5px 18px rgba(37,31,19,.035);text-align:center;font-size:11px;font-weight:700;min-height:68px;position:relative}.flow-step:before{content:'';position:absolute;top:0;left:18%;right:18%;height:2px;background:#F2A900;border-radius:2px}
.step-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:0 6px 20px rgba(37,31,19,.04);min-height:116px;display:flex;flex-direction:column;gap:5px;margin-bottom:9px}.step-card span{font-size:9px;font-weight:800;color:#A77700;letter-spacing:.12em}.step-card b{font-size:15px}.step-card small{color:#918A7E;line-height:1.45}
.mode-card{border-radius:20px;padding:20px 22px;min-height:158px;border:1px solid;box-shadow:0 9px 26px rgba(37,31,19,.045)}.mode-card.research{background:linear-gradient(145deg,#fff,#FFF9E9);border-color:#EED99D}.mode-card.live{background:linear-gradient(145deg,#fff,#F1FBF5);border-color:#C8E7D5}.mode-card h3{font-size:19px;margin:5px 0 8px}.mode-card p{color:#716B61;font-size:11px;line-height:1.6;min-height:36px}.mode-kicker{font-size:9px;letter-spacing:.12em;font-weight:800;color:#8A8174}

/* Calls */
.callout{border-radius:15px;padding:14px 16px;margin:8px 0 17px;font-size:11px;line-height:1.65;border:1px solid}.callout.source{background:#FFF8E8;border-color:#F1D58F;color:#654C12}.callout.extension{background:#F8F5ED;border-color:#E5DED0;color:#5B554B}.callout.live{background:#EDF9F2;border-color:#BDE4CD;color:#125E40}.callout.danger{background:#FFF1EF;border-color:#F1C7C2;color:#8E2E28}
.status-row{display:flex;gap:10px;align-items:center;padding:12px 14px;border-radius:14px;border:1px solid var(--line);background:#fff;margin-bottom:9px}.status-dot{width:9px;height:9px;border-radius:50%;background:#168A58;box-shadow:0 0 0 4px rgba(22,138,88,.10)}.status-dot.warn{background:#D39200;box-shadow:0 0 0 4px rgba(211,146,0,.12)}.status-row b{font-size:12px}.status-row span{font-size:10px;color:#8B8478;margin-left:auto}

/* Prediction */
.prediction-hero{background:linear-gradient(128deg,#0A0A0A,#181818 68%,#4A3305);border:1px solid rgba(242,169,0,.2);border-radius:22px;padding:25px 27px;color:#fff;box-shadow:0 16px 40px rgba(20,18,12,.11)}.prediction-hero small{color:#BDB6A9;font-size:10px;text-transform:uppercase;letter-spacing:.09em}.prediction-hero .price{font-size:38px;font-weight:800;margin:7px 0 3px;color:#FFD36B}.prediction-hero .delta-up{color:#5AD39B;font-weight:800}.prediction-hero .delta-down{color:#FF8A7F;font-weight:800}
.prediction-strip{display:flex;align-items:center;gap:12px;background:#FFF8E8;border:1px solid #EED99D;border-radius:14px;padding:12px 14px;margin-top:8px}.prediction-strip.live{background:#ECF8F1;border-color:#BFE3D0}.prediction-strip small{color:#756E63}.prediction-strip b{font-size:18px}.prediction-strip span{margin-left:auto;font-weight:800}

/* Detail, login */
.detail-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0}.detail-grid>div{background:#fff;border:1px solid var(--line);border-radius:13px;padding:12px}.detail-grid small{display:block;color:#9A9387;font-size:8px;text-transform:uppercase;letter-spacing:.08em}.detail-grid b{font-size:11px;display:block;margin-top:4px;word-break:break-word}
.login-shell{max-width:475px;margin:8vh auto 0;background:rgba(255,255,255,.98);border:1px solid #E8E2D7;border-radius:25px;padding:29px 31px 25px;box-shadow:0 24px 70px rgba(22,19,12,.13);position:relative;overflow:hidden}.login-shell:after{content:'₿';position:absolute;right:-16px;top:-28px;font-size:125px;color:rgba(242,169,0,.07);font-weight:800}.login-brand{width:53px;height:53px;border-radius:16px;background:linear-gradient(145deg,#FFCB58,#F2A900);display:flex;align-items:center;justify-content:center;color:#111;font-size:27px;font-weight:800;margin-bottom:18px;box-shadow:0 12px 28px rgba(242,169,0,.18)}

/* Native Streamlit */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #19170F !important;
    color: #F2A900 !important;
    border: 1px solid rgba(242, 169, 0, 0.55) !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #F2A900 !important;
    color: #111111 !important;
    border-color: #F2A900 !important;
}
[data-testid="stTextInput"] input,[data-testid="stDateInput"] input,[data-baseweb="select"]>div,[data-testid="stTextArea"] textarea{border-radius:11px!important;border-color:#DDD5C7!important}.stTextInput input:focus,.stTextArea textarea:focus{border-color:#F2A900!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:13px;overflow:hidden}[data-testid="stAlert"]{border-radius:14px!important}[data-testid="stStatusWidget"]{border-radius:15px!important;border:1px solid var(--line)!important;background:rgba(255,255,255,.96)!important}[data-testid="stTabs"] button{font-weight:700!important}hr{border-color:#E8E2D7}.small-muted{color:#989083;font-size:10px}

@media(max-width:900px){.hero{padding:23px}.hero h1{font-size:26px}.hero:after{font-size:120px}.detail-grid{grid-template-columns:repeat(2,minmax(0,1fr))}[data-testid="stSidebar"]{min-width:250px;max-width:250px}}
</style>
"""

def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)

def hero(title, subtitle, eyebrow=APP_NAME.upper()):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def metric_card(label, value, hint=''):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )
