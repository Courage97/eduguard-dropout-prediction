import streamlit as st

st.set_page_config(
    page_title="EduGuard — Dropout Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── PALETTE ──────────────────────────────────────────────────────────────────
BLUE1  = "#3A4EFB"
BLUE2  = "#33A4FA"
YELLOW = "#E3FF3B"
DARK   = "#252943"
INK    = "#12142B"
WHITE  = "#FFFFFF"
RED    = "#FF4B4B"
GREEN  = "#00C48C"
AMBER  = "#FFB800"
PURPLE = "#7B2FBE"
BORDER = "#E0E3FF"
MUTED  = "#6B7280"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,900&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',sans-serif!important;box-sizing:border-box;}
html{scroll-behavior:smooth;}
.main{background:#F7F8FF!important;}
.block-container{padding:0!important;max-width:100%!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}

/* ── NAVBAR ─────────────────────────────────────────────────────────── */
.edu-nav{position:sticky;top:0;z-index:9999;background:rgba(37,41,67,0.92);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  padding:0 56px;display:flex;align-items:center;justify-content:space-between;
  min-height:72px;border-bottom:1px solid rgba(255,255,255,0.07);}
.edu-brand{display:flex;align-items:center;gap:13px;text-decoration:none;}
.edu-brand-icon{background:linear-gradient(135deg,#3A4EFB,#33A4FA);width:42px;height:42px;
  border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:21px;
  box-shadow:0 6px 16px rgba(58,78,251,0.35);}
.edu-brand-name{color:#fff;font-size:25px;font-weight:900;letter-spacing:-0.5px;line-height:1.1;}
.edu-brand-sub{color:#E3FF3B;font-size:10px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin-top:2px;}
.edu-navlinks{display:flex;align-items:center;gap:2px;}
.edu-navlinks a{color:rgba(222,224,237,0.72)!important;font-size:17px;font-weight:500;
  padding:9px 18px;border-radius:8px;text-decoration:none!important;transition:color .2s,background .2s;}
.edu-navlinks a:hover{color:#fff!important;background:rgba(255,255,255,0.07);}
.edu-navlinks a.active{color:#fff;background:rgba(255,255,255,0.1);}
.edu-live{display:inline-flex;align-items:center;gap:7px;background:rgba(227,255,59,0.12);
  border:1px solid rgba(227,255,59,0.4);color:#E3FF3B;padding:7px 16px;border-radius:24px;
  font-size:11px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;}
.edu-live-dot{width:6px;height:6px;border-radius:50%;background:#E3FF3B;
  animation:eduPulse 1.8s ease-in-out infinite;}
@keyframes eduPulse{0%,100%{opacity:1;}50%{opacity:0.35;}}

/* ── HERO ───────────────────────────────────────────────────────────── */
.edu-hero{position:relative;overflow:hidden;display:grid;
  grid-template-columns:1.05fr 0.95fr;min-height:600px;
  background:
    radial-gradient(circle at 12% 15%, rgba(58,78,251,0.45), transparent 42%),
    radial-gradient(circle at 88% 10%, rgba(51,164,250,0.35), transparent 45%),
    radial-gradient(circle at 72% 90%, rgba(227,255,59,0.10), transparent 40%),
    linear-gradient(150deg, #171A34 0%, #252943 50%, #181B37 100%);}
.edu-hero::before{content:"";position:absolute;inset:0;z-index:0;
  background-image:radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
  background-size:26px 26px;pointer-events:none;}
.edu-hero-left{position:relative;z-index:1;padding:84px 60px;
  display:flex;flex-direction:column;justify-content:center;}
.edu-hero-badge{display:inline-block;background:#E3FF3B;color:#252943;padding:6px 16px;
  border-radius:24px;font-size:11px;font-weight:800;letter-spacing:2.5px;
  text-transform:uppercase;margin-bottom:26px;width:fit-content;
  animation:eduFadeInUp .6s ease both;}
.edu-h1-red,.edu-h1-white{font-family:'Fraunces','Georgia',ui-serif,serif!important;
  font-optical-sizing:auto;font-size:64px;font-weight:600;line-height:1.28;
  letter-spacing:0.5px;word-spacing:2px;padding:0!important;}
.edu-h1-red{color:#FF4B4B!important;margin:0!important;animation:eduFadeInUp .6s ease .1s both;}
.edu-h1-white{color:#fff!important;margin:0 0 22px!important;animation:eduFadeInUp .6s ease .2s both;}
.edu-hero-body{color:#A8B8D8;font-size:25px;line-height:1.75;margin:0 0 32px;
  font-weight:400;max-width:480px;animation:eduFadeInUp .6s ease .3s both;}
.edu-chips{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:32px;
  animation:eduFadeInUp .6s ease .4s both;}
.edu-chip{background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.14);
  padding:9px 18px;border-radius:10px;font-size:15px;font-weight:600;color:#fff;}
.edu-hero-btnrow{display:flex;gap:14px;flex-wrap:wrap;animation:eduFadeInUp .6s ease .5s both;}
.edu-cta-btn{background:#E3FF3B;color:#252943!important;padding:15px 30px;border-radius:12px;
  font-weight:800;font-size:20px;text-decoration:none!important;letter-spacing:0.3px;display:inline-block;
  box-shadow:0 14px 32px rgba(227,255,59,0.22);transition:transform .2s ease, box-shadow .2s ease;}
.edu-cta-btn:hover{transform:translateY(-2px);box-shadow:0 18px 38px rgba(227,255,59,0.32);}
.edu-cta-btn-ghost{background:rgba(255,255,255,0.06);color:#fff!important;padding:15px 26px;
  border-radius:12px;font-weight:700;font-size:20px;text-decoration:none!important;display:inline-block;
  border:1px solid rgba(255,255,255,0.18);transition:background .2s ease, transform .2s ease;}
.edu-cta-btn-ghost:hover{background:rgba(255,255,255,0.12);transform:translateY(-2px);}

.edu-hero-visual{position:relative;z-index:1;overflow:hidden;min-height:600px;
  display:flex;align-items:center;justify-content:center;}
.edu-network-svg{position:absolute;inset:0;width:100%;height:100%;opacity:0.85;}
.edu-predict-card{position:relative;z-index:3;background:rgba(255,255,255,0.98);
  border-radius:20px;padding:24px 26px;width:min(320px,88vw);
  box-shadow:0 30px 70px rgba(8,10,30,0.5),0 0 0 1px rgba(255,255,255,0.05);
  animation:eduFloat 6s ease-in-out infinite, eduFadeInUp .8s ease .35s both;}
.edu-predict-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}
.edu-predict-id{display:flex;align-items:center;gap:10px;}
.edu-predict-avatar{width:36px;height:36px;border-radius:10px;flex-shrink:0;
  background:linear-gradient(135deg,#3A4EFB,#33A4FA);display:flex;align-items:center;
  justify-content:center;font-size:20px;}
.edu-predict-name{font-size:18px;font-weight:700;color:#252943;}
.edu-predict-meta{font-size:15px;color:#9CA3AF;margin-top:1px;}
.edu-predict-risk-badge{background:#FFF0F0;color:#FF4B4B;font-size:15px;font-weight:800;
  letter-spacing:0.5px;padding:6px 11px;border-radius:8px;white-space:nowrap;}
.edu-predict-score-row{display:flex;align-items:baseline;gap:8px;margin-bottom:18px;}
.edu-predict-score{font-size:40px;font-weight:900;color:#252943;letter-spacing:-1px;}
.edu-predict-score-label{font-size:15px;color:#9CA3AF;font-weight:500;}
.edu-predict-factors{display:flex;flex-direction:column;gap:11px;margin-bottom:18px;}
.edu-factor-label{display:flex;justify-content:space-between;font-size:15px;
  color:#374151;font-weight:600;margin-bottom:4px;}
.edu-factor-track{height:6px;background:#F0F1FA;border-radius:6px;overflow:hidden;}
.edu-factor-fill{height:100%;border-radius:6px;}
.edu-predict-action{display:flex;align-items:center;gap:9px;background:#F4F5FF;
  border-radius:10px;padding:11px 13px;font-size:17px;color:#252943;margin-bottom:9px;}
.edu-predict-action-dot{width:7px;height:7px;border-radius:50%;background:#FF4B4B;flex-shrink:0;}
.edu-predict-caption{font-size:15px;color:#B0B4C8;text-align:right;font-style:italic;}

.edu-float-chip{position:absolute;z-index:4;background:rgba(255,255,255,0.1);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,0.2);color:#fff;
  font-size:17px;font-weight:600;padding:9px 16px;border-radius:24px;white-space:nowrap;
  animation:eduFloat2 5s ease-in-out infinite;}
.edu-float-chip-1{top:12%;right:5%;}
.edu-float-chip-2{bottom:14%;left:3%;animation-delay:1.5s;}

@keyframes eduFloat{0%,100%{transform:translateY(0);}50%{transform:translateY(-14px);}}
@keyframes eduFloat2{0%,100%{transform:translateY(0);}50%{transform:translateY(-10px);}}
@keyframes eduFadeInUp{from{opacity:0;transform:translateY(26px);}to{opacity:1;transform:translateY(0);}}

/* ── STATS STRIP ────────────────────────────────────────────────────── */
.edu-stats{background:#fff;border-bottom:1px solid #E0E3FF;padding:34px 56px;
  display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;}
.edu-stat{text-align:center;flex:1;min-width:120px;}
.edu-stat-icon{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;
  justify-content:center;font-size:22px;margin:0 auto 10px;}
.edu-stat-num{font-size:35px;font-weight:900;letter-spacing:-1px;line-height:1;}
.edu-stat-label{font-size:17px;color:#6B7280;font-weight:500;margin-top:5px;}
.edu-divider{width:1px;height:44px;background:#E0E3FF;flex-shrink:0;}

/* ── SECTION SHELL ──────────────────────────────────────────────────── */
.edu-section{padding:80px 56px 40px;}
.edu-eyebrow{font-size:17px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
  margin:0 0 12px;color:#3A4EFB;}
.edu-h2{font-size:43px;font-weight:900;color:#252943!important;letter-spacing:-1.2px;
  margin:0 0 14px!important;padding:0!important;line-height:1.15;}
.edu-sub{font-size:20px;color:#6B7280;margin:0 0 48px;max-width:620px;line-height:1.65;}
#how-it-works{scroll-margin-top:88px;}

/* ── HOW IT WORKS ───────────────────────────────────────────────────── */
.edu-steps-grid{display:grid;grid-template-columns:1fr 30px 1fr 30px 1fr 30px 1fr;
  align-items:stretch;}
.edu-step-arrow{display:flex;align-items:center;justify-content:center;
  color:#C7CCF5;font-size:25px;font-weight:600;}
.edu-step-card{background:#fff;border:1px solid #E0E3FF;border-radius:18px;
  padding:28px 22px;box-shadow:0 4px 18px rgba(58,78,251,0.06);height:100%;
  transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;}
.edu-step-card:hover{transform:translateY(-6px);
  box-shadow:0 22px 40px rgba(58,78,251,0.16);border-color:transparent;}
.edu-step-icon-wrap{width:46px;height:46px;border-radius:12px;display:flex;
  align-items:center;justify-content:center;font-size:26px;margin-bottom:16px;}
.edu-step-num{font-size:17px;font-weight:800;letter-spacing:2px;
  text-transform:uppercase;margin-bottom:10px;}
.edu-step-title{font-size:21px;font-weight:700;color:#252943;
  margin-bottom:6px;line-height:1.3;}
.edu-step-desc{font-size:17px;color:#9199B5;line-height:1.4;font-weight:500;}

/* ── FEATURE DOMAINS (bento grid) ──────────────────────────────────── */
.edu-domain-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:20px;align-items:start;}
.edu-span-2{grid-column:span 2;}
.edu-span-3{grid-column:span 3;}
.edu-domain-card{display:flex;flex-direction:column;background:#fff;border:1px solid #E0E3FF;
  border-radius:18px;padding:26px 22px 0;box-shadow:0 4px 18px rgba(58,78,251,0.06);
  overflow:hidden;transition:transform .25s ease, box-shadow .25s ease;}
.edu-domain-card:hover{transform:translateY(-6px);box-shadow:0 22px 40px rgba(58,78,251,0.14);}
.edu-domain-icon-wrap{width:48px;height:48px;border-radius:14px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:27px;margin-bottom:16px;}
.edu-domain-title{font-size:20px;font-weight:800;margin-bottom:14px;letter-spacing:0.2px;}
.edu-domain-feats{display:flex;flex-direction:column;}
.edu-domain-feat{font-size:18px;color:#374151;padding:8px 0;
  border-bottom:1px solid #EEF0FA;line-height:1.4;}
.edu-domain-feat:last-child{border-bottom:none;}
.edu-score-footnote{margin-top:2px;}
.edu-score-footnote summary{list-style:none;cursor:pointer;font-size:16px;font-weight:700;
  letter-spacing:0.3px;padding:10px 0;display:flex;align-items:center;gap:6px;
  color:inherit;}
.edu-score-footnote summary::-webkit-details-marker{display:none;}
.edu-score-footnote summary::after{content:"›";margin-left:auto;font-size:20px;
  transition:transform .2s ease;}
.edu-score-footnote[open] summary::after{transform:rotate(90deg);}
.edu-score-footnote-body{font-size:17px;color:#6B7280;line-height:1.65;
  padding:0 0 16px;}
.edu-domain-accent{margin-top:auto;margin-left:-22px;margin-right:-22px;
  height:4px;margin-top:22px;}

/* ── CTA BANNER ─────────────────────────────────────────────────────── */
.edu-cta-banner{position:relative;overflow:hidden;margin:16px 56px 80px;padding:56px 60px;
  background:linear-gradient(135deg,#3A4EFB 0%,#33A4FA 100%);
  border-radius:24px;display:flex;align-items:center;
  justify-content:space-between;flex-wrap:wrap;gap:28px;
  box-shadow:0 30px 60px rgba(58,78,251,0.28);}
.edu-cta-banner::before{content:"";position:absolute;inset:0;z-index:0;
  background-image:radial-gradient(rgba(255,255,255,0.14) 1.5px, transparent 1.5px);
  background-size:22px 22px;opacity:0.5;pointer-events:none;}
.edu-cta-banner::after{content:"";position:absolute;width:340px;height:340px;z-index:0;
  background:rgba(227,255,59,0.22);border-radius:50%;filter:blur(90px);
  top:-140px;right:-60px;pointer-events:none;}
.edu-cta-banner>div{position:relative;z-index:1;}
.edu-cta-eyebrow{color:rgba(255,255,255,0.65);font-size:17px;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;margin:0 0 8px;}
.edu-cta-h3{color:#fff!important;font-size:35px;font-weight:900;letter-spacing:-0.8px;
  margin:0 0 8px!important;padding:0!important;line-height:1.2;}
.edu-cta-sub{color:rgba(255,255,255,0.75);font-size:20px;margin:0;}
.edu-cta-btns{display:flex;gap:12px;flex-shrink:0;flex-wrap:wrap;}
.edu-btn-yellow{background:#E3FF3B;color:#252943!important;padding:14px 28px;border-radius:12px;
  font-weight:800;font-size:20px;text-decoration:none!important;white-space:nowrap;display:inline-block;
  transition:transform .2s ease;}
.edu-btn-yellow:hover{transform:translateY(-2px);}
.edu-btn-outline{background:rgba(255,255,255,0.14);color:#fff!important;padding:14px 28px;
  border-radius:12px;font-weight:700;font-size:20px;text-decoration:none!important;display:inline-block;
  border:1px solid rgba(255,255,255,0.3);white-space:nowrap;transition:transform .2s ease, background .2s ease;}
.edu-btn-outline:hover{transform:translateY(-2px);background:rgba(255,255,255,0.2);}

/* ── FOOTER ─────────────────────────────────────────────────────────────────── */
.edu-footer{background:#252943;padding:40px 56px 28px;}
.edu-footer-top{display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:16px;padding-bottom:22px;border-bottom:1px solid rgba(255,255,255,0.08);
  margin-bottom:20px;}
.edu-footer-brand{color:#fff;font-weight:800;font-size:17px;display:flex;align-items:center;gap:8px;}
.edu-footer-stack{display:flex;gap:8px;flex-wrap:wrap;}
.edu-footer-pill{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
  color:#A8B8D8;font-size:11px;font-weight:600;padding:5px 12px;border-radius:20px;}
.edu-footer-credit{color:#7A7F9A;font-size:15px;text-align:center;line-height:1.6;}

@media(max-width:1024px){
  .edu-domain-grid{grid-template-columns:repeat(2,1fr);}
  .edu-span-2,.edu-span-3{grid-column:span 1;}
  .edu-domain-card:nth-child(5){grid-column:span 2;}
  .edu-steps-grid{grid-template-columns:1fr 1fr;gap:16px;}
  .edu-step-arrow{display:none;}
}
@media(max-width:768px){
  .edu-nav{padding:0 20px;flex-wrap:wrap;min-height:60px;gap:8px;}
  .edu-navlinks{display:none;}
  .edu-hero{grid-template-columns:1fr;}
  .edu-hero-visual{min-height:320px;padding:48px 20px;}
  .edu-float-chip{display:none;}
  .edu-hero-left{padding:48px 24px;}
  .edu-h1-red,.edu-h1-white{font-size:42px;letter-spacing:0.3px;line-height:1.25;}
  .edu-stats{padding:24px 20px;flex-wrap:wrap;gap:16px;}
  .edu-divider{display:none;}
  .edu-stat{min-width:40%;}
  .edu-section{padding:52px 20px 36px;}
  .edu-h2{font-size:32px;}
  .edu-domain-grid{grid-template-columns:1fr;}
  .edu-domain-card:nth-child(5){grid-column:span 1;}
  .edu-cta-banner{margin:0 20px 48px;padding:32px 24px;flex-direction:column;}
  .edu-cta-h3{font-size:25px;}
  .edu-footer{padding:32px 20px 24px;}
  .edu-footer-top{flex-direction:column;align-items:flex-start;}
}
</style>
""", unsafe_allow_html=True)


# ── NAVBAR ─────────────────────────────────────────────────────────────────
st.markdown("""
<nav class="edu-nav">
  <a class="edu-brand" href="#">
    <div class="edu-brand-icon">🛡️</div>
    <div>
      <div class="edu-brand-name">EduGuard</div>
      <div class="edu-brand-sub">Early Dropout Intelligence</div>
    </div>
  </a>
  <div class="edu-navlinks">
    <a href="#" class="active">Home</a>
    <a href="/predictor">Student Predictor</a>
    <a href="/batch">Batch Dashboard</a>
    <a href="/performance">Model Performance</a>
    <a href="/about">About</a>
  </div>
  <div class="edu-live"><span class="edu-live-dot"></span>Live</div>
</nav>
""", unsafe_allow_html=True)


# ── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-hero">
  <div class="edu-hero-left">
    <div class="edu-hero-badge">Predict · Explain · Intervene</div>
    <h1 class="edu-h1-red">Stop Dropout</h1>
    <h1 class="edu-h1-white">Before It Happens</h1>
    <p class="edu-hero-body">
      EduGuard uses
      <span style="color:#E3FF3B;font-weight:600;">XGBoost ensemble learning</span>
      and
      <span style="color:#E3FF3B;font-weight:600;">SHAP explainability</span>
      to identify Nigerian secondary school students at risk of dropping out
      weeks before they leave and gives educators specific, ranked
      interventions to act on.
    </p>
    <div class="edu-chips">
      <div class="edu-chip">🎯 89.7% Recall</div>
      <div class="edu-chip">📊 99.4% AUC-ROC</div>
      <div class="edu-chip">⚡ 10,000 Records</div>
      <div class="edu-chip">🏫 17 Predictors</div>
    </div>
    <div class="edu-hero-btnrow">
      <a href="/predictor" class="edu-cta-btn">Use Student Predictor →</a>
      <a href="#how-it-works" class="edu-cta-btn-ghost">See how it works ↓</a>
    </div>
  </div>
  <div class="edu-hero-visual">
    <svg class="edu-network-svg" viewBox="0 0 600 600" preserveAspectRatio="xMidYMid slice">
      <line x1="70" y1="110" x2="215" y2="195" stroke="rgba(255,255,255,0.14)" stroke-width="1.5"/>
      <line x1="215" y1="195" x2="345" y2="100" stroke="rgba(255,255,255,0.14)" stroke-width="1.5"/>
      <line x1="215" y1="195" x2="165" y2="345" stroke="rgba(255,255,255,0.11)" stroke-width="1.5"/>
      <line x1="345" y1="100" x2="495" y2="175" stroke="rgba(255,255,255,0.11)" stroke-width="1.5"/>
      <line x1="165" y1="345" x2="320" y2="430" stroke="rgba(255,255,255,0.11)" stroke-width="1.5"/>
      <line x1="320" y1="430" x2="475" y2="375" stroke="rgba(255,255,255,0.09)" stroke-width="1.5"/>
      <line x1="475" y1="375" x2="495" y2="175" stroke="rgba(255,255,255,0.07)" stroke-width="1.5"/>
      <line x1="320" y1="430" x2="215" y2="195" stroke="rgba(255,255,255,0.06)" stroke-width="1.5"/>
      <circle cx="70" cy="110" r="5" fill="#33A4FA"/>
      <circle cx="215" cy="195" r="7" fill="#E3FF3B"/>
      <circle cx="345" cy="100" r="4" fill="#3A4EFB"/>
      <circle cx="495" cy="175" r="6" fill="#33A4FA"/>
      <circle cx="165" cy="345" r="5" fill="#3A4EFB"/>
      <circle cx="320" cy="430" r="8" fill="#E3FF3B"/>
      <circle cx="475" cy="375" r="4" fill="#33A4FA"/>
    </svg>
    <div class="edu-float-chip edu-float-chip-1">⚡ SHAP Explainable AI</div>
    <div class="edu-float-chip edu-float-chip-2">🤖 XGBoost Ensemble</div>
    <div class="edu-predict-card">
      <div class="edu-predict-top">
        <div class="edu-predict-id">
          <div class="edu-predict-avatar">🎓</div>
          <div>
            <div class="edu-predict-name">Student STU-2049</div>
            <div class="edu-predict-meta">SS2 · Term 2</div>
          </div>
        </div>
        <div class="edu-predict-risk-badge">HIGH RISK</div>
      </div>
      <div class="edu-predict-score-row">
        <span class="edu-predict-score">78%</span>
        <span class="edu-predict-score-label">dropout risk score</span>
      </div>
      <div class="edu-predict-factors">
        <div class="edu-factor">
          <div class="edu-factor-label"><span>Attendance</span><span>62%</span></div>
          <div class="edu-factor-track"><div class="edu-factor-fill" style="width:62%;background:#FF4B4B;"></div></div>
        </div>
        <div class="edu-factor">
          <div class="edu-factor-label"><span>Failed subjects</span><span>3</span></div>
          <div class="edu-factor-track"><div class="edu-factor-fill" style="width:50%;background:#FFB800;"></div></div>
        </div>
        <div class="edu-factor">
          <div class="edu-factor-label"><span>Distance to school</span><span>14 km</span></div>
          <div class="edu-factor-track"><div class="edu-factor-fill" style="width:38%;background:#33A4FA;"></div></div>
        </div>
      </div>
      <div class="edu-predict-action">
        <span class="edu-predict-action-dot"></span>
        <span><strong>Urgent:</strong>&nbsp;Schedule attendance counselling</span>
      </div>
      <div class="edu-predict-caption">Illustrative example · SHAP-ranked factors</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── STATS STRIP ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="edu-stats">
  <div class="edu-stat">
    <div class="edu-stat-icon" style="background:{BLUE1}18;">🗂️</div>
    <div class="edu-stat-num" style="color:{BLUE1};">10,000</div>
    <div class="edu-stat-label">Student records</div>
  </div>
  <div class="edu-divider"></div>
  <div class="edu-stat">
    <div class="edu-stat-icon" style="background:{BLUE2}18;">🧬</div>
    <div class="edu-stat-num" style="color:{BLUE2};">17</div>
    <div class="edu-stat-label">Predictive features</div>
  </div>
  <div class="edu-divider"></div>
  <div class="edu-stat">
    <div class="edu-stat-icon" style="background:{PURPLE}18;">🗃️</div>
    <div class="edu-stat-num" style="color:{PURPLE};">5</div>
    <div class="edu-stat-label">Feature domains</div>
  </div>
  <div class="edu-divider"></div>
  <div class="edu-stat">
    <div class="edu-stat-icon" style="background:{RED}18;">🎯</div>
    <div class="edu-stat-num" style="color:{RED};">89.7%</div>
    <div class="edu-stat-label">Recall (XGBoost)</div>
  </div>
  <div class="edu-divider"></div>
  <div class="edu-stat">
    <div class="edu-stat-icon" style="background:{GREEN}18;">📈</div>
    <div class="edu-stat-num" style="color:{GREEN};">99.4%</div>
    <div class="edu-stat-label">AUC-ROC</div>
  </div>
  <div class="edu-divider"></div>
  <div class="edu-stat">
    <div class="edu-stat-icon" style="background:{AMBER}18;">⚖️</div>
    <div class="edu-stat-num" style="color:{AMBER};">3</div>
    <div class="edu-stat-label">Models compared</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── HOW IT WORKS ─────────────────────────────────────────────────────────────
# Per the brief: icon + title + SHORT one-line label only, no paragraph text.
steps = [
    (BLUE1, "01", "📝", "Enter student data", "Manual entry or class CSV upload"),
    (BLUE2, "02", "🤖", "XGBoost predicts risk", "0–100% score, High / Medium / Low"),
    (PURPLE, "03", "🔍", "SHAP explains why", "Feature-level contribution ranking"),
    (GREEN, "04", "📋", "Interventions delivered", "Ranked actions, owner assigned"),
]

step_cards = []
for clr, num, icon, title, desc in steps:
    step_cards.append(f"""<div class="edu-step-card">
      <div class="edu-step-icon-wrap" style="background:{clr}18;border:1px solid {clr}33;">{icon}</div>
      <div class="edu-step-num" style="color:{clr};">Step {num}</div>
      <div class="edu-step-title">{title}</div>
      <div class="edu-step-desc">{desc}</div>
    </div>""")

steps_html = '<div class="edu-step-arrow">→</div>'.join(step_cards)

st.markdown(f"""
<div class="edu-section" id="how-it-works">
  <p class="edu-eyebrow">How it works</p>
  <h2 class="edu-h2">From student data to educator action</h2>
  <p class="edu-sub">
    Four steps connect a student's existing school record to a specific,
    evidence-based action that a teacher or counsellor can take today.
  </p>
  <div class="edu-steps-grid">
    {steps_html}
  </div>
</div>
""", unsafe_allow_html=True)


# ── FEATURE DOMAINS ───────────────────────────────────────────────────────────
# Per the brief: cards show icon, domain name, and feature list only — no
# extra description text in the card body. Longer methodology notes for the
# scored domains move into a collapsible <details> footnote instead of an
# always-visible paragraph box.
domains = [
    (BLUE1, "👤", "Demographic", "edu-span-2",
     ["Gender", "Age at enrolment", "Distance to school (km)"], None),
    (PURPLE, "💰", "Socioeconomic", "edu-span-2",
     ["Household income band", "Fee payment status",
      "Parental education level", "Parental involvement"], None),
    (BLUE2, "📚", "Academic", "edu-span-2",
     ["Attendance rate (%)", "Avg academic score (%)", "Failed subjects count"],
     ("How scores are collected",
      "Attendance = days present ÷ total school days. Academic score = mean "
      "CA + exam score across subjects. Failed subjects = count below the 40% "
      "pass mark. All three come directly from existing school records.")),
    (AMBER, "🧠", "Cognitive", "edu-span-3",
     ["Cognitive assessment score", "Learning pace", "Study habit score"],
     ("How scores are generated",
      "Cognitive score (0–100): school-administered reasoning test, similar "
      "to a CAT or Raven's Matrices. Learning pace: teacher-assigned from "
      "classroom observation. Study habit score (0–100): homework completion "
      "rate and independent study consistency from teacher records.")),
    (GREEN, "🏃", "Psychomotor", "edu-span-3",
     ["Practical subject score", "PE participation", "Psychomotor development score"],
     ("How scores are generated",
      "Practical score (0–100): average across Agric Science, Home Economics, "
      "Technical Drawing, and Basic Technology. PE participation: teacher-rated "
      "engagement, a proxy for school belonging. Psychomotor development "
      "(0–100): composite teacher rating per the FME (2020) assessment rubric.")),
]

domain_cards = []
for clr, icon, title, span_cls, feats, score_info in domains:
    # NOTE: every fragment below is built with zero embedded newlines. A
    # blank (or whitespace-only) line inside a raw-HTML block passed to
    # st.markdown(unsafe_allow_html=True) makes Streamlit's Markdown parser
    # end the HTML block early and dump the remaining tags as literal text —
    # which is exactly what happened here when score_html was "" for cards
    # with no info box. Single-line concatenation sidesteps that entirely.
    feat_html = "".join(f'<div class="edu-domain-feat">{f}</div>' for f in feats)
    if score_info:
        score_html = (
            f'<details class="edu-score-footnote">'
            f'<summary style="color:{clr};">📐 {score_info[0]}</summary>'
            f'<div class="edu-score-footnote-body">{score_info[1]}</div>'
            f'</details>'
        )
    else:
        score_html = ""
    card = (
        f'<div class="edu-domain-card {span_cls}">'
        f'<div class="edu-domain-icon-wrap" style="background:{clr}18;border:1px solid {clr}33;">{icon}</div>'
        f'<div class="edu-domain-title" style="color:{clr};">{title}</div>'
        f'<div class="edu-domain-feats">{feat_html}</div>'
        f'{score_html}'
        f'<div class="edu-domain-accent" style="background:{clr};"></div>'
        f'</div>'
    )
    domain_cards.append(card)

domains_html = "".join(domain_cards)

st.markdown(f"""
<div class="edu-section" style="padding-top:64px;">
  <p class="edu-eyebrow">Feature coverage</p>
  <h2 class="edu-h2">17 predictors across 5 domains</h2>
  <p class="edu-sub">
    Each domain captures a distinct dimension of student dropout risk.
    Academic, Cognitive, and Psychomotor domains score assessments, tap
    "How scores are generated" inside a card for the methodology.
  </p>
  <div class="edu-domain-grid">
    {domains_html}
  </div>
</div>
""", unsafe_allow_html=True)


# ── CTA BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-cta-banner">
  <div>
    <p class="edu-cta-eyebrow">Get started</p>
    <h3 class="edu-cta-h3">Ready to identify at-risk students?</h3>
    <p class="edu-cta-sub">Use the navbar above to access any section of EduGuard.</p>
  </div>
  <div class="edu-cta-btns">
    <a href="/predictor" class="edu-btn-yellow">🎯 Student Predictor</a>
    <a href="/batch" class="edu-btn-outline">📊 Batch Dashboard</a>
  </div>
</div>
""", unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-footer">
  <div class="edu-footer-top">
    <div class="edu-footer-brand"><span>🛡️</span> EduGuard</div>
    <div class="edu-footer-stack">
      <span class="edu-footer-pill">Streamlit</span>
      <span class="edu-footer-pill">XGBoost</span>
      <span class="edu-footer-pill">SHAP</span>
    </div>
  </div>
  <div class="edu-footer-credit">
    EduGuard — Early Dropout Intelligence System &nbsp;·&nbsp;
  </div>
</div>
""", unsafe_allow_html=True)