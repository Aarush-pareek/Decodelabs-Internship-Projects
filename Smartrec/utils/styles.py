"""Global CSS for the SmartRec dashboard (restrained, single-accent palette)."""

CSS = """
<style>
:root { --accent:#4F46E5; --ink:#0F172A; --muted:#64748B; --line:#E2E8F0; --soft:#EEF2FF; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
h1, h2, h3 { letter-spacing: -0.01em; color: var(--ink); }

.brand { font-size: 1.35rem; font-weight: 700; color: var(--ink); margin-bottom: 0; }
.brand span { color: var(--accent); }
.tagline { color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }

.hero { background:#FFFFFF; border:1px solid var(--line); border-radius:14px;
        padding:1.6rem 1.8rem; margin-bottom:1.2rem; }
.hero h1 { font-size:1.7rem; margin:0 0 .4rem 0; }
.hero p { color:var(--muted); margin:0; font-size:1rem; }

.steps { display:flex; gap:.75rem; flex-wrap:wrap; margin:.5rem 0 1.2rem 0; }
.step { flex:1; min-width:150px; background:#FFFFFF; border:1px solid var(--line);
        border-radius:12px; padding:.8rem 1rem; }
.step b { color:var(--accent); font-size:.8rem; letter-spacing:.04em; }
.step div { color:var(--ink); font-size:.92rem; margin-top:.15rem; }

.chip { display:inline-block; padding:.12rem .6rem; margin:0 .3rem .3rem 0; border-radius:999px;
        background:#F1F5F9; color:#334155; font-size:.78rem; border:1px solid var(--line); }
.chip-pref { background:var(--soft); color:#3730A3; border-color:#C7D2FE; font-size:.85rem; }
.chip-match { background:var(--soft); color:#3730A3; border-color:#C7D2FE; }
.muted { color:var(--muted); font-size:.85rem; }
.section-label { font-size:.8rem; font-weight:600; color:var(--muted); text-transform:uppercase;
                 letter-spacing:.06em; margin:.9rem 0 .35rem 0; }

.card { background:#FFFFFF; border:1px solid var(--line); border-radius:14px;
        padding:1.1rem 1.3rem; margin-top:.9rem; }
.card-top { display:flex; align-items:flex-start; gap:.9rem; }
.rank { width:2rem; height:2rem; border-radius:8px; background:var(--accent); color:#FFFFFF;
        display:flex; align-items:center; justify-content:center; font-weight:700; flex:none; }
.card-title { flex:1; }
.title { font-size:1.1rem; font-weight:650; color:var(--ink); }
.year { color:var(--muted); font-weight:400; font-size:.95rem; }
.liked { background:#DCFCE7; color:#166534; font-size:.72rem; padding:.1rem .5rem;
         border-radius:999px; margin-left:.4rem; }
.match { text-align:right; flex:none; }
.match-value { font-size:1.5rem; font-weight:700; color:var(--accent); line-height:1.1; }
.match-label { font-size:.75rem; color:var(--muted); }
.bar { height:6px; background:#E2E8F0; border-radius:999px; margin:.7rem 0 .6rem 0; }
.bar-fill { height:6px; background:var(--accent); border-radius:999px; }
.meta-line { color:var(--muted); font-size:.85rem; }
.overview { color:#334155; font-size:.92rem; margin:.5rem 0 .2rem 0; }
.matched-label { font-size:.75rem; font-weight:600; color:var(--muted); margin-top:.5rem;
                 text-transform:uppercase; letter-spacing:.05em; }
.why { background:#F8FAFC; border-left:3px solid var(--accent); border-radius:6px;
       padding:.6rem .8rem; margin-top:.6rem; font-size:.9rem; color:#1E293B; }
div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--line);
                              border-radius:12px; padding:.7rem 1rem; }
</style>
"""
