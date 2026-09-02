"""The design system: Liquid Glass, in light and dark.

The look borrows Apple's Liquid Glass language, which rests on four things
working together. Any one of them alone reads as a flat card with a blur
filter:

1. **An ambient ground.** Glass only looks like glass when there is something
   behind it worth refracting. A soft mesh of colour blobs sits fixed behind
   the whole app; every panel samples it.
2. **Blur plus saturation.** ``backdrop-filter: blur() saturate()`` - the
   saturation boost is what makes the material feel like glass rather than
   frosted plastic, because real glass concentrates the colour behind it.
3. **A specular edge.** A bright inset hairline along the top and a faint rim
   elsewhere, so light appears to catch the lip of the panel.
4. **Concentric radii.** Nested corners share a centre: an inner element's
   radius is the outer radius minus its inset. Matching radii instead makes
   the nesting look accidental.

Both modes are generated from one function so a token can never drift between
them, and motion is disabled wholesale under ``prefers-reduced-motion``.
"""

from __future__ import annotations

# Palettes. Only these differ between modes; every rule below is shared.
PALETTES = {
    "light": {
        "ground": "#eef1f8",
        "blob1": "rgba(120,160,255,.42)",
        "blob2": "rgba(255,170,190,.38)",
        "blob3": "rgba(150,230,215,.34)",
        "blob4": "rgba(210,180,255,.34)",
        "glass": "rgba(255,255,255,.58)",
        "glass_hi": "rgba(255,255,255,.86)",
        "glass_edge": "rgba(255,255,255,.75)",
        "glass_rim": "rgba(15,25,50,.09)",
        "sheen": "rgba(255,255,255,.92)",
        "shadow": "0 10px 34px -14px rgba(24,39,75,.28), 0 3px 10px -6px rgba(24,39,75,.18)",
        "shadow_hi": "0 20px 48px -18px rgba(24,39,75,.40), 0 6px 16px -8px rgba(24,39,75,.24)",
        "text": "#171a21",
        "dim": "#5b6478",
        "faint": "#8a92a6",
        "accent": "#0a6cff",
        "accent_soft": "rgba(10,108,255,.12)",
        "gain": "#0f9d58",
        "gain_soft": "rgba(15,157,88,.13)",
        "loss": "#e0413c",
        "loss_soft": "rgba(224,65,60,.12)",
        "warn": "#c8830a",
        "warn_soft": "rgba(200,131,10,.14)",
        "track": "rgba(20,32,60,.10)",
        "grid": "rgba(20,32,60,.10)",
    },
    "dark": {
        "ground": "#06070c",
        "blob1": "rgba(58,110,255,.30)",
        "blob2": "rgba(190,80,255,.22)",
        "blob3": "rgba(30,200,190,.18)",
        "blob4": "rgba(255,110,150,.16)",
        "glass": "rgba(255,255,255,.055)",
        "glass_hi": "rgba(255,255,255,.10)",
        "glass_edge": "rgba(255,255,255,.14)",
        "glass_rim": "rgba(0,0,0,.5)",
        "sheen": "rgba(255,255,255,.22)",
        "shadow": "0 12px 38px -16px rgba(0,0,0,.9), 0 3px 10px -6px rgba(0,0,0,.7)",
        "shadow_hi": "0 22px 54px -20px rgba(0,0,0,.95), 0 6px 18px -8px rgba(0,0,0,.8)",
        "text": "#f2f4f9",
        "dim": "#a3abbd",
        "faint": "#727b8f",
        "accent": "#3d95ff",
        "accent_soft": "rgba(61,149,255,.16)",
        "gain": "#3ddc84",
        "gain_soft": "rgba(61,220,132,.14)",
        "loss": "#ff6b6b",
        "loss_soft": "rgba(255,107,107,.14)",
        "warn": "#ffc857",
        "warn_soft": "rgba(255,200,87,.14)",
        "track": "rgba(255,255,255,.10)",
        "grid": "rgba(255,255,255,.08)",
    },
}


def css(mode: str = "light") -> str:
    """The full stylesheet for one mode."""
    p = PALETTES.get(mode, PALETTES["light"])
    return f"""
<style>
  /* Inter, served from this machine rather than a font CDN: a webfont request
     would be the one thing on the page that phones out, and it would leave the
     app looking wrong offline. The variable file carries every weight in
     352KB, so one request covers the whole interface.

     `font-feature-settings` is the reason Inter earns its place here over the
     system font: `tnum` locks every digit to the same width so columns of
     prices align on the decimal, `cv05`/`cv08` give the l and t disambiguated
     shapes, and `ss03` uses the flat-topped 1 that is far easier to tell from
     an I in a ticker. */
  @font-face {{
    font-family: "InterVar";
    src: url("/app/static/fonts/InterVariable.woff2") format("woff2");
    font-weight: 100 900;
    font-style: normal;
    font-display: swap;
  }}

  :root {{
    --font: "InterVar", -apple-system, BlinkMacSystemFont, "Segoe UI",
            "Helvetica Neue", Arial, sans-serif;
    --ground: {p["ground"]};
    --glass: {p["glass"]};
    --glass-hi: {p["glass_hi"]};
    --glass-edge: {p["glass_edge"]};
    --glass-rim: {p["glass_rim"]};
    --sheen: {p["sheen"]};
    --shadow: {p["shadow"]};
    --shadow-hi: {p["shadow_hi"]};
    --text: {p["text"]};
    --dim: {p["dim"]};
    --faint: {p["faint"]};
    --accent: {p["accent"]};
    --accent-soft: {p["accent_soft"]};
    --gain: {p["gain"]};
    --gain-soft: {p["gain_soft"]};
    --loss: {p["loss"]};
    --loss-soft: {p["loss_soft"]};
    --warn: {p["warn"]};
    --warn-soft: {p["warn_soft"]};
    --track: {p["track"]};

    /* Concentric radii: an inner corner is the outer minus its inset. */
    --r-xl: 26px; --r-lg: 20px; --r-md: 14px; --r-sm: 10px;
    --blur: 30px;
    --spring: cubic-bezier(.34, 1.4, .64, 1);
    --ease: cubic-bezier(.32, .72, 0, 1);
  }}

  /* ---------- Ambient ground ------------------------------------------- */

  .stApp {{ background: var(--ground); }}

  /* The colour behind the glass. Fixed, so panels refract a stable field
     rather than a scrolling one. */
  .stApp::before {{
    content: ""; position: fixed; inset: -12%; z-index: 0; pointer-events: none;
    background:
      radial-gradient(46vw 42vw at 12% 8%,  {p["blob1"]}, transparent 62%),
      radial-gradient(42vw 40vw at 88% 4%,  {p["blob2"]}, transparent 60%),
      radial-gradient(50vw 44vw at 78% 88%, {p["blob3"]}, transparent 62%),
      radial-gradient(40vw 38vw at 20% 92%, {p["blob4"]}, transparent 60%);
    filter: blur(26px) saturate(120%);
  }}
  .stApp > * {{ position: relative; z-index: 1; }}

  /* ---------- Type ------------------------------------------------------ */

  /* Applied to everything, including the widgets Streamlit renders itself -
     a half-converted interface reads worse than one consistent fallback. */
  html, body, [class*="css"], .stApp, button, input, textarea, select,
  [data-testid="stMarkdownContainer"], [class*="st-emotion"],
  [data-baseweb],
  /* Headings carry an explicit font-family from Streamlit, so inheritance
     never reaches them - without naming them the titles stay on Source Sans
     while everything around them changes. */
  h1, h2, h3, h4, h5, h6, p, span, div, li, a, label, td, th, caption {{
    font-family: var(--font) !important;
    font-feature-settings: "tnum" 1, "cv05" 1, "cv08" 1, "ss03" 1;
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  }}
  html, body, [class*="css"] {{ color: var(--text); }}

  /* Material Symbols must keep its own family or every icon becomes a word. */
  [data-testid="stIconMaterial"], .material-symbols-rounded,
  [class*="material-symbols"] {{
    font-family: "Material Symbols Rounded" !important;
    font-feature-settings: normal;
  }}
  .block-container {{ padding-top: 2.4rem; padding-bottom: 6rem; max-width: 1320px; }}

  h1, h2, h3, h4, p, span, div, label, li {{ color: var(--text); }}
  h1 {{ font-size: 2rem !important; font-weight: 680 !important;
       letter-spacing: -.032em; margin: 0 0 .1rem !important; }}
  h3 {{ font-size: 1.04rem !important; font-weight: 620 !important;
       letter-spacing: -.018em; }}
  h2 {{
    font-size: .72rem !important; font-weight: 640 !important;
    text-transform: uppercase; letter-spacing: .12em;
    color: var(--faint) !important; margin: 2.5rem 0 .9rem !important;
  }}
  .muted {{ color: var(--faint); font-size: .84rem; line-height: 1.6; }}
  hr {{ border-color: var(--glass-rim) !important; margin: 1.8rem 0 !important; }}
  code {{
    background: var(--accent-soft) !important; color: var(--accent) !important;
    padding: .1em .4em !important; border-radius: 5px !important;
    font-size: .87em !important;
  }}

  [data-testid="stMetricValue"], .num, .stat-value, .row-num, .meter-val,
  .ring-num, [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

  /* ---------- The glass material ---------------------------------------- */

  .glass, .stat, .finding, .rows, div[data-testid="stExpander"] details,
  [data-testid="stDataFrame"], [data-testid="stAlert"] {{
    background: var(--glass);
    -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
    backdrop-filter: blur(var(--blur)) saturate(180%);
    border: 1px solid var(--glass-edge);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow), inset 0 1px 0 var(--sheen);
  }}

  /* ---------- Stat cards ------------------------------------------------- */

  /* Two competing constraints: a currency figure must not wrap, and a row of
     five must not spill to 4 + 1. A fixed minimum can only satisfy one at a
     time, so the columns are fractional and the *type* scales down instead -
     the row stays intact and the figures shrink to fit. */
  .stat-grid {{
    display: grid; gap: .7rem; margin: .2rem 0 .4rem;
    grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
  }}
  /* Below roughly 1100px the row would crush; wrap to two lines there. */
  @media (max-width: 1100px) {{
    .stat-grid {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
  }}
  .stat-value {{
    /* clamp() keeps the figure legible while letting it shrink rather than
       wrap or clip when the card is narrow. */
    font-size: clamp(1.02rem, 1.45vw, 1.42rem) !important;
  }}
  .stat {{
    position: relative; overflow: hidden; padding: .95rem 1.05rem 1rem;
    transition: transform .38s var(--spring), box-shadow .32s var(--ease),
                background .32s var(--ease);
  }}
  /* Specular sweep across the top lip, brightest at the left where the
     ambient light sits. */
  .stat::after {{
    content: ""; position: absolute; inset: 0 0 auto 0; height: 42%;
    background: linear-gradient(180deg, var(--glass-hi), transparent);
    opacity: .55; pointer-events: none;
  }}
  .stat:hover {{
    transform: translateY(-3px) scale(1.012);
    box-shadow: var(--shadow-hi), inset 0 1px 0 var(--sheen);
    background: var(--glass-hi);
  }}
  .stat-label {{
    position: relative; z-index: 1; font-size: .69rem; font-weight: 620;
    letter-spacing: .06em; text-transform: uppercase; color: var(--faint);
    line-height: 1.3; min-height: 1.8em;
  }}
  .stat-value {{
    position: relative; z-index: 1; font-size: 1.42rem; font-weight: 680;
    letter-spacing: -.028em; margin-top: .35rem; line-height: 1.18;
    /* A figure must never break across lines - "$5,481.6" above "0" is not a
       number. Inter runs slightly wider than the previous face, which is what
       exposed this. */
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .stat-value.up {{ color: var(--gain); }}
  .stat-value.down {{ color: var(--loss); }}
  .stat-value.sm {{ font-size: 1.1rem; }}
  .stat-delta {{ position: relative; z-index: 1; font-size: .785rem;
                 margin-top: .25rem; color: var(--dim); font-weight: 560; }}
  .stat-delta.up {{ color: var(--gain); }} .stat-delta.down {{ color: var(--loss); }}
  .stat-sub {{ position: relative; z-index: 1; font-size: .72rem;
               color: var(--faint); margin-top: .2rem; }}

  /* ---------- Verdict --------------------------------------------------- */

  .verdict {{
    display: inline-flex; align-items: center; padding: .5rem 1.1rem;
    border-radius: 999px; font-weight: 680; font-size: .92rem;
    letter-spacing: .01em; border: 1px solid var(--glass-edge);
    -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
    backdrop-filter: blur(var(--blur)) saturate(180%);
    box-shadow: var(--shadow), inset 0 1px 0 var(--sheen);
  }}
  .verdict.buy  {{ color: var(--gain); background: var(--gain-soft); }}
  .verdict.sell {{ color: var(--loss); background: var(--loss-soft); }}
  .verdict.hold {{ color: var(--warn); background: var(--warn-soft); }}
  .verdict-meta {{ font-size: .745rem; color: var(--faint); margin-top: .45rem;
                   text-align: right; }}

  .hero-name {{ display: flex; align-items: baseline; gap: .8rem; flex-wrap: wrap; }}
  .hero-price {{ font-size: 1.7rem; font-weight: 640; letter-spacing: -.03em; }}
  .hero-sub {{ color: var(--faint); font-size: .87rem; margin-top: .3rem; }}

  /* ---------- Rings & meters --------------------------------------------- */

  .ring-row {{ display: flex; gap: 1.8rem; flex-wrap: wrap; align-items: center;
               margin: .2rem 0 .5rem; }}
  .ring-item {{ display: flex; align-items: center; gap: .8rem; }}
  .ring {{
    --pct: 0; --ring-col: var(--accent);
    width: 62px; height: 62px; border-radius: 50%; position: relative; flex: none;
    background: conic-gradient(var(--ring-col) calc(var(--pct) * 1%), var(--track) 0);
    animation: sweep .9s var(--ease) both;
    box-shadow: inset 0 1px 0 var(--sheen), var(--shadow);
  }}
  .ring::after {{
    content: ""; position: absolute; inset: 6px; border-radius: 50%;
    background: var(--glass);
    -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
    box-shadow: inset 0 1px 0 var(--sheen);
  }}
  .ring-num {{ position: absolute; inset: 0; display: grid; place-items: center;
               font-size: .92rem; font-weight: 700; z-index: 1; }}
  .ring-label {{ font-size: .705rem; color: var(--faint); text-transform: uppercase;
                 letter-spacing: .07em; font-weight: 620; }}
  .ring-cap {{ font-size: .81rem; color: var(--dim); margin-top: .15rem; }}
  @keyframes sweep {{ from {{ --pct: 0; }} }}
  @property --pct {{ syntax: "<number>"; inherits: false; initial-value: 0; }}

  .meter {{ margin: .2rem 0 .3rem; }}
  .meter-head {{ display: flex; justify-content: space-between; align-items: baseline;
                 font-size: .72rem; color: var(--faint); margin-bottom: .35rem;
                 text-transform: uppercase; letter-spacing: .07em; font-weight: 620; }}
  .meter-val {{ font-size: .98rem; color: var(--text); font-weight: 660; }}
  .meter-track {{ height: 7px; border-radius: 999px; background: var(--track);
                  overflow: hidden; box-shadow: inset 0 1px 2px var(--glass-rim); }}
  .meter-fill {{ height: 100%; border-radius: 999px; transform-origin: left;
                 animation: grow .75s var(--ease) both .1s; }}

  /* ---------- Findings ---------------------------------------------------- */

  .finding {{
    display: flex; gap: .85rem; padding: .9rem 1.05rem; margin-bottom: .6rem;
    border-radius: var(--r-md);
    transition: transform .34s var(--spring), box-shadow .3s var(--ease);
  }}
  .finding:hover {{ transform: translateX(2px); box-shadow: var(--shadow-hi),
                    inset 0 1px 0 var(--sheen); }}
  .finding::before {{
    content: ""; width: 3px; border-radius: 999px; flex: none;
    background: var(--accent);
  }}
  .finding.critical::before {{ background: var(--loss); }}
  .finding.warning::before  {{ background: var(--warn); }}
  .finding.good::before     {{ background: var(--gain); }}
  .finding-head {{ font-weight: 640; font-size: .9rem; }}
  .finding-detail {{ font-size: .835rem; color: var(--dim); margin-top: .25rem;
                     line-height: 1.6; }}

  /* ---------- Rows --------------------------------------------------------- */

  .rows {{ overflow: hidden; padding: .25rem; }}
  .row {{
    display: grid; align-items: center; gap: .85rem; position: relative;
    grid-template-columns: 5rem 1fr 6.4rem 5.6rem 6.8rem 5.4rem;
    padding: .66rem .85rem; border-radius: var(--r-sm);
    transition: background .22s var(--ease), transform .3s var(--spring);
  }}
  .row:hover {{ background: var(--glass-hi); transform: scale(1.006); }}
  .row-sym {{ font-weight: 660; font-size: .9rem; letter-spacing: -.015em; }}
  .row-name {{ font-size: .715rem; color: var(--faint); overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }}
  .row-num {{ text-align: right; font-size: .875rem; font-weight: 580; }}
  .row-sub {{ text-align: right; font-size: .705rem; color: var(--faint); }}
  .row-head {{ font-size: .672rem; text-transform: uppercase; letter-spacing: .09em;
               color: var(--faint); font-weight: 660; }}
  .row-head:hover {{ background: none; transform: none; }}
  .row-weight {{ position: absolute; left: .85rem; bottom: .18rem; height: 2px;
                 border-radius: 999px; opacity: .5;
                 background: linear-gradient(90deg, var(--accent), transparent);
                 transform-origin: left; animation: grow .75s var(--ease) both; }}

  .pill {{
    display: inline-block; padding: .18rem .62rem; border-radius: 999px;
    font-size: .69rem; font-weight: 700; letter-spacing: .015em;
    border: 1px solid transparent; white-space: nowrap;
  }}
  .pill.buy  {{ color: var(--gain); background: var(--gain-soft); border-color: var(--gain-soft); }}
  .pill.sell {{ color: var(--loss); background: var(--loss-soft); border-color: var(--loss-soft); }}
  .pill.hold {{ color: var(--warn); background: var(--warn-soft); border-color: var(--warn-soft); }}
  .up {{ color: var(--gain); }} .down {{ color: var(--loss); }}

  /* ---------- Allocation ---------------------------------------------------- */

  .alloc {{ display: flex; height: 34px; border-radius: 999px; overflow: hidden;
            border: 1px solid var(--glass-edge); margin: .3rem 0 .6rem;
            box-shadow: var(--shadow), inset 0 1px 0 var(--sheen); }}
  .alloc-seg {{ position: relative; display: grid; place-items: center;
                font-size: .67rem; font-weight: 700; color: #08101c;
                overflow: hidden; white-space: nowrap;
                transform-origin: left; animation: grow .8s var(--ease) both;
                transition: filter .22s var(--ease); }}
  .alloc-seg:hover {{ filter: brightness(1.15) saturate(1.15); }}

  /* ---------- Sparklines ------------------------------------------------------ */

  .spark {{ display: block; overflow: visible; }}
  .spark path.line {{ fill: none; stroke-width: 1.9; stroke-linecap: round;
                      stroke-linejoin: round; stroke-dasharray: 1000;
                      stroke-dashoffset: 1000; animation: draw 1.2s var(--ease) forwards; }}
  .spark path.area {{ stroke: none; opacity: 0; animation: fade .8s var(--ease) .4s forwards; }}
  @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes fade {{ to {{ opacity: 1; }} }}
  @keyframes grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
  @keyframes rise {{ from {{ opacity: 0; transform: translateY(9px); }}
                     to {{ opacity: 1; transform: none; }} }}

  .block-container > div > div > div > [data-testid="stVerticalBlock"] > div {{
    animation: rise .5s var(--ease) both;
  }}
  .block-container > div > div > div > [data-testid="stVerticalBlock"] > div:nth-child(1) {{ animation-delay: .02s; }}
  .block-container > div > div > div > [data-testid="stVerticalBlock"] > div:nth-child(2) {{ animation-delay: .07s; }}
  .block-container > div > div > div > [data-testid="stVerticalBlock"] > div:nth-child(3) {{ animation-delay: .12s; }}
  .block-container > div > div > div > [data-testid="stVerticalBlock"] > div:nth-child(n+4) {{ animation-delay: .16s; }}

  /* ---------- Streamlit chrome -------------------------------------------------- */

  [data-testid="stMetricValue"] {{ font-size: 1.3rem; font-weight: 660; }}
  [data-testid="stMetricLabel"] {{ font-size: .705rem !important; color: var(--faint) !important;
                                   text-transform: uppercase; letter-spacing: .07em; }}

  .stTabs [data-baseweb="tab-list"] {{
    gap: .3rem; padding: .3rem; border-radius: 999px; border: 1px solid var(--glass-edge);
    background: var(--glass); -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
    backdrop-filter: blur(var(--blur)) saturate(180%);
    box-shadow: inset 0 1px 0 var(--sheen); width: fit-content;
  }}
  .stTabs [data-baseweb="tab"] {{
    padding: .42rem 1rem; font-size: .855rem; font-weight: 580; color: var(--dim);
    border-radius: 999px; transition: all .3s var(--ease);
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
  .stTabs [aria-selected="true"] {{
    color: var(--text) !important; background: var(--glass-hi);
    box-shadow: var(--shadow), inset 0 1px 0 var(--sheen);
  }}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

  div[data-testid="stExpander"] details {{ overflow: hidden; }}
  div[data-testid="stExpander"] summary {{ padding: .7rem .95rem !important; }}
  div[data-testid="stExpander"] summary p {{ font-weight: 600; font-size: .875rem; }}

  .stButton > button {{
    border-radius: 999px; font-weight: 620; font-size: .87rem; padding: .5rem 1.1rem;
    background: var(--glass); color: var(--text); border: 1px solid var(--glass-edge);
    -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
    backdrop-filter: blur(var(--blur)) saturate(180%);
    box-shadow: var(--shadow), inset 0 1px 0 var(--sheen);
    transition: transform .34s var(--spring), box-shadow .3s var(--ease),
                background .3s var(--ease);
  }}
  .stButton > button:hover {{
    transform: translateY(-2px) scale(1.02); background: var(--glass-hi);
    box-shadow: var(--shadow-hi), inset 0 1px 0 var(--sheen); color: var(--accent);
    border-color: var(--accent);
  }}
  .stButton > button:active {{ transform: translateY(0) scale(.985); }}
  .stButton > button[kind="primary"] {{
    background: var(--accent); border-color: transparent; color: #fff;
    box-shadow: 0 8px 24px -10px var(--accent), inset 0 1px 0 rgba(255,255,255,.35);
  }}
  .stButton > button[kind="primary"]:hover {{ color: #fff; filter: brightness(1.06); }}

  .stTextInput input, .stNumberInput input, .stTextArea textarea {{
    border-radius: 999px !important; background: var(--glass) !important;
    border: 1px solid var(--glass-edge) !important; color: var(--text) !important;
    font-size: .885rem !important; padding: .5rem .95rem !important;
    -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px);
    box-shadow: inset 0 1px 0 var(--sheen);
    transition: border-color .28s var(--ease), box-shadow .28s var(--ease);
  }}
  .stTextInput input:focus, .stNumberInput input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-soft), inset 0 1px 0 var(--sheen) !important;
  }}
  .stTextInput input::placeholder {{ color: var(--faint) !important; }}

  [data-baseweb="select"] > div {{
    border-radius: 999px !important; background: var(--glass) !important;
    border-color: var(--glass-edge) !important; color: var(--text) !important;
    -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px);
  }}
  [data-baseweb="popover"] div[role="listbox"], [data-baseweb="menu"] {{
    background: var(--glass) !important; border-radius: var(--r-md) !important;
    border: 1px solid var(--glass-edge) !important;
    -webkit-backdrop-filter: blur(var(--blur)) saturate(180%) !important;
    backdrop-filter: blur(var(--blur)) saturate(180%) !important;
    box-shadow: var(--shadow-hi) !important;
  }}
  [data-baseweb="menu"] li {{ color: var(--text) !important; }}
  [data-baseweb="menu"] li:hover {{ background: var(--accent-soft) !important; }}

  [data-testid="stAlert"] {{ padding: .85rem 1rem; }}
  [data-testid="stFileUploaderDropzone"] {{
    background: var(--glass) !important; border-radius: var(--r-lg) !important;
    border: 1px dashed var(--glass-edge) !important;
    -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px);
  }}

  /* Segmented control (st.segmented_control renders as a button group).
     Without this the unselected option keeps Streamlit's own white fill and
     turns into a blank slab in dark mode. */
  [data-testid="stButtonGroup"] {{
    gap: .18rem; padding: .26rem; border-radius: 999px; width: fit-content;
    background: var(--glass); border: 1px solid var(--glass-edge);
    -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
    backdrop-filter: blur(var(--blur)) saturate(180%);
    box-shadow: inset 0 1px 0 var(--sheen);
  }}
  [data-testid="stButtonGroup"] button {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; color: var(--dim) !important;
    border-radius: 999px !important; padding: .34rem .95rem !important;
    font-size: .82rem !important; font-weight: 600 !important;
    transition: background .28s var(--ease), color .28s var(--ease);
  }}
  [data-testid="stButtonGroup"] button:hover {{
    background: var(--glass-hi) !important; color: var(--text) !important;
    transform: none;
  }}
  [data-testid="stButtonGroup"] button[aria-pressed="true"],
  [data-testid="stButtonGroup"] button[aria-checked="true"],
  [data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {{
    background: var(--glass-hi) !important; color: var(--accent) !important;
    box-shadow: var(--shadow), inset 0 1px 0 var(--sheen) !important;
  }}

  /* Radio group styled as a segmented control, Apple-style. */
  [role="radiogroup"] {{
    gap: .2rem; padding: .28rem; border-radius: 999px; width: fit-content;
    background: var(--glass); border: 1px solid var(--glass-edge);
    -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
    backdrop-filter: blur(var(--blur)) saturate(180%);
    box-shadow: inset 0 1px 0 var(--sheen);
  }}
  [role="radiogroup"] label {{
    padding: .34rem .82rem !important; border-radius: 999px;
    transition: background .28s var(--ease), box-shadow .28s var(--ease);
    font-size: .82rem;
  }}
  [role="radiogroup"] label:hover {{ background: var(--glass-hi); }}
  [role="radiogroup"] label:has(input:checked) {{
    background: var(--glass-hi); box-shadow: var(--shadow), inset 0 1px 0 var(--sheen);
  }}
  [role="radiogroup"] label > div:first-child {{ display: none; }}

  /* ---------- Sidebar ------------------------------------------------------------ */

  section[data-testid="stSidebar"] {{
    background: var(--glass);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    backdrop-filter: blur(40px) saturate(180%);
    border-right: 1px solid var(--glass-edge);
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
  [data-testid="stSidebarNav"] a {{
    border-radius: 999px; margin: 2px .3rem; padding: .4rem .8rem !important;
    transition: background .28s var(--ease), transform .3s var(--spring);
  }}
  [data-testid="stSidebarNav"] a:hover {{ background: var(--glass-hi); transform: translateX(2px); }}
  /* Inactive nav labels inherit Streamlit's base-theme ink, which is the wrong
     colour in whichever mode is not the configured one - they vanished
     entirely in dark. */
  [data-testid="stSidebarNav"] a span, [data-testid="stSidebarNav"] a p,
  [data-testid="stSidebarNav"] li a * {{ color: var(--dim) !important; }}
  [data-testid="stSidebarNav"] a:hover span {{ color: var(--text) !important; }}
  [data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: var(--accent-soft); box-shadow: inset 0 1px 0 var(--sheen);
  }}
  [data-testid="stSidebarNav"] a[aria-current="page"] span,
  [data-testid="stSidebarNav"] a[aria-current="page"] * {{
    color: var(--accent) !important; font-weight: 640;
  }}
  [data-testid="stHeader"] {{ background: transparent; }}

  /* ---------- Streamlit chrome contrast ------------------------------------------------
     config.toml pins Streamlit's own base theme, and it cannot change at
     runtime. Everything Streamlit colours itself - typed input text, header
     icons, the sidebar collapse control, slider labels - therefore has to be
     re-declared here, or half the interface is invisible in whichever mode is
     not the configured one. `-webkit-text-fill-color` is required because
     BaseWeb sets it on inputs and it wins over `color`. */

  .stTextInput input, .stNumberInput input, .stTextArea textarea,
  [data-baseweb="input"] input, [data-baseweb="base-input"] input,
  [data-baseweb="textarea"] textarea, [data-baseweb="select"] input {{
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--accent) !important;
  }}
  .stTextInput input::placeholder, [data-baseweb="input"] input::placeholder {{
    color: var(--faint) !important; -webkit-text-fill-color: var(--faint) !important;
  }}

  /* Streamlit draws most icons as Material Symbols **text ligatures**, not
     SVG, so they take `color` rather than `fill`. Styling only svg left the
     sidebar collapse arrow and the expander chevrons as dark glyphs on a dark
     ground - invisible, and the collapse control is not discoverable without
     it. Both mechanisms are covered here. */
  [data-testid="stIconMaterial"],
  [data-testid="stBaseButton-headerNoPadding"],
  [data-testid="stBaseButton-headerNoPadding"] *,
  [data-testid="stExpander"] [data-testid="stIconMaterial"],
  [data-testid="stSidebarCollapseButton"] *,
  [data-testid="stSidebarCollapsedControl"] * {{
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    opacity: 1 !important;
  }}
  [data-testid="stHeader"] svg, [data-testid="stSidebarCollapseButton"] svg,
  [data-testid="stSidebarCollapsedControl"] svg, [data-testid="stExpanderToggleIcon"],
  [data-testid="stNumberInputStepUp"] svg, [data-testid="stNumberInputStepDown"] svg,
  [data-testid="stToolbar"] svg, [data-testid="stMainMenu"] svg,
  section[data-testid="stSidebar"] svg, .stTabs svg, [data-baseweb="select"] svg {{
    fill: var(--text) !important; color: var(--text) !important;
  }}
  [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {{
    background: var(--glass) !important; border-radius: 999px;
    border: 1px solid var(--glass-edge);
    -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px);
    opacity: 1 !important; visibility: visible !important;
  }}
  [data-testid="stSidebarCollapseButton"]:hover,
  [data-testid="stSidebarCollapsedControl"]:hover {{ background: var(--glass-hi) !important; }}

  /* Widget labels, captions and help text. */
  label, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"],
  [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
  .stSlider label, .stSelectbox label, .stToggle label, .stRadio label {{
    color: var(--text) !important;
  }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
  small {{ color: var(--faint) !important; }}

  /* Sliders: the value bubble and end labels are drawn by BaseWeb. */
  [data-testid="stSlider"] [data-testid="stTickBar"],
  [data-testid="stSlider"] div[role="slider"] + div,
  [data-baseweb="slider"] div {{ color: var(--text) !important; }}
  [data-baseweb="slider"] [role="slider"] {{
    background: var(--accent) !important; border-color: var(--accent) !important;
  }}

  /* Toggles read as dead switches without an explicit on-state. */
  [data-baseweb="checkbox"] span[aria-checked="true"] {{
    background: var(--accent) !important; border-color: var(--accent) !important;
  }}

  /* Alerts: Streamlit tints these for its own base theme, which is unreadable
     in the other one. */
  [data-testid="stAlert"], [data-testid="stAlert"] p,
  [data-testid="stAlertContentInfo"], [data-testid="stAlertContentWarning"],
  [data-testid="stAlertContentError"], [data-testid="stAlertContentSuccess"] {{
    color: var(--text) !important;
  }}
  [data-testid="stAlert"] svg {{ fill: var(--dim) !important; }}

  /* The dataframe is glide-data-grid painting on a canvas, so CSS cannot reach
     its cells - but it reads its palette from these custom properties, which
     CSS can set. Without them the grid keeps config.toml's base theme and
     shows as a bright slab in dark mode. */
  [data-testid="stDataFrame"] {{
    color: var(--text);
    --gdg-bg-cell: transparent;
    --gdg-bg-cell-medium: var(--glass-hi);
    --gdg-bg-header: var(--glass-hi);
    --gdg-bg-header-has-focus: var(--glass-hi);
    --gdg-bg-header-hovered: var(--glass-hi);
    --gdg-bg-bubble: var(--glass-hi);
    --gdg-bg-search-result: var(--warn-soft);
    --gdg-text-dark: var(--text);
    --gdg-text-medium: var(--dim);
    --gdg-text-light: var(--faint);
    --gdg-text-header: var(--faint);
    --gdg-text-header-selected: var(--text);
    --gdg-border-color: var(--glass-edge);
    --gdg-horizontal-border-color: var(--glass-edge);
    --gdg-accent-color: var(--accent);
    --gdg-accent-fg: #ffffff;
    --gdg-accent-light: var(--accent-soft);
    --gdg-drilldown-border: var(--glass-edge);
  }}

  .stFileUploader label, [data-testid="stFileUploaderDropzoneInstructions"],
  [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: var(--dim) !important; }}

  /* ---------- Reduced motion -------------------------------------------------------- */

  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
      animation-duration: .001ms !important; animation-iteration-count: 1 !important;
      transition-duration: .001ms !important;
    }}
    .stat:hover, .row:hover, .stButton > button:hover, .finding:hover {{ transform: none; }}
  }}
</style>
"""


# Kept so existing imports of the old constant keep working.
CSS = css("light")
