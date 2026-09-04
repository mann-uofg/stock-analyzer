"""The design system: flat surfaces on a quiet ground, dark by default.

This replaces a Liquid Glass treatment - blurred translucent panels floating
over a four-colour mesh of radial gradients. It photographed well and read
badly: the mesh sat behind every panel at a different colour depending on where
that panel happened to land, so identical cards looked different from each
other for no reason a reader could act on, and a page of numbers competed with
a background that was busier than the data.

What replaces it is deliberately plain, in the manner of the better consumer
finance apps:

1. **One flat ground, one flat surface.** Two values, no gradient anywhere. A
   card is distinguished by a hairline border and a small step in lightness,
   which is enough and never varies with position.
2. **Borders instead of shadows.** A hairline costs nothing, does not smear
   under a blur filter, and keeps edges honest at any zoom.
3. **Colour reserved for meaning.** The interface itself is greyscale, so the
   only saturated things on screen are gains, losses, warnings and the one
   accent. Decorative colour competes with data for the same signal.
4. **Restraint over ornament.** No specular sweeps, no glow, no hover lift.
   Movement is limited to what communicates state.

Both modes come from one function so a token cannot drift between them, and
motion is disabled wholesale under ``prefers-reduced-motion``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SCROLLNAV_DIR = Path(__file__).parent / "scrollnav"
_scrollnav: Any = None


def scroll_nav() -> None:
    """Mount the listener that hides the top nav on the way down.

    A component rather than a script tag because Streamlit strips scripts from
    markdown. It renders nothing; failure here costs the auto-hide and nothing
    else, so it never raises into the page.
    """
    global _scrollnav
    try:
        import streamlit.components.v1 as components

        if _scrollnav is None:
            _scrollnav = components.declare_component(
                "stock_analyzer_scrollnav", path=str(_SCROLLNAV_DIR)
            )
        _scrollnav(key="_scrollnav")
    except Exception:
        pass

# Palettes. Only these differ between modes; every rule below is shared.
#
# The token names are inherited from the previous glass system - `glass` is now
# simply the card surface and `sheen` is transparent - because several hundred
# rules reference them. Renaming would have churned the whole stylesheet to say
# the same thing.
PALETTES = {
    # Dark is the default and the mode the palette was designed in.
    "dark": {
        "ground": "#0a0a0c",
        "glass": "#141417",
        "glass_hi": "#1c1c21",
        "glass_edge": "rgba(255,255,255,.075)",
        "glass_rim": "rgba(0,0,0,.35)",
        # No specular highlights anywhere: this kills every `inset 0 1px 0`
        # in one place rather than in eighty rules.
        "sheen": "transparent",
        "shadow": "none",
        "shadow_hi": "none",
        "text": "#f4f4f7",
        "dim": "#9a9aa4",
        "faint": "#6a6a75",
        "accent": "#5b9cff",
        "accent_soft": "rgba(91,156,255,.14)",
        "gain": "#34d399",
        "gain_soft": "rgba(52,211,153,.13)",
        "loss": "#f87171",
        "loss_soft": "rgba(248,113,113,.13)",
        "warn": "#fbbf24",
        "warn_soft": "rgba(251,191,36,.13)",
        "track": "rgba(255,255,255,.09)",
        "grid": "rgba(255,255,255,.06)",
    },
    "light": {
        "ground": "#fafafb",
        "glass": "#ffffff",
        "glass_hi": "#f4f4f6",
        "glass_edge": "rgba(0,0,0,.09)",
        "glass_rim": "rgba(0,0,0,.05)",
        "sheen": "transparent",
        # One hairline shadow only, to lift a white card off a near-white page.
        "shadow": "0 1px 2px rgba(0,0,0,.05)",
        "shadow_hi": "0 1px 3px rgba(0,0,0,.08)",
        "text": "#111114",
        "dim": "#5a5a64",
        "faint": "#8a8a95",
        "accent": "#2563eb",
        "accent_soft": "rgba(37,99,235,.10)",
        "gain": "#0f9d58",
        "gain_soft": "rgba(15,157,88,.11)",
        "loss": "#d92d20",
        "loss_soft": "rgba(217,45,32,.10)",
        "warn": "#b45309",
        "warn_soft": "rgba(180,83,9,.11)",
        "track": "rgba(0,0,0,.08)",
        "grid": "rgba(0,0,0,.07)",
    },
}


def css(mode: str = "dark") -> str:
    """The full stylesheet for one mode."""
    p = PALETTES.get(mode, PALETTES["dark"])
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

    /* Gauge zones. Muted on purpose: the track says which end of a scale is
       good, while the marker on top of it stays the thing you actually read. */
    --zone-good: color-mix(in srgb, {p["gain"]} 26%, transparent);
    --zone-warn: color-mix(in srgb, {p["warn"]} 26%, transparent);
    --zone-bad: color-mix(in srgb, {p["loss"]} 26%, transparent);
    --zone-neutral: {p["track"]};

    /* Radii are tighter and more uniform than the concentric scheme they
       replace: with flat cards there is no lip to echo, and a smaller radius
       reads as more precise. */
    --r-xl: 16px; --r-lg: 14px; --r-md: 12px; --r-sm: 8px;
    --spring: cubic-bezier(.34, 1.2, .64, 1);
    --ease: cubic-bezier(.32, .72, 0, 1);

    /* Layout. The sidebar was taking a quarter of a laptop screen while
       holding a symbol box and four links. */
    --sidebar-w: 236px;
    --content-max: 1180px;
  }}

  /* ---------- Ground ----------------------------------------------------- */

  /* One flat colour. The mesh of radial gradients that used to sit here made
     every panel a slightly different shade depending on where it landed. */
  .stApp {{ background: var(--ground); }}

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
  /* Generous side padding rather than a hard centre column: with the sidebar
     no longer eating a quarter of the window, the content can use the width
     it has without the numbers running edge to edge. */
  .block-container {{
    padding-top: 2rem; padding-bottom: 5rem;
    padding-left: 2.4rem; padding-right: 2.4rem;
    max-width: var(--content-max);
  }}
  @media (max-width: 900px) {{
    .block-container {{ padding-left: 1.1rem; padding-right: 1.1rem; }}
  }}

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

  /* ---------- The card surface ------------------------------------------- */

  /* A solid fill and a hairline. No blur: filtering the backdrop cost a
     repaint on every scroll and, with a plain ground behind it, bought
     nothing but a slight muddying of the text sitting on top. */
  .glass, .stat, .finding, .rows, div[data-testid="stExpander"] details,
  [data-testid="stDataFrame"], [data-testid="stAlert"] {{
    background: var(--glass);
    border: 1px solid var(--glass-edge);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow);
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
    transition: border-color .18s var(--ease);
  }}
  /* Hover marks the border only. The card used to lift, scale and brighten,
     which on a dashboard of thirty read as the page twitching under the
     cursor - and none of these cards are clickable, so the affordance was
     promising something that never happened. */
  .stat:hover {{ border-color: var(--glass-hi); }}
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
    letter-spacing: .01em; border: 1px solid transparent;
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
  /* The one conic gradient left in the stylesheet, and the only one that
     carries information: it is the arc of a progress dial, not decoration. */
  .ring {{
    --pct: 0; --ring-col: var(--accent);
    width: 58px; height: 58px; border-radius: 50%; position: relative; flex: none;
    background: conic-gradient(var(--ring-col) calc(var(--pct) * 1%), var(--track) 0);
    animation: sweep .9s var(--ease) both;
  }}
  .ring::after {{
    content: ""; position: absolute; inset: 5px; border-radius: 50%;
    background: var(--glass);
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

  /* The storage bridge is a zero-height iframe that only moves data between
     Python and localStorage. Streamlit still reserves layout space for the
     block wrapping it, which showed as a ~26px gap above the page heading.
     Matched on the tail of the title: Streamlit prefixes the declared name
     with the module path, so this renders as
     "analyzer.browserstore.stock_analyzer_localstore". */
  iframe[title$="stock_analyzer_localstore"] {{ display: none !important; }}
  div[data-testid="stElementContainer"]:has(
    > iframe[title$="stock_analyzer_localstore"]
  ) {{ display: none !important; }}

  /* ---------- Gauges: a number shown on the scale it belongs to ----------- */

  .gauge-grid {{ display: grid; gap: .7rem;
                 grid-template-columns: repeat(auto-fit, minmax(255px, 1fr)); }}
  .gauge {{ padding: .85rem .95rem 1rem; border-radius: var(--r-md);
            background: var(--glass); border: 1px solid var(--glass-edge);
            box-shadow: 0 1px 2px var(--shadow); }}
  .gauge-head {{ display: flex; justify-content: space-between;
                 align-items: baseline; gap: .6rem; margin-bottom: .5rem; }}
  .gauge-label {{ font-size: .72rem; color: var(--faint); font-weight: 620;
                  text-transform: uppercase; letter-spacing: .07em; }}
  .gauge-value {{ font-size: 1.2rem; font-weight: 680; color: var(--text);
                  white-space: nowrap; }}
  .gauge-track {{ position: relative; height: 8px; border-radius: 999px;
                  background: var(--track);
                  box-shadow: inset 0 1px 2px var(--glass-rim); }}
  /* The marker is a pin, not a fill: these scales have a meaningful middle,
     so a bar growing from the left would imply more is always better. */
  .gauge-marker {{ position: absolute; top: 50%; width: 13px; height: 13px;
                   border-radius: 50%; background: var(--text);
                   border: 2.5px solid var(--ground);
                   transform: translate(-50%, -50%);
                   box-shadow: 0 1px 4px var(--shadow);
                   animation: pin .5s var(--spring) both .15s; }}
  @keyframes pin {{ from {{ opacity: 0; transform: translate(-50%, -50%) scale(.3); }}
                    to {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }} }}
  .gauge-ends {{ display: flex; justify-content: space-between;
                 font-size: .62rem; color: var(--faint); margin-top: .3rem;
                 font-variant-numeric: tabular-nums; }}
  .gauge-verdict {{ font-size: .84rem; font-weight: 680; margin-top: .5rem; }}
  .gauge-verdict.good {{ color: var(--gain); }}
  .gauge-verdict.warn {{ color: var(--warn); }}
  .gauge-verdict.bad {{ color: var(--loss); }}
  .gauge-verdict.neutral {{ color: var(--dim); }}
  .gauge-plain {{ font-size: .8rem; color: var(--dim); line-height: 1.45;
                  margin-top: .2rem; }}
  .gauge-note {{ font-size: .72rem; color: var(--faint); line-height: 1.4;
                 margin-top: .4rem; padding-top: .4rem;
                 border-top: 1px solid var(--glass-edge); }}

  .plain-summary {{ font-size: .92rem; line-height: 1.55; color: var(--text);
                    padding: .8rem 1rem; margin: .2rem 0 .9rem;
                    border-radius: var(--r-md); background: var(--glass);
                    border: 1px solid var(--glass-edge); }}

  /* ---------- Earnings calendar ------------------------------------------- */

  /* The strip scrolls sideways rather than wrapping: a calendar that reflows
     into rows stops reading as a timeline. */
  .cal-strip {{ display: flex; gap: .55rem; overflow-x: auto; padding: .2rem .1rem .6rem;
                scrollbar-width: thin; }}
  .cal-day {{ flex: 0 0 auto; min-width: 92px; text-align: center;
              padding: .7rem .6rem .6rem; border-radius: var(--r-md);
              background: var(--glass); border: 1px solid var(--glass-edge);
              box-shadow: 0 1px 2px var(--shadow); }}
  .cal-dow {{ font-size: .66rem; text-transform: uppercase; letter-spacing: .08em;
              color: var(--faint); font-weight: 640; }}
  .cal-num {{ font-size: 1.5rem; font-weight: 700; line-height: 1.1;
              color: var(--text); font-variant-numeric: tabular-nums; }}
  .cal-mon {{ font-size: .72rem; color: var(--dim); margin-bottom: .35rem; }}
  .cal-count {{ font-size: .68rem; font-weight: 640; color: var(--accent); }}
  .cal-away {{ font-size: .62rem; color: var(--faint); margin-top: .15rem; }}

  .earn-card {{ padding: .75rem .9rem; border-radius: var(--r-md);
                background: var(--glass); border: 1px solid var(--glass-edge);
                margin-bottom: .45rem; }}
  .earn-head {{ display: flex; align-items: center; gap: .45rem;
                flex-wrap: wrap; margin-bottom: .3rem; }}
  .earn-meta {{ font-size: .8rem; color: var(--dim); }}

  .earn-table {{ width: 100%; border-collapse: collapse; font-size: .84rem; }}
  .earn-table th {{ text-align: right; font-size: .68rem; font-weight: 640;
                    text-transform: uppercase; letter-spacing: .06em;
                    color: var(--faint); padding: .45rem .6rem;
                    border-bottom: 1px solid var(--glass-edge); }}
  .earn-table th:first-child, .earn-table td:first-child {{ text-align: left; }}
  .earn-table td {{ padding: .5rem .6rem; color: var(--text);
                    border-bottom: 1px solid var(--glass-edge); }}
  .earn-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .earn-table td.up {{ color: var(--gain); font-weight: 640; }}
  .earn-table td.down {{ color: var(--loss); font-weight: 640; }}

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
                 background: var(--accent);
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
    background: var(--glass);
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
  }}
  [data-baseweb="popover"] div[role="listbox"], [data-baseweb="menu"] {{
    background: var(--glass) !important; border-radius: var(--r-md) !important;
    border: 1px solid var(--glass-edge) !important;
    box-shadow: var(--shadow-hi) !important;
  }}
  [data-baseweb="menu"] li {{ color: var(--text) !important; }}
  [data-baseweb="menu"] li:hover {{ background: var(--accent-soft) !important; }}

  [data-testid="stAlert"] {{ padding: .85rem 1rem; }}
  [data-testid="stFileUploaderDropzone"] {{
    background: var(--glass) !important; border-radius: var(--r-lg) !important;
    border: 1px dashed var(--glass-edge) !important;
  }}

  /* Segmented control (st.segmented_control renders as a button group).
     Without this the unselected option keeps Streamlit's own white fill and
     turns into a blank slab in dark mode. */
  [data-testid="stButtonGroup"] {{
    gap: .18rem; padding: .26rem; border-radius: 999px; width: fit-content;
    background: var(--glass); border: 1px solid var(--glass-edge);
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

  /* The header carries the navigation now. Sticky rather than fixed so it
     participates in layout, and translated out of view by the scroll listener
     in views/scrollnav rather than by a media query. */
  [data-testid="stHeader"] {{
    position: sticky; top: 0; z-index: 90;
    background: var(--ground);
    border-bottom: 1px solid var(--glass-edge);
    transition: transform .26s var(--ease);
    will-change: transform;
  }}
  [data-testid="stHeader"].nav-hidden {{ transform: translateY(-100%); }}

  /* Centre the links. Streamlit lays the header out as a flex row and leaves
     the nav wherever the row puts it, which is hard left under the logo. */
  [data-testid="stHeader"] nav,
  [data-testid="stHeader"] > div:has([data-testid="stTopNavLink"]),
  [data-testid="stHeader"] div:has(> [data-testid="stTopNavLink"]) {{
    justify-content: center !important;
    gap: .15rem;
  }}

  [data-testid="stTopNavLink"] {{
    border-radius: var(--r-sm);
    padding: .4rem .85rem !important;
    transition: background .18s var(--ease), color .18s var(--ease);
    font-weight: 560;
  }}
  [data-testid="stTopNavLink"], [data-testid="stTopNavLink"] * {{
    color: var(--dim) !important;
  }}
  [data-testid="stTopNavLink"]:hover {{ background: var(--glass-hi); }}
  [data-testid="stTopNavLink"]:hover * {{ color: var(--text) !important; }}
  [data-testid="stTopNavLink"][aria-current="page"] {{
    background: var(--accent-soft);
  }}
  [data-testid="stTopNavLink"][aria-current="page"] * {{
    color: var(--accent) !important; font-weight: 640;
  }}

  /* The scroll listener renders nothing; Streamlit still reserves a block for
     it, which would show as a gap under the settings row. */
  iframe[title$="stock_analyzer_scrollnav"] {{ display: none !important; }}

  /* Settings sits at the top right of the content, level with the page title
     rather than above it. */
  .page-foot {{
    margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid var(--glass-edge);
    color: var(--faint); font-size: .76rem; line-height: 1.55;
  }}

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
