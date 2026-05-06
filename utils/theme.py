"""
theme.py
========
Premium SaaS-grade design system for the flange-detection app.
Inspired by modern enterprise dashboards (Linear, Stripe, Vercel,
Datadog, Notion).

Import once at the top of every Streamlit page and call:
    inject_theme()
    side_nav(active="home" | "train" | "predict" | "analyze")
"""
import os
import streamlit as st


# -----------------------------------------------------------------------------
# Color tokens
# -----------------------------------------------------------------------------
COLORS = {
    'brand'           : '#C8102E',
    'brand_dark'      : '#9B0C24',
    'brand_glow'      : 'rgba(200,16,46,0.18)',
    'brand_50'        : '#FFF1F3',
    'brand_100'       : '#FFE0E5',

    'ink'             : '#0F1419',
    'ink_2'           : '#1F2937',
    'ink_3'           : '#374151',
    'muted'           : '#6B7280',
    'subtle'          : '#9CA3AF',
    'whisper'         : '#D1D5DB',

    'border'          : '#E5E7EB',
    'border_strong'   : '#CBD5E1',
    'border_focus'    : '#C8102E',

    'surface'         : '#FFFFFF',
    'surface_2'       : '#FAFAFA',
    'surface_3'       : '#F4F4F5',
    'page_bg'         : '#FAFAFB',

    'success'         : '#10B981',
    'success_50'      : '#ECFDF5',
    'success_900'     : '#064E3B',
    'warn'            : '#F59E0B',
    'warn_50'         : '#FFFBEB',
    'warn_900'        : '#78350F',
    'info'            : '#3B82F6',
    'info_50'         : '#EFF6FF',
    'info_900'        : '#1E3A8A',
    'danger'          : '#EF4444',
    'danger_50'       : '#FEF2F2',

    'class_0'         : '#2563EB',
    'class_1'         : '#EA580C',
    'class_2'         : '#16A34A',
}


# =============================================================================
# Master CSS
# =============================================================================
_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200&display=swap');

/* ====================================================================
   ICON FONT HARDENING + FALLBACK
   ====================================================================
   Streamlit emits icons as raw text strings like "keyboard_arrow_right",
   "expand_more", "upload", "arrow_drop_down" inside <span> tags. These
   are meant to be rendered by the Material Icons font. If the font
   fails to load (slow network, corporate proxy, ad blocker, certain
   Streamlit versions), the literal text leaks through into the UI.

   Two-layer defense:
     1. Apply the icon font correctly (the @import above + rules below).
     2. If the literal text leaks anyway, HIDE the text and replace
        the icon with a CSS-drawn SVG via ::before, so the UI never
        shows raw "keyboard_arrow_right" strings.
   ==================================================================== */

.material-icons,
.material-icons-outlined,
.material-symbols-outlined,
.material-symbols-rounded,
[class*="material-symbols"],
[class*="material-icons"],
span[data-testid*="StyledIcon"],
[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded',
                 'Material Symbols Outlined',
                 'Material Icons',
                 'Material Icons Outlined' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: inherit;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    font-variation-settings:
        'FILL' 0, 'wght' 500, 'GRAD' 0, 'opsz' 24;
}}

/* ----- BULLETPROOF EXPANDER CHEVRON ---------------------------------
   Don't trust the font. Hide whatever text Streamlit puts in the
   chevron icon position and draw our own with an SVG background. */

[data-testid="stExpander"] details > summary [data-testid="stIconMaterial"],
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary [class*="material"] {{
    /* Hide whatever Streamlit emits, in every conceivable way */
    font-size: 0 !important;
    color: transparent !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
    line-height: 0 !important;
    /* Replace with our own SVG chevron */
    width: 20px !important;
    min-width: 20px !important;
    height: 20px !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%236B7280' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'><polyline points='9 18 15 12 9 6'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 20px 20px !important;
    transition: transform 200ms cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: inline-block !important;
    flex-shrink: 0 !important;
}}

/* Rotate when expanded */
[data-testid="stExpander"] details[open] > summary [data-testid="stIconMaterial"],
[data-testid="stExpander"] details[open] > summary svg,
[data-testid="stExpander"] details[open] > summary [class*="material"] {{
    transform: rotate(90deg) !important;
}}

/* ----- BULLETPROOF FILE UPLOADER ICON ------------------------------- */

[data-testid="stFileUploader"] [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    color: transparent !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
    line-height: 0 !important;
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24' fill='none' stroke='%23C8102E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/><polyline points='17 8 12 3 7 8'/><line x1='12' y1='3' x2='12' y2='15'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 32px 32px !important;
    display: inline-block !important;
    flex-shrink: 0 !important;
}}

/* ----- BULLETPROOF TOOLTIP / HELP "?" ICON --------------------------- */

[data-testid="stTooltipIcon"] [data-testid="stIconMaterial"],
[data-testid="stTooltipHoverTarget"] [data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    color: transparent !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
    line-height: 0 !important;
    width: 16px !important;
    min-width: 16px !important;
    height: 16px !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%236B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 16px 16px !important;
    display: inline-block !important;
    transition: opacity 140ms !important;
    flex-shrink: 0 !important;
}}
[data-testid="stTooltipIcon"]:hover [data-testid="stIconMaterial"],
[data-testid="stTooltipHoverTarget"]:hover [data-testid="stIconMaterial"] {{
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23C8102E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>") !important;
}}

/* ----- BULLETPROOF NUMBER INPUT +/- BUTTONS ------------------------- */

[data-testid="stNumberInput"] button [data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    color: transparent !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
    line-height: 0 !important;
    width: 12px !important;
    min-width: 12px !important;
    height: 12px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 12px 12px !important;
    display: inline-block !important;
    flex-shrink: 0 !important;
}}
/* Plus button */
[data-testid="stNumberInput"] button[aria-label*="ncrement"]
    [data-testid="stIconMaterial"],
[data-testid="stNumberInput"] button[aria-label*="ncrease"]
    [data-testid="stIconMaterial"],
[data-testid="stNumberInput"] button[aria-label*="lus"]
    [data-testid="stIconMaterial"] {{
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23374151' stroke-width='3' stroke-linecap='round'><line x1='12' y1='5' x2='12' y2='19'/><line x1='5' y1='12' x2='19' y2='12'/></svg>") !important;
}}
/* Minus button */
[data-testid="stNumberInput"] button[aria-label*="ecrement"]
    [data-testid="stIconMaterial"],
[data-testid="stNumberInput"] button[aria-label*="ecrease"]
    [data-testid="stIconMaterial"],
[data-testid="stNumberInput"] button[aria-label*="inus"]
    [data-testid="stIconMaterial"] {{
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23374151' stroke-width='3' stroke-linecap='round'><line x1='5' y1='12' x2='19' y2='12'/></svg>") !important;
}}

/* ----- BULLETPROOF SELECTBOX DROPDOWN ARROW ------------------------- */

[data-testid="stSelectbox"] svg,
[data-baseweb="select"] svg {{
    color: var(--muted) !important;
}}

/* ----- LAST-RESORT CATCH-ALL --------------------------------------
   Any [data-testid="stIconMaterial"] that wasn't caught by the
   specific rules above gets its raw text squashed to zero size.
   This is the safety net that ensures NO Material icon name text
   ever reaches the user, even if our CSS doesn't anticipate the
   exact wrapper Streamlit uses. */

[data-testid="stIconMaterial"] {{
    font-size: 0 !important;
    color: transparent !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
    line-height: 0 !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
}}

</style>
<style>


:root {{
    --brand:           {COLORS['brand']};
    --brand-dark:      {COLORS['brand_dark']};
    --brand-glow:      {COLORS['brand_glow']};
    --brand-50:        {COLORS['brand_50']};
    --brand-100:       {COLORS['brand_100']};

    --ink:             {COLORS['ink']};
    --ink-2:           {COLORS['ink_2']};
    --ink-3:           {COLORS['ink_3']};
    --muted:           {COLORS['muted']};
    --subtle:          {COLORS['subtle']};
    --whisper:         {COLORS['whisper']};

    --border:          {COLORS['border']};
    --border-strong:   {COLORS['border_strong']};
    --border-focus:    {COLORS['border_focus']};

    --surface:         {COLORS['surface']};
    --surface-2:       {COLORS['surface_2']};
    --surface-3:       {COLORS['surface_3']};
    --page-bg:         {COLORS['page_bg']};

    --success:         {COLORS['success']};
    --success-50:      {COLORS['success_50']};
    --success-900:     {COLORS['success_900']};
    --warn:            {COLORS['warn']};
    --warn-50:         {COLORS['warn_50']};
    --warn-900:        {COLORS['warn_900']};
    --info:            {COLORS['info']};
    --info-50:         {COLORS['info_50']};
    --info-900:        {COLORS['info_900']};
    --danger:          {COLORS['danger']};
    --danger-50:       {COLORS['danger_50']};

    --class-0:         {COLORS['class_0']};
    --class-1:         {COLORS['class_1']};
    --class-2:         {COLORS['class_2']};

    --shadow-xs:       0 1px 2px rgba(15,20,25,0.04);
    --shadow-sm:       0 1px 3px rgba(15,20,25,0.06),
                       0 1px 2px rgba(15,20,25,0.04);
    --shadow-md:       0 4px 8px -2px rgba(15,20,25,0.08),
                       0 2px 4px -2px rgba(15,20,25,0.04);
    --shadow-lg:       0 12px 24px -8px rgba(15,20,25,0.14),
                       0 4px 8px -4px rgba(15,20,25,0.06);
    --shadow-xl:       0 24px 48px -12px rgba(15,20,25,0.18),
                       0 8px 16px -8px rgba(15,20,25,0.08);
    --shadow-glow:     0 0 0 4px var(--brand-glow);

    --radius-xs:       4px;
    --radius-sm:       6px;
    --radius-md:       10px;
    --radius-lg:       14px;
    --radius-xl:       20px;

    --ease:            cubic-bezier(0.4, 0, 0.2, 1);
    --ease-bounce:     cubic-bezier(0.34, 1.56, 0.64, 1);
}}

/* ====================================================================
   Global reset + typography
   ==================================================================== */
html, body, [class*="css"], .stApp, .stApp * {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, Oxygen, Ubuntu, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
code, pre, .stCode, [class*="stCodeBlock"] {{
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas,
                 monospace !important;
    font-size: 0.92em !important;
}}

/* ====================================================================
   Page background -- aurora gradients + dot grid
   ==================================================================== */
.stApp {{
    background:
        radial-gradient(1000px 500px at 5% -10%,
            rgba(200,16,46,0.06), transparent 55%),
        radial-gradient(1000px 500px at 95% -10%,
            rgba(37,99,235,0.05), transparent 55%),
        radial-gradient(800px 400px at 50% 110%,
            rgba(22,163,74,0.04), transparent 60%),
        var(--page-bg) !important;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(circle at 1px 1px,
                        rgba(15,20,25,0.025) 1px, transparent 0);
    background-size: 24px 24px;
    opacity: 0.5;
}}
.stApp > * {{ position: relative; z-index: 1; }}

.block-container {{
    padding-top: 1.6em !important;
    padding-bottom: 4em !important;
    max-width: 1320px !important;
}}

/* ====================================================================
   Page-load fade-in animation
   ==================================================================== */
@keyframes fade-up {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
@keyframes pulse-dot {{
    0%, 100% {{ box-shadow: 0 0 0 3px rgba(16,185,129,0.18); }}
    50%      {{ box-shadow: 0 0 0 6px rgba(16,185,129,0.06); }}
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}

.hero, .kpi, .section-card, .step-mini, .result-card, .app-bar {{
    animation: fade-up 0.4s var(--ease) backwards;
}}
.kpi:nth-child(1) {{ animation-delay: 0.05s; }}
.kpi:nth-child(2) {{ animation-delay: 0.10s; }}
.kpi:nth-child(3) {{ animation-delay: 0.15s; }}
.kpi:nth-child(4) {{ animation-delay: 0.20s; }}

/* ====================================================================
   Top app bar -- glassmorphic with breadcrumbs + status + actions
   ==================================================================== */
.app-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7em 1.1em;
    margin: -0.2em 0 1.7em 0;
    background: rgba(255,255,255,0.72);
    backdrop-filter: saturate(180%) blur(14px);
    -webkit-backdrop-filter: saturate(180%) blur(14px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
}}
.app-bar .brand {{
    display: flex;
    align-items: center;
    gap: 0.7em;
    font-size: 0.93em;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.01em;
}}
.app-bar .brand .mark {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px; height: 30px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--brand) 0%,
                                          var(--brand-dark) 100%);
    color: #fff;
    font-weight: 800;
    font-size: 0.85em;
    box-shadow: 0 2px 6px var(--brand-glow),
                inset 0 1px 0 rgba(255,255,255,0.2);
    letter-spacing: -0.02em;
}}
.app-bar .brand .name {{
    font-weight: 700;
    color: var(--ink);
}}
.app-bar .brand .ver {{
    font-size: 0.74em;
    font-weight: 600;
    color: var(--brand);
    background: var(--brand-50);
    padding: 0.18em 0.5em;
    border-radius: 999px;
    margin-left: 0.3em;
    letter-spacing: 0.04em;
}}
.app-bar .brand .sep {{
    color: var(--whisper);
    font-weight: 400;
    margin: 0 0.3em;
}}
.app-bar .brand .crumb {{
    color: var(--muted);
    font-weight: 500;
}}
.app-bar .actions {{
    display: flex;
    align-items: center;
    gap: 0.85em;
}}
.app-bar .kbd {{
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
    padding: 0.22em 0.55em;
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75em;
    font-weight: 600;
    color: var(--ink-3);
}}
.app-bar .status {{
    display: flex;
    align-items: center;
    gap: 0.5em;
    font-size: 0.83em;
    color: var(--ink-3);
    font-weight: 500;
    padding: 0.35em 0.75em;
    background: var(--success-50);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 999px;
}}
.app-bar .status .dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 3px rgba(16,185,129,0.18);
    animation: pulse-dot 2.4s ease-in-out infinite;
}}

/* ====================================================================
   Sidebar navigation -- premium SaaS look
   ==================================================================== */
section[data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    width: 252px !important;
    min-width: 252px !important;
    max-width: 252px !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1.2em; }}

.side-brand {{
    display: flex;
    align-items: center;
    gap: 0.7em;
    padding: 0 1.0em 1.2em 1.0em;
    margin-bottom: 0.3em;
    border-bottom: 1px solid var(--border);
}}
.side-brand .mark {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--brand) 0%,
                                          var(--brand-dark) 100%);
    color: #fff;
    font-weight: 800;
    font-size: 0.95em;
    box-shadow: 0 2px 8px var(--brand-glow),
                inset 0 1px 0 rgba(255,255,255,0.2);
}}
.side-brand .text .t {{
    font-size: 0.95em;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.1;
    letter-spacing: -0.01em;
}}
.side-brand .text .s {{
    font-size: 0.74em;
    color: var(--muted);
    font-weight: 500;
    margin-top: 0.15em;
}}

.side-section {{
    padding: 1.0em 1.0em 0.4em 1.0em;
    font-size: 0.7em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--subtle);
}}
.side-nav {{
    display: flex;
    flex-direction: column;
    gap: 0.15em;
    padding: 0 0.6em;
}}
.side-nav a {{
    display: flex;
    align-items: center;
    gap: 0.7em;
    padding: 0.6em 0.75em;
    border-radius: var(--radius-md);
    font-size: 0.93em;
    font-weight: 500;
    color: var(--ink-3);
    text-decoration: none !important;
    transition: all 140ms var(--ease);
    border: 1px solid transparent;
}}
.side-nav a:hover {{
    background: var(--surface-2);
    color: var(--ink);
    text-decoration: none !important;
}}
.side-nav a.active {{
    background: var(--brand-50);
    color: var(--brand);
    font-weight: 600;
    border-color: var(--brand-100);
    box-shadow: var(--shadow-xs);
}}
.side-nav a .ico {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 6px;
    background: var(--surface-3);
    color: var(--muted);
    font-size: 0.85em;
    font-weight: 700;
    flex-shrink: 0;
    transition: all 140ms var(--ease);
}}
.side-nav a.active .ico {{
    background: var(--brand);
    color: #fff;
}}
.side-nav a .badge {{
    margin-left: auto;
    font-size: 0.7em;
    font-weight: 700;
    padding: 0.15em 0.5em;
    border-radius: 999px;
    background: var(--surface-3);
    color: var(--muted);
}}
.side-nav a .badge.live {{
    background: var(--success-50);
    color: var(--success-900);
}}

.side-card {{
    margin: 1.0em 0.8em 0.3em 0.8em;
    padding: 0.95em 1.0em;
    border-radius: var(--radius-md);
    background: linear-gradient(135deg,
        var(--brand-50) 0%, rgba(255,255,255,0.5) 100%);
    border: 1px solid var(--brand-100);
    font-size: 0.85em;
}}
.side-card .t {{
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.3em;
    font-size: 0.95em;
}}
.side-card .s {{
    color: var(--ink-3);
    line-height: 1.5;
    font-size: 0.93em;
}}

.side-footer {{
    margin-top: auto;
    padding: 1.2em 1.0em 0.8em 1.0em;
    border-top: 1px solid var(--border);
    font-size: 0.78em;
    color: var(--muted);
    line-height: 1.55;
}}
.side-footer b {{ color: var(--ink); font-weight: 600; }}

/* Hide Streamlit's auto-generated sidebar page list since we provide
   our own custom nav. */
section[data-testid="stSidebarNav"] {{ display: none !important; }}

/* ====================================================================
   Hero
   ==================================================================== */
.hero {{
    position: relative;
    padding: 2.6em 2.8em 2.2em 2.8em;
    border-radius: var(--radius-xl);
    background:
        linear-gradient(135deg,
            rgba(200,16,46,0.05) 0%,
            rgba(255,255,255,0.0) 38%),
        rgba(255,255,255,0.78);
    backdrop-filter: saturate(180%) blur(10px);
    -webkit-backdrop-filter: saturate(180%) blur(10px);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-md);
    overflow: hidden;
    margin-bottom: 1.5em;
}}

/* Subtle grid pattern overlay -- adds technical / data-product feel */
.hero > .grid-bg {{
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(to right,
            rgba(15,20,25,0.025) 1px, transparent 1px),
        linear-gradient(to bottom,
            rgba(15,20,25,0.025) 1px, transparent 1px);
    background-size: 32px 32px;
    -webkit-mask-image: linear-gradient(135deg,
        rgba(0,0,0,0.5), transparent 60%);
            mask-image: linear-gradient(135deg,
        rgba(0,0,0,0.5), transparent 60%);
    pointer-events: none;
    z-index: 0;
}}
.hero::after {{
    content: "";
    position: absolute;
    top: -160px; right: -160px;
    width: 420px; height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle,
                                rgba(200,16,46,0.10) 0%,
                                transparent 65%);
    pointer-events: none;
}}
.hero::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg,
        var(--brand) 0%, var(--brand-dark) 100%);
}}
.hero .eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 0.5em;
    font-size: 0.71em;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--brand);
    background: var(--brand-50);
    border: 1px solid var(--brand-100);
    padding: 0.4em 0.8em;
    border-radius: 999px;
    margin: 0 0 1.0em 0;
    position: relative;
    z-index: 1;
}}
.hero .eyebrow .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--brand);
    box-shadow: 0 0 0 3px var(--brand-glow);
}}
.hero h1 {{
    font-size: 2.7em !important;
    font-weight: 800 !important;
    line-height: 1.06 !important;
    color: var(--ink) !important;
    margin: 0 0 0.4em 0 !important;
    letter-spacing: -0.03em !important;
    position: relative;
    z-index: 1;
}}
.hero h1 .accent {{
    background: linear-gradient(135deg, var(--brand) 0%,
                                         var(--brand-dark) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero .subtitle {{
    font-size: 1.18em;
    font-weight: 400;
    color: var(--ink-3);
    margin: 0 0 1.0em 0;
    line-height: 1.55;
    max-width: 720px;
    position: relative;
    z-index: 1;
}}
.hero .meta {{
    font-size: 0.92em;
    color: var(--muted);
    line-height: 1.7;
    position: relative;
    z-index: 1;
}}
.hero .meta b {{ color: var(--ink); font-weight: 600; }}

.hero .actions {{
    display: flex;
    gap: 0.6em;
    margin-top: 1.2em;
    position: relative;
    z-index: 1;
}}
.hero .actions .chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    padding: 0.45em 0.85em;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
    font-size: 0.84em;
    font-weight: 600;
    color: var(--ink-3);
    box-shadow: var(--shadow-xs);
}}
.hero .actions .chip.live {{
    background: var(--success-50);
    border-color: rgba(16,185,129,0.25);
    color: var(--success-900);
}}
.hero .actions .chip.live .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse-dot 2.4s ease-in-out infinite;
}}

/* ====================================================================
   KPI tiles -- with animated number entry + sparkline accent
   ==================================================================== */
.kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.2em 1.3em;
    transition: all 200ms var(--ease);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-xs);
}}
.kpi::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg,
        transparent, var(--brand), transparent);
    opacity: 0;
    transition: opacity 200ms;
}}
.kpi:hover {{
    transform: translateY(-3px);
    border-color: var(--border-strong);
    box-shadow: var(--shadow-md);
}}
.kpi:hover::before {{ opacity: 1; }}
.kpi .icon-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.95em;
}}
.kpi .icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px; height: 38px;
    border-radius: 11px;
    background: linear-gradient(135deg,
        var(--brand-50) 0%, var(--brand-100) 100%);
    color: var(--brand);
    font-size: 1.1em;
    font-weight: 800;
    box-shadow: inset 0 0 0 1px var(--brand-100);
}}
.kpi .trend {{
    font-size: 0.74em;
    font-weight: 600;
    color: var(--success-900);
    background: var(--success-50);
    padding: 0.22em 0.6em;
    border-radius: 999px;
    letter-spacing: 0.01em;
    border: 1px solid rgba(16,185,129,0.2);
}}
.kpi .trend.neutral {{
    color: var(--ink-3);
    background: var(--surface-3);
    border-color: var(--border);
}}
.kpi .trend.info {{
    color: var(--info-900);
    background: var(--info-50);
    border-color: rgba(59,130,246,0.2);
}}
.kpi .v {{
    font-size: 2.2em;
    font-weight: 800;
    color: var(--ink);
    line-height: 1.0;
    letter-spacing: -0.03em;
    margin-bottom: 0.25em;
    font-variant-numeric: tabular-nums;
}}
.kpi .v .unit {{
    font-size: 0.5em;
    color: var(--muted);
    font-weight: 700;
    margin-left: 0.2em;
    letter-spacing: 0;
}}
.kpi .l {{
    font-size: 0.86em;
    color: var(--muted);
    line-height: 1.4;
    font-weight: 500;
}}
.kpi .spark {{
    margin-top: 0.85em;
    height: 28px;
    display: flex;
    align-items: flex-end;
    gap: 2px;
}}
.kpi .spark span {{
    flex: 1;
    background: var(--brand-100);
    border-radius: 1.5px;
    transition: background 200ms;
}}
.kpi:hover .spark span {{ background: var(--brand); }}

/* ====================================================================
   Section cards (home)
   ==================================================================== */
.section-card {{
    position: relative;
    padding: 1.9em 1.8em 1.5em 1.8em;
    border-radius: var(--radius-lg);
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-xs);
    transition: all 280ms var(--ease);
    height: 100%;
    overflow: hidden;
}}
.section-card::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(450px 220px at 100% 0%,
                                rgba(200,16,46,0.07), transparent 50%);
    opacity: 0;
    transition: opacity 280ms;
    pointer-events: none;
}}
.section-card:hover {{
    transform: translateY(-5px);
    box-shadow: var(--shadow-xl);
    border-color: var(--brand);
}}
.section-card:hover::before {{ opacity: 1; }}
.section-card .num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px; height: 48px;
    border-radius: 13px;
    background: linear-gradient(135deg, var(--brand-50) 0%,
                                          var(--brand-100) 100%);
    color: var(--brand);
    font-weight: 800;
    font-size: 1.2em;
    margin-bottom: 1.0em;
    box-shadow: inset 0 0 0 1px var(--brand-100),
                0 4px 12px -4px var(--brand-glow);
    position: relative;
    z-index: 1;
    transition: all 280ms var(--ease-bounce);
}}
.section-card:hover .num {{
    transform: scale(1.08) rotate(-3deg);
    background: linear-gradient(135deg, var(--brand) 0%,
                                          var(--brand-dark) 100%);
    color: #fff;
}}
.section-card h3 {{
    margin: 0 0 0.55em 0 !important;
    font-size: 1.28em !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
}}
.section-card p {{
    margin: 0 0 0.6em 0;
    font-size: 0.94em;
    color: var(--ink-3);
    line-height: 1.6;
    position: relative;
    z-index: 1;
}}
.section-card .tag-row {{
    display: flex;
    gap: 0.4em;
    flex-wrap: wrap;
    margin-top: 1.0em;
    position: relative;
    z-index: 1;
}}
.section-card .tag {{
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    padding: 0.32em 0.7em;
    border-radius: 999px;
    background: var(--surface-3);
    border: 1px solid var(--border);
    color: var(--ink-3);
    font-size: 0.76em;
    font-weight: 600;
}}
.section-card .tag .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--muted);
}}
.section-card .tag.success {{
    background: var(--success-50);
    border-color: rgba(16,185,129,0.25);
    color: var(--success-900);
}}
.section-card .tag.success .dot {{ background: var(--success); }}
.section-card .tag.warn {{
    background: var(--warn-50);
    border-color: rgba(245,158,11,0.3);
    color: var(--warn-900);
}}
.section-card .tag.warn .dot {{ background: var(--warn); }}
.section-card .tag.info {{
    background: var(--info-50);
    border-color: rgba(59,130,246,0.25);
    color: var(--info-900);
}}
.section-card .tag.info .dot {{ background: var(--info); }}

/* ====================================================================
   Result hero (Page 2)
   ==================================================================== */
.result-card {{
    padding: 2.0em 2.2em;
    border-radius: var(--radius-xl);
    background: var(--surface);
    border: 1px solid var(--border);
    margin: 1.0em 0;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}}
.result-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 6px; height: 100%;
    background: var(--brand);
}}
.result-card.cls-0::before {{ background: var(--class-0); }}
.result-card.cls-1::before {{ background: var(--class-1); }}
.result-card.cls-2::before {{ background: var(--class-2); }}
.result-card::after {{
    content: "";
    position: absolute;
    top: -100px; right: -100px;
    width: 320px; height: 320px;
    border-radius: 50%;
    pointer-events: none;
    opacity: 0.6;
}}
.result-card.cls-0::after {{
    background: radial-gradient(circle,
        rgba(37,99,235,0.12) 0%, transparent 65%);
}}
.result-card.cls-1::after {{
    background: radial-gradient(circle,
        rgba(234,88,12,0.12) 0%, transparent 65%);
}}
.result-card.cls-2::after {{
    background: radial-gradient(circle,
        rgba(22,163,74,0.12) 0%, transparent 65%);
}}
.result-card .label {{
    display: inline-flex;
    align-items: center;
    gap: 0.6em;
    font-size: 0.74em;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5em;
    position: relative;
    z-index: 1;
}}
.result-card .label .badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    padding: 0.25em 0.65em;
    border-radius: 999px;
    background: var(--success-50);
    color: var(--success-900);
    font-size: 0.95em;
    letter-spacing: 0.05em;
    border: 1px solid rgba(16,185,129,0.25);
}}
.result-card .label .badge .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse-dot 2.4s ease-in-out infinite;
}}
.result-card .value {{
    font-size: 3.2em;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.035em;
    margin-bottom: 0.35em;
    position: relative;
    z-index: 1;
}}
.result-card.cls-0 .value {{
    background: linear-gradient(135deg, var(--class-0) 0%, #1D4ED8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.result-card.cls-1 .value {{
    background: linear-gradient(135deg, var(--class-1) 0%, #C2410C 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.result-card.cls-2 .value {{
    background: linear-gradient(135deg, var(--class-2) 0%, #15803D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.result-card .pills {{
    display: flex;
    gap: 0.5em;
    flex-wrap: wrap;
    margin-top: 0.8em;
    position: relative;
    z-index: 1;
}}
.result-card .pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.5em;
    padding: 0.5em 0.9em;
    border-radius: 999px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--ink-3);
    font-size: 0.9em;
    font-weight: 500;
    transition: all 150ms;
}}
.result-card .pill:hover {{
    background: var(--surface-3);
    border-color: var(--border-strong);
    transform: translateY(-1px);
}}
.result-card .pill .k {{ color: var(--muted); }}
.result-card .pill b {{ color: var(--ink); font-weight: 700; }}

/* ====================================================================
   Notice / disclaimer
   ==================================================================== */
.notice {{
    display: flex;
    align-items: flex-start;
    gap: 0.85em;
    background: var(--warn-50);
    border: 1px solid rgba(245,158,11,0.3);
    border-left: 3px solid var(--warn);
    padding: 0.95em 1.2em;
    border-radius: var(--radius-md);
    font-size: 0.93em;
    color: var(--warn-900);
    margin: 0.5em 0 1.0em 0;
    line-height: 1.6;
}}
.notice .ico {{
    flex-shrink: 0;
    width: 24px; height: 24px;
    border-radius: 50%;
    background: var(--warn);
    color: #fff;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78em;
    font-weight: 800;
    margin-top: 0.05em;
}}
.notice b {{ color: var(--warn-900); }}
.notice.info {{
    background: var(--info-50);
    border-color: rgba(59,130,246,0.3);
    border-left-color: var(--info);
    color: var(--info-900);
}}
.notice.info b {{ color: var(--info-900); }}
.notice.info .ico {{ background: var(--info); }}
.notice.success {{
    background: var(--success-50);
    border-color: rgba(16,185,129,0.3);
    border-left-color: var(--success);
    color: var(--success-900);
}}
.notice.success b {{ color: var(--success-900); }}
.notice.success .ico {{ background: var(--success); }}

/* ====================================================================
   Step header
   ==================================================================== */
.step-header {{
    display: flex;
    align-items: flex-start;
    gap: 1.1em;
    margin: 2.0em 0 1.0em 0;
    padding-bottom: 0.85em;
    border-bottom: 1px solid var(--border);
}}
.step-header .chip {{
    flex-shrink: 0;
    width: 40px; height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg,
        var(--brand-50) 0%, var(--brand-100) 100%);
    color: var(--brand);
    font-weight: 800;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05em;
    box-shadow: inset 0 0 0 1px var(--brand-100),
                0 2px 6px -2px var(--brand-glow);
}}
.step-header .body {{ flex: 1; min-width: 0; }}
.step-header .title {{
    font-size: 1.32em;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.2;
    letter-spacing: -0.02em;
}}
.step-header .sub {{
    color: var(--muted);
    font-size: 0.93em;
    margin-top: 0.3em;
    line-height: 1.55;
}}

/* ====================================================================
   Step mini-cards (home pipeline)
   ==================================================================== */
.step-mini {{
    padding: 1.1em 1.3em;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface);
    margin-bottom: 0.8em;
    display: flex;
    gap: 1.0em;
    align-items: flex-start;
    box-shadow: var(--shadow-xs);
    transition: all 200ms var(--ease);
    position: relative;
    overflow: hidden;
}}
.step-mini::before {{
    content: "";
    position: absolute;
    left: 0; top: 0;
    width: 3px; height: 100%;
    background: var(--brand);
    transform: scaleY(0);
    transform-origin: top;
    transition: transform 200ms var(--ease);
}}
.step-mini:hover {{
    border-color: var(--border-strong);
    box-shadow: var(--shadow-sm);
    transform: translateX(3px);
}}
.step-mini:hover::before {{ transform: scaleY(1); }}
.step-mini .num {{
    flex-shrink: 0;
    width: 32px; height: 32px;
    border-radius: 9px;
    background: var(--surface-3);
    color: var(--ink-3);
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85em;
    font-variant-numeric: tabular-nums;
    transition: all 200ms;
}}
.step-mini:hover .num {{
    background: var(--brand-50);
    color: var(--brand);
}}
.step-mini .body .t {{
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.2em;
    font-size: 0.96em;
    letter-spacing: -0.01em;
}}
.step-mini .body .d {{
    color: var(--ink-3);
    font-size: 0.88em;
    line-height: 1.55;
}}

/* ====================================================================
   Soft divider
   ==================================================================== */
.thin-divider {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.8em 0;
}}

/* ====================================================================
   Streamlit primary button
   ==================================================================== */
button[kind="primary"], div[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg,
        var(--brand) 0%, var(--brand-dark) 100%) !important;
    border: none !important;
    box-shadow: 0 1px 2px rgba(200,16,46,0.18),
                inset 0 1px 0 rgba(255,255,255,0.15) !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: -0.005em !important;
    transition: all 160ms var(--ease) !important;
}}
button[kind="primary"]:hover,
div[data-testid="stBaseButton-primary"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 8px 20px -4px rgba(200,16,46,0.4),
                inset 0 1px 0 rgba(255,255,255,0.15) !important;
}}
button[kind="primary"]:active {{
    transform: translateY(0);
}}

button[kind="secondary"], div[data-testid="stBaseButton-secondary"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink-2) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all 140ms !important;
}}
button[kind="secondary"]:hover {{
    border-color: var(--border-strong) !important;
    background: var(--surface-2) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm) !important;
}}

/* ====================================================================
   Form inputs
   ==================================================================== */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
    border-radius: var(--radius-md) !important;
    border-color: var(--border) !important;
    transition: all 140ms !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {{
    border-color: var(--brand) !important;
    box-shadow: 0 0 0 3px var(--brand-glow) !important;
}}

[data-testid="stFileUploader"] {{
    margin-top: 0.4em;
}}
[data-testid="stFileUploader"] > section {{
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    background:
        linear-gradient(135deg,
            rgba(200,16,46,0.015) 0%,
            rgba(255,255,255,0) 50%),
        var(--surface-2) !important;
    padding: 1.4em 1.4em !important;
    transition: all 220ms var(--ease) !important;
    position: relative;
    overflow: hidden;
}}
[data-testid="stFileUploader"] > section::before {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(400px 200px at 50% 0%,
            rgba(200,16,46,0.05), transparent 60%);
    opacity: 0;
    transition: opacity 220ms;
    pointer-events: none;
}}
[data-testid="stFileUploader"] > section:hover {{
    border-color: var(--brand) !important;
    background:
        linear-gradient(135deg,
            rgba(200,16,46,0.04) 0%,
            rgba(255,255,255,0) 50%),
        var(--brand-50) !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}}
[data-testid="stFileUploader"] > section:hover::before {{ opacity: 1; }}

/* Style the inner button (Browse files) */
[data-testid="stFileUploader"] > section button {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink-2) !important;
    font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5em 1.1em !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all 140ms var(--ease) !important;
}}
[data-testid="stFileUploader"] > section button:hover {{
    border-color: var(--brand) !important;
    color: var(--brand) !important;
    background: var(--brand-50) !important;
    transform: translateY(-1px);
}}

/* The "drag and drop" small text inside the uploader */
[data-testid="stFileUploader"] > section > div > small,
[data-testid="stFileUploaderDropzoneInstructions"] {{
    color: var(--muted) !important;
    font-size: 0.88em !important;
    font-weight: 500 !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: var(--ink-2) !important;
    font-weight: 600 !important;
}}

/* ====================================================================
   Headings
   ==================================================================== */
h1, h2, h3, h4, h5 {{
    color: var(--ink) !important;
    letter-spacing: -0.02em !important;
    font-weight: 700 !important;
}}
h2 {{ font-size: 1.7em !important; }}
h3 {{ font-size: 1.3em !important; }}

/* ====================================================================
   DataFrames
   ==================================================================== */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    box-shadow: var(--shadow-xs);
}}

/* ====================================================================
   Tabs
   ==================================================================== */
[data-testid="stTabs"] button[role="tab"] {{
    font-weight: 600 !important;
    color: var(--muted) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: all 150ms !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: var(--brand) !important;
    border-bottom-color: var(--brand) !important;
}}

/* ====================================================================
   Metrics / expanders
   ==================================================================== */
[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.0em 1.2em;
    box-shadow: var(--shadow-xs);
    transition: all 160ms;
}}
[data-testid="stMetric"]:hover {{
    border-color: var(--border-strong);
    box-shadow: var(--shadow-sm);
}}

[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-xs);
    overflow: hidden;
    transition: all 160ms var(--ease);
}}
[data-testid="stExpander"]:hover {{
    border-color: var(--border-strong) !important;
    box-shadow: var(--shadow-sm);
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {{
    font-weight: 600 !important;
    color: var(--ink-2) !important;
    padding: 0.85em 1.1em !important;
    transition: background 140ms;
}}
[data-testid="stExpander"] summary:hover {{
    background: var(--surface-2);
}}

/* Help / tooltip ? icon -- the WRAPPER. Inner icon is handled by the
   bulletproof SVG rules at the top of the file. */
[data-testid="stTooltipIcon"],
[data-testid="stTooltipHoverTarget"] {{
    opacity: 0.85;
    transition: opacity 140ms;
}}
[data-testid="stTooltipIcon"]:hover,
[data-testid="stTooltipHoverTarget"]:hover {{
    opacity: 1;
}}

/* Number-input +/- buttons -- WRAPPER style only. Icons are
   bulletproof-replaced at the top of the file. */
[data-testid="stNumberInput"] button {{
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
    color: var(--ink-3) !important;
    transition: all 140ms;
}}
[data-testid="stNumberInput"] button:hover {{
    background: var(--brand-50) !important;
    border-color: var(--brand) !important;
}}

/* st.toggle / switch */
label[data-baseweb="checkbox"] [role="checkbox"],
[data-testid="stToggle"] [role="checkbox"] {{
    background: var(--whisper) !important;
    transition: background 200ms !important;
}}
label[data-baseweb="checkbox"] [aria-checked="true"],
[data-testid="stToggle"] [aria-checked="true"] {{
    background: linear-gradient(135deg,
        var(--brand) 0%, var(--brand-dark) 100%) !important;
}}

/* Progress bar polish */
[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg,
        var(--brand) 0%, var(--brand-dark) 100%) !important;
    border-radius: 999px !important;
}}
[data-testid="stProgress"] > div {{
    background: var(--surface-3) !important;
    border-radius: 999px !important;
}}

/* Audio player polish */
audio {{
    width: 100%;
    border-radius: var(--radius-md);
    height: 44px;
}}

/* ====================================================================
   Hide Streamlit branding for commercial feel
   ==================================================================== */
[data-testid="stToolbarActions"],
[data-testid="stStatusWidget"],
footer,
#MainMenu {{
    display: none !important;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: var(--whisper);
    border-radius: 5px;
    border: 2px solid var(--page-bg);
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--subtle);
}}

hr {{ border-color: var(--border) !important; }}
</style>
"""


# =============================================================================
# Public API
# =============================================================================
def inject_theme():
    """Call once near the top of every page (after st.set_page_config)."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def side_nav(active="home"):
    """
    Render the persistent left-side navigation.
    active: 'home' | 'train' | 'predict' | 'analyze'
    """
    items = [
        ("home",    "Overview",        "⌂", "app.py",                            None),
        ("train",   "Train Model",     "▲", "pages/1_Train_Your_Own_Model.py",   None),
        ("predict", "Predict",         "⚡", "pages/2_Competition_Prediction.py", "Live"),
        ("analyze", "Signal Analysis", "≡", "pages/3_Signal_Analysis.py",        None),
    ]

    with st.sidebar:
        # Brand block
        st.markdown("""
            <div class="side-brand">
                <span class="mark">UH</span>
                <div class="text">
                    <div class="t">Flange Detection</div>
                    <div class="s">v1.0 · Spring 2026</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Section label
        st.markdown(
            "<div class='side-section'>Workspace</div>",
            unsafe_allow_html=True,
        )

        # CSS to make st.page_link blend with our design + an active-state
        # overlay we render via a sibling marker.
        st.markdown("""
            <style>
            section[data-testid='stSidebar'] [data-testid='stPageLink'] {
                display: block;
                margin: 0 0.6em 0.18em 0.6em;
            }
            section[data-testid='stSidebar'] [data-testid='stPageLink'] a {
                display: flex;
                align-items: center;
                gap: 0.7em;
                padding: 0.6em 0.8em !important;
                border-radius: var(--radius-md);
                font-size: 0.94em !important;
                font-weight: 500 !important;
                color: var(--ink-3) !important;
                text-decoration: none !important;
                transition: all 140ms var(--ease);
                border: 1px solid transparent;
            }
            section[data-testid='stSidebar'] [data-testid='stPageLink'] a:hover {
                background: var(--surface-2);
                color: var(--ink) !important;
            }
            section[data-testid='stSidebar'] [data-testid='stPageLink'].active a {
                background: var(--brand-50) !important;
                color: var(--brand) !important;
                font-weight: 600 !important;
                border-color: var(--brand-100) !important;
                box-shadow: var(--shadow-xs);
            }
            </style>
        """, unsafe_allow_html=True)

        # Render nav items. We render each st.page_link, then a small JS
        # snippet immediately below assigns the 'active' class to the one
        # matching the current page.
        for key, label, icon, target, badge in items:
            label_render = f"{icon}   {label}"
            if badge:
                label_render += f"   ·  {badge}"
            try:
                st.page_link(target, label=label_render,
                             use_container_width=True)
            except Exception:
                pass

        # Mark the active item via a small script that finds the page_link
        # whose href matches the active target and adds the .active class.
        active_target = next((t for k, _, _, t, _ in items if k == active),
                             None)
        if active_target:
            target_short = os.path.basename(active_target).replace('.py', '')
            st.markdown(f"""
                <script>
                (function() {{
                    const links = parent.document.querySelectorAll(
                        'section[data-testid="stSidebar"] '
                        + '[data-testid="stPageLink"]'
                    );
                    links.forEach(el => {{
                        const a = el.querySelector('a');
                        if (a && (a.href.includes('{target_short}') ||
                            (a.href.endsWith('/') && '{target_short}' === 'app'))) {{
                            el.classList.add('active');
                        }}
                    }});
                }})();
                </script>
            """, unsafe_allow_html=True)

        # Pro-tip card
        st.markdown("""
            <div class="side-card">
                <div class="t">⌘  Pro tip</div>
                <div class="s">For best accuracy at the competition,
                upload an iPhone <code>.m4a</code> rather than using
                live browser recording.</div>
            </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
            <div class="side-footer">
                <b>Amrit Tiwari</b><br>
                Mechanical Engineering<br>
                University of Houston<br>
                <span style='color:var(--subtle)'>Spring 2026</span>
            </div>
        """, unsafe_allow_html=True)


def app_bar(crumb=None, status="System ready"):
    """Top brand bar with breadcrumbs + status indicator + keyboard hint."""
    crumb_html = ""
    if crumb:
        crumb_html = (
            f"<span class='sep'>/</span><span class='crumb'>{crumb}</span>"
        )
    st.markdown(f"""
        <div class="app-bar">
            <div class="brand">
                <span class="mark">UH</span>
                <span class="name">Flange Detection</span>
                <span class="ver">v1.0</span>
                {crumb_html}
            </div>
            <div class="actions">
                <span class="status">
                    <span class="dot"></span>
                    <span>{status}</span>
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def hero(eyebrow, title_html, subtitle, meta, chips=None):
    """
    Page hero card.
    chips : optional list of (label, kind) for status chips below the meta.
            kind: 'live' | None
    """
    chips_html = ""
    if chips:
        chip_items = []
        for lbl, kind in chips:
            cls = f"chip {kind}".strip() if kind else "chip"
            dot = "<span class='dot'></span>" if kind == 'live' else ""
            chip_items.append(f"<span class='{cls}'>{dot}{lbl}</span>")
        chips_html = (
            "<div class='actions'>" + "".join(chip_items) + "</div>"
        )

    st.markdown(f"""
        <div class="hero">
            <div class="grid-bg"></div>
            <div class="eyebrow">
                <span class="dot"></span>
                <span>{eyebrow}</span>
            </div>
            <h1>{title_html}</h1>
            <div class="subtitle">{subtitle}</div>
            <div class="meta">{meta}</div>
            {chips_html}
        </div>
    """, unsafe_allow_html=True)


def kpi(value, label, icon="●", trend=None, trend_kind=None,
        unit=None, sparkline=None):
    """
    KPI tile with optional sparkline (list of 0..1 floats, length 8-12).
    trend_kind : 'success' (green) | 'info' (blue) | 'neutral' (grey) | None
    """
    unit_html = f"<span class='unit'>{unit}</span>" if unit else ""

    if trend:
        kind = trend_kind or "success"
        trend_html = f"<span class='trend {kind}'>{trend}</span>"
    else:
        trend_html = "<span></span>"

    spark_html = ""
    if sparkline:
        bars = "".join(
            f"<span style='height:{int(max(8, v*100))}%'></span>"
            for v in sparkline
        )
        spark_html = f"<div class='spark'>{bars}</div>"

    st.markdown(f"""
        <div class="kpi">
            <div class="icon-row">
                <span class="icon">{icon}</span>
                {trend_html}
            </div>
            <div class="v">{value}{unit_html}</div>
            <div class="l">{label}</div>
            {spark_html}
        </div>
    """, unsafe_allow_html=True)


def section_card(num, title, body_html, tags=None):
    """tags: list of (label, kind) tuples."""
    tag_html = ""
    if tags:
        chips = []
        for lbl, kind in tags:
            cls = f"tag {kind}".strip() if kind else "tag"
            chips.append(
                f"<span class='{cls}'><span class='dot'></span>{lbl}</span>"
            )
        tag_html = "<div class='tag-row'>" + "".join(chips) + "</div>"
    st.markdown(f"""
        <div class="section-card">
            <div class="num">{num}</div>
            <h3>{title}</h3>
            {body_html}
            {tag_html}
        </div>
    """, unsafe_allow_html=True)


def result_hero(class_index, predicted_label, pills, status_text=None):
    badge_html = ""
    if status_text:
        badge_html = (
            f"<span class='badge'><span class='dot'></span>"
            f"{status_text}</span>"
        )
    pill_html = "".join(
        f"<div class='pill'><span class='k'>{lbl}</span><b>{val}</b></div>"
        for lbl, val in pills
    )
    st.markdown(f"""
        <div class="result-card cls-{class_index}">
            <div class="label">Predicted Torque {badge_html}</div>
            <div class="value">{predicted_label}</div>
            <div class="pills">{pill_html}</div>
        </div>
    """, unsafe_allow_html=True)


def disclaimer(html, kind="warn"):
    icons = {'warn': '!', 'info': 'i', 'success': '✓'}
    cls = f"notice {kind}".strip() if kind != "warn" else "notice"
    ico = icons.get(kind, '!')
    st.markdown(
        f"<div class='{cls}'><span class='ico'>{ico}</span>"
        f"<div>{html}</div></div>",
        unsafe_allow_html=True,
    )


def step_header(num, title, subtitle=None):
    sub_html = (f"<div class='sub'>{subtitle}</div>" if subtitle else "")
    st.markdown(f"""
        <div class="step-header">
            <div class="chip">{num}</div>
            <div class="body">
                <div class="title">{title}</div>
                {sub_html}
            </div>
        </div>
    """, unsafe_allow_html=True)


def step_mini(num, title, description):
    st.markdown(f"""
        <div class="step-mini">
            <div class="num">{num}</div>
            <div class="body">
                <div class="t">{title}</div>
                <div class="d">{description}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def thin_divider():
    st.markdown("<hr class='thin-divider' />", unsafe_allow_html=True)
