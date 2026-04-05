# =============================================================================
# app.py -- Exploring Europe's Top 5 Football Leagues in the 2025/26 Season
# =============================================================================
# Run locally:   python3 app.py
# Deploy:        gunicorn app:server
#
# Compatible versions:
#   pip install "dash==2.18.1" "dash-bootstrap-components==1.6.0"
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------------------
# 1. Design system -- Editorial Sports Journalism aesthetic
#    Deep slate + amber gold + pitch green + warm cream text
# -----------------------------------------------------------------------------

# Core palette
BG        = "#0d1117"       # near-black -- deep slate
SURFACE   = "#161b22"       # card surface
SURFACE2  = "#21262d"       # raised element
BORDER    = "#30363d"       # subtle border
TEXT      = "#e6edf3"       # warm off-white
MUTED     = "#adbac7"       # muted text
GOLD      = "#d4a017"       # amber gold -- stadium lights
GOLD_L    = "#f0c040"       # lighter gold for hover/accent
GREEN     = "#238636"       # pitch green -- used sparingly
RED       = "#da3633"       # red for negative/low scores
BLUE      = "#1f6feb"       # data blue

# Position colors -- slightly warmer than before
POS_COLORS = {
    "Forward":    "#2f81f7",   # electric blue
    "Midfielder": "#3fb950",   # pitch green
    "Wingback":   "#d29922",   # amber
    "Defender":   "#f85149",   # warm red
}

MARKER_STYLE = dict(
    colorscale=[[0, "#f85149"], [0.33, "#d29922"], [0.67, "#3fb950"], [1, "#2f81f7"]],
    showscale=True,
    colorbar=dict(
        title=dict(text="Position", font=dict(color=MUTED, size=11)),
        tickvals=[0, 1, 2, 3],
        ticktext=["Defender", "Wingback", "Midfielder", "Forward"],
        tickfont=dict(color=MUTED, size=10),
        bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
    ),
    cmin=0, cmax=3,
    line=dict(width=0.5, color=BG),
    size=9, opacity=0.85,
)

# Typography -- Georgia for headings (already loaded), system monospace for data
FONT_HEAD = "Georgia, 'Times New Roman', serif"
FONT_DATA = "'SF Mono', 'Fira Code', 'Courier New', monospace"
FONT_BODY = "Georgia, 'Times New Roman', serif"

# Shared Plotly layout base
FIG_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT_BODY, color=TEXT),
    legend=dict(
        font=dict(family=FONT_BODY, size=11, color=TEXT),
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER, borderwidth=1,
    ),
    margin=dict(l=60, r=40, t=60, b=60),
)
AXIS_BASE = dict(
    showgrid=True, gridcolor=SURFACE2, zeroline=False,
    tickfont=dict(family=FONT_BODY, color=MUTED, size=11),
    title_font=dict(family=FONT_BODY, color=MUTED, size=12),
    linecolor=BORDER, showline=True,
)
COG_AXIS = dict(
    range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1],
    zeroline=True, zerolinewidth=1, zerolinecolor=BORDER,
    showgrid=True, gridcolor=SURFACE2,
    tickfont=dict(family=FONT_BODY, color=MUTED, size=11),
    title_font=dict(family=FONT_BODY, color=MUTED, size=12),
    linecolor=BORDER, showline=True,
)

# Inline CSS injected via a hidden div -- drives typography and hover states
GLOBAL_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&display=swap');

  * { box-sizing: border-box; }

  body {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: Georgia, 'Times New Roman', serif;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #161b22; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

  /* Dropdown overrides */
  .Select-control { background-color: #21262d !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 4px !important; }
  .Select-menu-outer { background-color: #21262d !important; border: 1px solid #30363d !important; z-index: 9999 !important; }
  .Select-option { background-color: #21262d !important; color: #e6edf3 !important; }
  .Select-option:hover, .Select-option.is-focused { background-color: #30363d !important; color: #f0c040 !important; }
  .Select-option.is-selected { background-color: #1f6feb22 !important; color: #f0c040 !important; }
  .Select-value-label { color: #e6edf3 !important; }
  .Select-placeholder { color: #adbac7 !important; }
  .Select--multi .Select-value { background-color: #1f6feb33 !important; border-color: #1f6feb !important; color: #e6edf3 !important; }
  .Select-clear, .Select-arrow { color: #adbac7 !important; }
  .VirtualizedSelectOption { background-color: #21262d !important; color: #e6edf3 !important; }

  /* Slider overrides */
  .rc-slider-track { background-color: #d4a017 !important; }
  .rc-slider-handle { border-color: #f0c040 !important; background-color: #f0c040 !important; box-shadow: 0 0 0 3px rgba(240,192,64,0.25) !important; }
  .rc-slider-rail { background-color: #30363d !important; }

  /* Nav link hover */
  .nav-link:hover { color: #f0c040 !important; }

  /* Table row hover */
  tr:hover td { background-color: rgba(212,160,23,0.06) !important; }

  /* Modular section number labels */
  .section-num {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: #d4a017;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  /* Gold rule accent */
  .gold-rule {
    border: none;
    border-top: 2px solid #d4a017;
    width: 40px;
    margin: 0 0 16px 0;
  }

  /* Pitch green pip */
  .green-pip {
    display: inline-block;
    width: 8px; height: 8px;
    background: #238636;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
  }
</style>
"""

# Score colors -- warm traffic light
def score_color(val):
    if val >= 80: return "#3fb950"
    if val >= 65: return "#7ee787"
    if val >= 50: return GOLD
    if val >= 35: return "#d29922"
    return "#f85149"

# Pitch drawing constants
PITCH_LENGTH = 120
PITCH_WIDTH  = 80
CENTER_X     = PITCH_WIDTH / 2
SPACING      = 12

COG_COL_MAP = {
    "Defender":   "cog_defender",
    "Wingback":   "cog_wingback",
    "Midfielder": "cog_midfielder",
    "Forward":    "cog_forward",
}

def cog_to_y(cog):
    return (cog + 1) / 2 * PITCH_LENGTH

# Feature sets (unchanged from notebook)
FEATURE_SETS = {
    "Attack": [
        "Non-Penalty Expected Goals per 90",
        "Non-Penalty xG Overperformance",
        "Shots on Target %",
        "Expected Assisted Goals per 90",
        "Goal-Creating Actions per 90",
        "Passes into Penalty Area per 100 Touches",
    ],
    "Progression": [
        "Progressive Passes per 100 Touches",
        "Progressive Passing Distance per 100 Touches",
        "Progressive Carries per 100 Touches",
        "Progressive Carrying Distance per 100 Touches",
        "Take-Ons Attempted per 90",
        "Take-On Success %",
    ],
    "Defense": [
        "Tackles per 90",
        "Dribblers Tackled %",
        "Interceptions per 90",
        "Aerial Win %",
        "Ball Recoveries per 90",
        "Turnovers per 100 Touches",
    ],
}
ALL_KNN_FEATURES = sorted({f for fs in FEATURE_SETS.values() for f in fs})

SCORE_COLS = {
    "Attack":      "attack_score",
    "Progression": "progression_score",
    "Defense":     "defense_score",
    "Composite":   "composite_score",
}

X_OPTIONS = {
    "Attacking Compactness": "attacking_compactness",
    "Squad Rotation (MAD)":  "minutes_mad",
    "Touch COG":             "team_touch_cog",
    "Tackle COG":            "team_tackle_cog",
}
Y_OPTIONS = {
    "xGoals":   "xg",
    "Goals":    "goals",
    "Mistakes": "mistakes",
}

ORDER_OPTIONS = [
    {"label": "Descending", "value": "desc"},
    {"label": "Ascending",  "value": "asc"},
]

TEAM_SORT_OPTIONS = [
    {"label": "xG",              "value": "xg"},
    {"label": "Goals",           "value": "goals"},
    {"label": "Squad Rotation",  "value": "minutes_mad"},
    {"label": "Touch COG",       "value": "team_touch_cog"},
    {"label": "Tackle COG",      "value": "team_tackle_cog"},
    {"label": "Compactness",     "value": "attacking_compactness"},
    {"label": "Mistakes",        "value": "mistakes"},
    {"label": "Yellow Cards",    "value": "yellow_cards"},
    {"label": "Fouls Committed", "value": "fouls_committed"},
]

# -----------------------------------------------------------------------------
# 2. Load data
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

players_filtered = pd.read_parquet(DATA_DIR / "players_filtered.parquet")
players_filtered = players_filtered.drop(
    columns=[c for c in players_filtered.columns if c.startswith("Per 90 Minutes_")],
    errors="ignore",
)
scores_df        = pd.read_parquet(DATA_DIR / "scores_df.parquet")
team_df          = pd.read_parquet(DATA_DIR / "team_df.parquet")

# Enrich scores_df with Birth Year from players_filtered
_by_map = (
    players_filtered[["player", "team", "league", "Birth Year"]]
    .drop_duplicates(subset=["player", "team", "league"])
)
scores_df = scores_df.merge(_by_map, on=["player", "team", "league"], how="left")

LEAGUES_LIST   = sorted(players_filtered["league"].dropna().unique())
POSITIONS_LIST = sorted(players_filtered["new_position"].dropna().unique())
TEAMS_LIST     = sorted(players_filtered["team"].dropna().unique())
TEAM_LEAGUES   = sorted(team_df["league"].dropna().unique())
NATIONALITIES  = sorted(players_filtered["Nationality"].dropna().unique())

_by         = pd.to_numeric(players_filtered["Birth Year"], errors="coerce").dropna()
_by         = _by[_by > 1900]
BIRTH_YEARS = sorted(_by.unique().astype(int).tolist(), reverse=True)

_exclude_numeric = {"touches_total", "tackles_total", "Positional COG"}
ALL_NUMERIC_COLS = sorted([
    c for c in players_filtered.select_dtypes(include="number").columns
    if c not in _exclude_numeric
])

DEFAULT_STAT1 = "Non-Penalty Expected Goals per 90"
DEFAULT_STAT2 = "Non-Penalty xG Overperformance"
DEFAULT_STAT3 = "Expected Assisted Goals per 90"

_radar_base = players_filtered.dropna(subset=ALL_KNN_FEATURES + ["new_position"]).copy()
df_pct = _radar_base.copy()
for _f in ALL_KNN_FEATURES:
    if _f in df_pct.columns:
        df_pct[_f] = df_pct.groupby("new_position")[_f].rank(pct=True)

knn_base = players_filtered.dropna(subset=ALL_KNN_FEATURES).copy().reset_index(drop=True)
_scaler  = StandardScaler()
_scaler.fit(knn_base[ALL_KNN_FEATURES])

def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

_pitch_df   = team_df.dropna(subset=list(COG_COL_MAP.values())).copy()
_pitch_opts = [
    {"label": f"{r['team']}  ({r['league']})", "value": r["team"]}
    for _, r in _pitch_df[["team","league"]].drop_duplicates()
    .sort_values(["league","team"]).iterrows()
]
_pitch_teams = _pitch_df["team"].unique().tolist()

# -----------------------------------------------------------------------------
# 3. Layout helpers
# -----------------------------------------------------------------------------

def dd(id_, options, value, width="220px", multi=False,
       searchable=False, placeholder=None, clearable=False):
    return dcc.Dropdown(
        id=id_, options=options, value=value,
        clearable=clearable, multi=multi,
        searchable=searchable, placeholder=placeholder,
        style={"width": width, "fontFamily": FONT_BODY, "fontSize": "13px"},
    )

def labeled(label_text, child):
    return html.Div([
        html.Label(label_text, style={
            "color": MUTED, "fontSize": "11px", "fontFamily": FONT_DATA,
            "letterSpacing": "1px", "textTransform": "uppercase",
            "marginBottom": "6px", "display": "block",
        }),
        child,
    ])

def filter_row(*children):
    return html.Div(list(children), style={
        "display": "flex", "flexDirection": "row",
        "alignItems": "flex-end", "flexWrap": "wrap",
        "gap": "16px", "marginBottom": "24px",
    })

def section_title(text, label=None):
    return html.Div([
        html.Div(label or "", className="section-num") if label else None,
        html.Hr(className="gold-rule"),
        html.H4(text, style={
            "color": TEXT, "fontFamily": FONT_HEAD,
            "fontWeight": "700", "fontSize": "22px",
            "marginBottom": "20px", "letterSpacing": "-0.3px",
        }),
    ])

def card(*children, extra_style=None):
    s = {
        "backgroundColor": SURFACE,
        "borderRadius": "6px",
        "padding": "28px 32px",
        "marginBottom": "24px",
        "border": f"1px solid {BORDER}",
        "boxShadow": "0 4px 24px rgba(0,0,0,0.4)",
    }
    if extra_style:
        s.update(extra_style)
    return html.Div(list(children), style=s)

def empty_fig(msg="Select filters above"):
    fig = go.Figure()
    fig.update_layout(
        **FIG_BASE,
        annotations=[dict(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(family=FONT_BODY, color=MUTED, size=14),
        )],
    )
    return fig

def _th_style(h, center_cols):
    return {
        "fontFamily": FONT_DATA, "color": MUTED,
        "padding": "10px 14px",
        "textAlign": "center" if h in center_cols else "left",
        "borderBottom": f"1px solid {GOLD}",
        "fontWeight": "400", "fontSize": "10px",
        "letterSpacing": "1.2px", "textTransform": "uppercase",
        "whiteSpace": "nowrap", "backgroundColor": SURFACE2,
    }

def _table_header(cols, center_cols=None):
    center_cols = center_cols or []
    return html.Thead(html.Tr([
        html.Th(h, style=_th_style(h, center_cols)) for h in cols
    ]))

# -----------------------------------------------------------------------------
# 4. Navbar
# -----------------------------------------------------------------------------
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            html.Div([
                html.Span("European Football", style={
                    "fontFamily": FONT_HEAD, "fontWeight": "800",
                    "fontSize": "16px", "color": TEXT, "letterSpacing": "-0.3px",
                }),
                html.Span(" 2025/26", style={
                    "fontFamily": FONT_DATA, "fontSize": "12px",
                    "color": GOLD, "marginLeft": "4px", "letterSpacing": "1px",
                }),
            ]),
            href="/", style={"textDecoration": "none"},
        ),
        dbc.Nav([
            dbc.NavLink("Player Leaderboards", href="/player-leaderboards", active="partial",
                        style={"color": MUTED, "fontFamily": FONT_DATA, "fontSize": "11px",
                               "letterSpacing": "0.8px", "textTransform": "uppercase"}),
            dbc.NavLink("Advanced Players",    href="/players",             active="partial",
                        style={"color": MUTED, "fontFamily": FONT_DATA, "fontSize": "11px",
                               "letterSpacing": "0.8px", "textTransform": "uppercase"}),
            dbc.NavLink("Team Leaderboards",   href="/team-leaderboards",   active="partial",
                        style={"color": MUTED, "fontFamily": FONT_DATA, "fontSize": "11px",
                               "letterSpacing": "0.8px", "textTransform": "uppercase"}),
            dbc.NavLink("Advanced Teams",      href="/teams",               active="partial",
                        style={"color": MUTED, "fontFamily": FONT_DATA, "fontSize": "11px",
                               "letterSpacing": "0.8px", "textTransform": "uppercase"}),
        ], navbar=True, style={"gap": "8px"}),
    ], fluid=True),
    color=BG, dark=True,
    style={
        "borderBottom": f"1px solid {BORDER}",
        "padding": "12px 0",
    },
)

# Page wrapper (adds global CSS + navbar padding)
def page_wrap(*content):
    return html.Div([
        dcc.Markdown(
            GLOBAL_CSS,
            dangerously_allow_html=True,
            style={"display": "none"},
            id="__css__",
        ),
        navbar,
        html.Div(list(content), style={"padding": "36px 40px", "maxWidth": "1400px", "margin": "0 auto"}),
    ], style={"backgroundColor": BG, "minHeight": "100vh"})

# -----------------------------------------------------------------------------
# 5. Home layout
# -----------------------------------------------------------------------------
def _home_card(title, body, href, btn_label):
    return dbc.Col(html.Div([
        html.Div(style={"height": "3px", "backgroundColor": GOLD,
                        "borderRadius": "2px 2px 0 0"}),
        html.Div([
            html.H5(title, style={
                "color": TEXT, "fontFamily": FONT_HEAD,
                "fontWeight": "700", "fontSize": "16px",
                "marginBottom": "10px",
            }),
            html.P(body, style={
                "color": MUTED, "fontFamily": FONT_BODY,
                "fontSize": "13px", "lineHeight": "1.6", "marginBottom": "20px",
            }),
            html.A(btn_label, href=href, style={
                "fontFamily": FONT_DATA, "fontSize": "10px",
                "letterSpacing": "1.5px", "textTransform": "uppercase",
                "color": GOLD, "textDecoration": "none",
                "borderBottom": f"1px solid {GOLD}",
                "paddingBottom": "2px",
            }),
        ], style={"padding": "20px 22px 24px"}),
    ], style={
        "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
        "borderRadius": "6px", "height": "100%",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.35)",
    }), md=3)

home_layout = page_wrap(
    # Hero
    html.Div([
        html.Div(style={
            "width": "48px", "height": "3px",
            "backgroundColor": GOLD, "marginBottom": "24px",
        }),
        html.H1(
            "Exploring Europe's Top 5 Football Leagues",
            style={
                "fontFamily": FONT_HEAD, "fontWeight": "800",
                "fontSize": "clamp(28px, 4vw, 48px)",
                "color": TEXT, "lineHeight": "1.1",
                "letterSpacing": "-1px", "marginBottom": "4px",
            },
        ),
        html.H2("in the 2025/26 Season", style={
            "fontFamily": FONT_HEAD, "fontWeight": "700",
            "fontSize": "clamp(22px, 3vw, 36px)",
            "color": GOLD, "marginBottom": "12px",
            "letterSpacing": "-0.5px",
        }),
        html.P("by Isaiah Woram", style={
            "fontFamily": FONT_DATA, "fontSize": "12px",
            "color": MUTED, "letterSpacing": "2px",
            "textTransform": "uppercase", "marginBottom": "48px",
        }),
    ]),

    # Four section cards
    dbc.Row([
        _home_card(
            "Standard Player Leaderboards",
            "Filter by league, position, nationality, team, and year of birth. Sort by any stat in the dataset.",
            "/player-leaderboards", "Explore Players",
        ),
        _home_card(
            "Advanced Player Visualizations",
            "Custom scatter, radar charts, player rating leaderboard, KNN similarity finder, and COG map.",
            "/players", "Advanced Players",
        ),
        _home_card(
            "Standard Team Leaderboards",
            "Sort all 96 teams across the Top 5 leagues by goals, xG, rotation, COG, and more.",
            "/team-leaderboards", "Explore Teams",
        ),
        _home_card(
            "Advanced Team Visualizations",
            "Touch vs Tackle COG, tactics vs output scatter, and side-by-side positional COG pitch maps.",
            "/teams", "Advanced Teams",
        ),
    ], className="g-3", style={"marginBottom": "56px"}),

    # Memoriam footnote -- plain, not italic
    html.Div([
        html.Div(style={
            "width": "28px", "height": "2px",
            "backgroundColor": BORDER, "marginBottom": "20px",
        }),
        html.P(
            "After Opta sold out to FIFA and billion-dollar betting companies, "
            "rejecting data democratization and creative outlets for the average football nerd, "
            "I wanted to memorialize the FBref 2025/26 domestic season data that remains accessible "
            "for us commoners. If this sounds of any intrigue, please take a look! "
            "For questions or feedback, feel free to email: idw2005@nyu.edu",
            style={
                "color": MUTED, "fontFamily": FONT_BODY,
                "fontSize": "15px", "lineHeight": "1.8",
                "maxWidth": "680px",
            },
        ),
    ], style={"marginBottom": "40px"}),

    html.Hr(style={"borderColor": BORDER, "marginBottom": "16px"}),
    html.P(
        "Powered by FBref, soccerdata, Plotly, and Dash",
        style={
            "color": BORDER, "fontFamily": FONT_DATA,
            "fontSize": "10px", "letterSpacing": "1px",
            "textTransform": "uppercase",
        },
    ),
)

# -----------------------------------------------------------------------------
# 6. Standard Player Leaderboards layout  (drawer + pagination redesign)
# -----------------------------------------------------------------------------
_stat_opts = [{"label": c, "value": c} for c in ALL_NUMERIC_COLS]

_default_sort_opts = [
    {"label": "Stat 1 - Descending", "value": DEFAULT_STAT1 + "|desc"},
    {"label": "Stat 1 - Ascending",  "value": DEFAULT_STAT1 + "|asc"},
    {"label": "Stat 2 - Descending", "value": DEFAULT_STAT2 + "|desc"},
    {"label": "Stat 2 - Ascending",  "value": DEFAULT_STAT2 + "|asc"},
    {"label": "Stat 3 - Descending", "value": DEFAULT_STAT3 + "|desc"},
    {"label": "Stat 3 - Ascending",  "value": DEFAULT_STAT3 + "|asc"},
]

_plb_drawer = html.Div([
    html.Div([
        html.Span("Filters & Stats", style={
            "fontFamily": FONT_DATA, "fontSize": "11px", "letterSpacing": "1px",
            "textTransform": "uppercase", "color": TEXT,
        }),
    ], style={"padding": "16px 20px", "borderBottom": f"1px solid {BORDER}",
              "backgroundColor": SURFACE2}),

    html.Div([
        html.Div("Filters", style={
            "fontFamily": FONT_DATA, "fontSize": "10px", "letterSpacing": "1.2px",
            "textTransform": "uppercase", "color": MUTED, "marginBottom": "14px",
        }),
        labeled("League", dd("plb_league",
            [{"label": "All Leagues", "value": "ALL"}] +
            [{"label": lg, "value": lg} for lg in LEAGUES_LIST], "ALL", width="100%")),
        html.Div(style={"height": "12px"}),
        labeled("Position", dd("plb_position",
            [{"label": "All Positions", "value": "ALL"}] +
            [{"label": p, "value": p} for p in POSITIONS_LIST], "ALL", width="100%")),
        html.Div(style={"height": "12px"}),
        labeled("Team", dd("plb_team",
            [{"label": "All Teams", "value": "ALL"}] +
            [{"label": t, "value": t} for t in TEAMS_LIST],
            "ALL", width="100%", searchable=True)),
        html.Div(style={"height": "12px"}),
        labeled("Nationality", dd("plb_nat",
            [{"label": "All", "value": "ALL"}] +
            [{"label": n, "value": n} for n in NATIONALITIES],
            "ALL", width="100%", searchable=True)),
        html.Div(style={"height": "12px"}),
        html.Div([
            html.Div(labeled("Born From", dd("plb_yob_min",
                [{"label": "Any", "value": "ANY"}] +
                [{"label": str(y), "value": y} for y in BIRTH_YEARS],
                "ANY", width="100%")), style={"flex": "1"}),
            html.Div(labeled("Born To", dd("plb_yob_max",
                [{"label": "Any", "value": "ANY"}] +
                [{"label": str(y), "value": y} for y in BIRTH_YEARS],
                "ANY", width="100%")), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "10px"}),
    ], style={"padding": "20px", "borderBottom": f"1px solid {BORDER}"}),

    html.Div([
        html.Div("Columns", style={
            "fontFamily": FONT_DATA, "fontSize": "10px", "letterSpacing": "1.2px",
            "textTransform": "uppercase", "color": MUTED, "marginBottom": "14px",
        }),
        labeled("Stat 1", dd("plb_s1", _stat_opts, DEFAULT_STAT1, width="100%", searchable=True)),
        html.Div(style={"height": "12px"}),
        labeled("Stat 2", dd("plb_s2", _stat_opts, DEFAULT_STAT2, width="100%", searchable=True)),
        html.Div(style={"height": "12px"}),
        labeled("Stat 3", dd("plb_s3", _stat_opts, DEFAULT_STAT3, width="100%", searchable=True)),
    ], style={"padding": "20px", "borderBottom": f"1px solid {BORDER}"}),

    html.Div([
        html.Div("Sort By", style={
            "fontFamily": FONT_DATA, "fontSize": "10px", "letterSpacing": "1.2px",
            "textTransform": "uppercase", "color": MUTED, "marginBottom": "14px",
        }),
        dd("plb_sort", _default_sort_opts, _default_sort_opts[0]["value"], width="100%"),
    ], style={"padding": "20px"}),

], style={
    "width": "280px", "flexShrink": "0",
    "backgroundColor": SURFACE,
    "borderLeft": f"1px solid {BORDER}",
    "overflowY": "auto",
})

_plb_main = html.Div([
    html.Div([
        html.Span("Standard Player Leaderboards", style={
            "fontFamily": FONT_HEAD, "fontWeight": "700",
            "fontSize": "18px", "color": TEXT, "letterSpacing": "-0.3px",
        }),
    ], style={"padding": "18px 24px", "borderBottom": f"1px solid {BORDER}",
              "backgroundColor": SURFACE2}),
    html.Div(id="plb_table", style={"flex": "1", "overflowY": "auto", "padding": "0 24px"}),
    html.Div(id="plb_pagination", style={
        "borderTop": f"1px solid {BORDER}", "padding": "12px 24px",
        "backgroundColor": SURFACE2,
    }),
], style={"flex": "1", "display": "flex", "flexDirection": "column",
          "overflow": "hidden", "minWidth": "0"})

player_lb_layout = page_wrap(
    dcc.Store(id="plb_page", data=1),
    html.Div([_plb_main, _plb_drawer], style={
        "display": "flex", "flexDirection": "row",
        "backgroundColor": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "6px",
        "overflow": "hidden",
        "height": "82vh",
        "boxShadow": "0 4px 24px rgba(0,0,0,0.4)",
    }),
)

# -----------------------------------------------------------------------------
# 7. Advanced Player Visualizations layout
# -----------------------------------------------------------------------------
_numeric_opts = [{"label": c, "value": c} for c in ALL_NUMERIC_COLS]

adv_players_layout = page_wrap(

    card(
        section_title("Player Scatter", "01"),
        filter_row(
            labeled("League", dd("ps_league",
                [{"label": "All Leagues", "value": "ALL"}] +
                [{"label": lg, "value": lg} for lg in LEAGUES_LIST], "ALL")),
            labeled("Position", dd("ps_position",
                [{"label": "All Positions", "value": "ALL"}] +
                [{"label": p, "value": p} for p in POSITIONS_LIST], "ALL")),
            labeled("X Axis", dd("ps_x", _numeric_opts,
                "Expected Goals", width="280px", searchable=True)),
            labeled("Y Axis", dd("ps_y", _numeric_opts,
                "Expected Assisted Goals", width="280px", searchable=True)),
        ),
        dcc.Graph(id="ps_scatter"),
    ),

    card(
        section_title("Player Radar Charts", "02"),
        filter_row(
            labeled("Profile", dd("radar_profile",
                [{"label": k, "value": k} for k in FEATURE_SETS], "Attack", width="180px")),
            labeled("Position", dd("radar_position",
                [{"label": p, "value": p} for p in POSITIONS_LIST],
                POSITIONS_LIST[0] if POSITIONS_LIST else None, width="180px")),
            labeled("Players (max 3)", dd("radar_players", [], None,
                width="420px", multi=True, searchable=True,
                placeholder="Type player name(s)...")),
        ),
        dcc.Graph(id="radar_chart"),
    ),

    card(
        section_title("Player Rating Leaderboard", "03"),
        filter_row(
            labeled("League", dd("lb_league",
                [{"label": "All Leagues", "value": "ALL"}] +
                [{"label": lg, "value": lg} for lg in LEAGUES_LIST], "ALL")),
            labeled("Position", dd("lb_position",
                [{"label": "All Positions", "value": "ALL"}] +
                [{"label": p, "value": p} for p in POSITIONS_LIST], "ALL")),
            labeled("Sort By", dd("lb_sort",
                [{"label": k, "value": v} for k, v in SCORE_COLS.items()],
                "composite_score", width="180px")),
            labeled("Order", dd("lb_direction", ORDER_OPTIONS, "desc", width="160px")),
            labeled("Show", dd("lb_n",
                [{"label": str(n), "value": n} for n in [10, 20, 30, 50]], 20, width="100px")),
        ),
        html.Div([
            html.Span("Minimum Scores", style={
                "fontFamily": FONT_DATA, "color": MUTED, "fontSize": "11px",
                "letterSpacing": "1px", "textTransform": "uppercase",
                "marginRight": "20px", "lineHeight": "36px",
            }),
            html.Div([
                html.Label("Attack", style={"fontFamily": FONT_DATA, "color": MUTED,
                                            "fontSize": "10px", "letterSpacing": "1px",
                                            "textTransform": "uppercase"}),
                dcc.Slider(id="lb_min_atk", min=0, max=90, step=5, value=0,
                           marks={i: {"label": str(i), "style": {"color": MUTED, "fontSize": "10px"}}
                                  for i in range(0, 91, 10)},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"width": "200px", "marginRight": "32px"}),
            html.Div([
                html.Label("Progression", style={"fontFamily": FONT_DATA, "color": MUTED,
                                                  "fontSize": "10px", "letterSpacing": "1px",
                                                  "textTransform": "uppercase"}),
                dcc.Slider(id="lb_min_prg", min=0, max=90, step=5, value=0,
                           marks={i: {"label": str(i), "style": {"color": MUTED, "fontSize": "10px"}}
                                  for i in range(0, 91, 10)},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"width": "200px", "marginRight": "32px"}),
            html.Div([
                html.Label("Defense", style={"fontFamily": FONT_DATA, "color": MUTED,
                                              "fontSize": "10px", "letterSpacing": "1px",
                                              "textTransform": "uppercase"}),
                dcc.Slider(id="lb_min_def", min=0, max=90, step=5, value=0,
                           marks={i: {"label": str(i), "style": {"color": MUTED, "fontSize": "10px"}}
                                  for i in range(0, 91, 10)},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"width": "200px"}),
        ], style={
            "display": "flex", "flexDirection": "row", "flexWrap": "wrap",
            "alignItems": "flex-start", "backgroundColor": SURFACE2,
            "padding": "16px 20px", "borderRadius": "4px",
            "marginBottom": "20px", "border": f"1px solid {BORDER}",
        }),
        html.Div(id="lb_summary", style={
            "fontFamily": FONT_DATA, "color": MUTED, "fontSize": "11px",
            "letterSpacing": "0.8px", "textTransform": "uppercase", "marginBottom": "16px",
        }),
        html.Div(id="lb_table"),
    ),

    card(
        section_title("Similar Players (KNN)", "04"),
        filter_row(
            labeled("Position", dd("knn_position",
                [{"label": p, "value": p} for p in POSITIONS_LIST],
                POSITIONS_LIST[0] if POSITIONS_LIST else None, width="160px")),
            labeled("League", dd("knn_league",
                [{"label": "All leagues", "value": "ALL"}] +
                [{"label": lg, "value": lg} for lg in LEAGUES_LIST], "ALL")),
            labeled("Team", dd("knn_team", [], "ALL", width="200px")),
            labeled("Player", dd("knn_player", [], None,
                width="280px", searchable=True, placeholder="Type to search...")),
            labeled("# Similar", dd("knn_n",
                [{"label": str(n), "value": n} for n in [5, 8, 10, 15]], 8, width="100px")),
        ),
        html.Div(id="knn_header", style={
            "fontFamily": FONT_HEAD, "color": GOLD,
            "fontSize": "16px", "fontWeight": "700", "marginBottom": "12px",
        }),
        dcc.Graph(id="knn_radar"),
        html.Div(id="knn_table", style={"marginTop": "20px"}),
    ),

    card(
        section_title("Touch COG vs. Tackle COG -- Players", "05"),
        filter_row(
            labeled("Color By", dd("cog_color", [
                {"label": "Position", "value": "position"},
                {"label": "Team",     "value": "team"},
            ], "position", width="200px")),
            labeled("League", dd("cog_league",
                [{"label": "All Leagues", "value": "ALL"}] +
                [{"label": lg, "value": lg} for lg in LEAGUES_LIST], "ALL")),
            labeled("Team", dd("cog_team",
                [{"label": "All Teams", "value": "ALL"}] +
                [{"label": t, "value": t} for t in TEAMS_LIST], "ALL", width="220px")),
        ),
        dcc.Graph(id="cog_scatter"),
    ),
)

# -----------------------------------------------------------------------------
# 8. Standard Team Leaderboards layout
# -----------------------------------------------------------------------------
team_lb_layout = page_wrap(card(
    section_title("Standard Team Leaderboards", "01"),
    filter_row(
        labeled("League", dd("tlb_league",
            [{"label": "All Leagues", "value": "ALL"}] +
            [{"label": lg, "value": lg} for lg in TEAM_LEAGUES], "ALL")),
        labeled("Sort By", dd("tlb_sort", TEAM_SORT_OPTIONS, "xg", width="200px")),
        labeled("Order", dd("tlb_dir", ORDER_OPTIONS, "desc", width="160px")),
        labeled("Show", dd("tlb_n",
            [{"label": str(n), "value": n} for n in [10, 20, 30, 50, 96]], 20, width="100px")),
    ),
    html.Div(id="tlb_summary", style={
        "fontFamily": FONT_DATA, "color": MUTED, "fontSize": "11px",
        "letterSpacing": "0.8px", "textTransform": "uppercase", "marginBottom": "16px",
    }),
    html.Div(id="tlb_table"),
))

# -----------------------------------------------------------------------------
# 9. Advanced Teams layout
# -----------------------------------------------------------------------------
adv_teams_layout = page_wrap(

    card(
        section_title("Team Touch COG vs. Tackle COG", "01"),
        html.P(
            "Each dot is a team, colored by total xG. "
            "Axes are league-relative z-scores (0 = league average). "
            "X = where players operate on the pitch. Y = where they win the ball.",
            style={"color": MUTED, "fontFamily": FONT_BODY,
                   "fontSize": "13px", "marginBottom": "20px", "lineHeight": "1.7"},
        ),
        filter_row(
            labeled("League", dd("tc_league",
                [{"label": lg, "value": lg} for lg in TEAM_LEAGUES],
                TEAM_LEAGUES[0] if TEAM_LEAGUES else None, width="240px")),
        ),
        dcc.Graph(id="tc_scatter"),
    ),

    card(
        section_title("Effect of Tactics & Rotation on Output", "02"),
        filter_row(
            labeled("X Axis", dd("mv_x",
                [{"label": k, "value": k} for k in X_OPTIONS],
                "Attacking Compactness", width="240px")),
            labeled("Y Axis", dd("mv_y",
                [{"label": k, "value": k} for k in Y_OPTIONS], "xGoals", width="160px")),
            labeled("League", dd("mv_league",
                [{"label": "All Leagues", "value": "ALL"}] +
                [{"label": lg, "value": lg} for lg in TEAM_LEAGUES], "ALL")),
        ),
        html.H5(id="mv_title", style={
            "color": TEXT, "fontFamily": FONT_HEAD,
            "fontWeight": "700", "marginBottom": "12px",
        }),
        dcc.Graph(id="mv_scatter"),
    ),

    card(
        section_title("Side-by-Side Positional COG Pitch Maps", "03"),
        html.P(
            "Dots represent the average positional center of gravity for each position group, "
            "plotted on a 4-3-3 template. Compare any two teams across any leagues.",
            style={"color": MUTED, "fontFamily": FONT_BODY,
                   "fontSize": "13px", "marginBottom": "20px", "lineHeight": "1.7"},
        ),
        filter_row(
            labeled("Team A", dd("pitch_team_a", _pitch_opts,
                _pitch_teams[0] if _pitch_teams else None, width="300px", searchable=True)),
            labeled("Team B", dd("pitch_team_b", _pitch_opts,
                _pitch_teams[1] if len(_pitch_teams) > 1 else None,
                width="300px", searchable=True)),
        ),
        dcc.Graph(id="pitch_map"),
    ),
)

# -----------------------------------------------------------------------------
# 10. App init & routing
# -----------------------------------------------------------------------------
app    = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
              suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page_content"),
    # Global CSS injected here so it's always present regardless of page
    dcc.Markdown(
        GLOBAL_CSS,
        dangerously_allow_html=True,
        style={"display": "none"},
    ),
], style={"backgroundColor": BG})

@app.callback(Output("page_content", "children"), Input("url", "pathname"))
def route(pathname):
    if pathname == "/player-leaderboards": return player_lb_layout
    if pathname == "/players":             return adv_players_layout
    if pathname == "/team-leaderboards":   return team_lb_layout
    if pathname == "/teams":               return adv_teams_layout
    return home_layout

# -----------------------------------------------------------------------------
# 11. Standard Player Leaderboard callbacks  (drawer + pagination)
# -----------------------------------------------------------------------------

PAGE_SIZE = 50

# Reset page to 1 whenever any filter or stat/sort changes
@app.callback(
    Output("plb_page", "data"),
    Input("plb_league",   "value"),
    Input("plb_position", "value"),
    Input("plb_nat",      "value"),
    Input("plb_team",     "value"),
    Input("plb_yob_min",  "value"),
    Input("plb_yob_max",  "value"),
    Input("plb_sort",     "value"),
    Input("plb_s1",       "value"),
    Input("plb_s2",       "value"),
    Input("plb_s3",       "value"),
)
def reset_plb_page(*_):
    return 1


# Update sort dropdown options when stats change
@app.callback(
    Output("plb_sort", "options"),
    Output("plb_sort", "value"),
    Input("plb_s1", "value"),
    Input("plb_s2", "value"),
    Input("plb_s3", "value"),
)
def update_plb_sort_options(s1, s2, s3):
    opts = []
    for label, stat in [("Stat 1", s1), ("Stat 2", s2), ("Stat 3", s3)]:
        if stat:
            opts.append({"label": f"{label} - Descending", "value": stat + "|desc"})
            opts.append({"label": f"{label} - Ascending",  "value": stat + "|asc"})
    default = opts[0]["value"] if opts else None
    return opts, default


def _apply_plb_filters(league, position, nat, team, yob_min, yob_max):
    sub = players_filtered.copy()
    if league   != "ALL": sub = sub[sub["league"]       == league]
    if position != "ALL": sub = sub[sub["new_position"] == position]
    if nat      != "ALL": sub = sub[sub["Nationality"]  == nat]
    if team     != "ALL": sub = sub[sub["team"]         == team]
    by = pd.to_numeric(sub["Birth Year"], errors="coerce")
    if yob_min != "ANY":
        sub = sub[by >= int(yob_min)]
        by  = pd.to_numeric(sub["Birth Year"], errors="coerce")
    if yob_max != "ANY":
        sub = sub[by <= int(yob_max)]
    return sub


# Build a league -> country-code map from league names
_LEAGUE_CODE = {
    lg: lg.split("-")[0].strip() if "-" in lg else lg[:3].upper()
    for lg in LEAGUES_LIST
}
# Hardcode the known leagues for clean codes
for _lg, _code in [
    ("ENG-Premier League", "ENG"), ("ESP-La Liga", "ESP"),
    ("GER-Bundesliga", "GER"),     ("ITA-Serie A", "ITA"),
    ("FRA-Ligue 1", "FRA"),
]:
    if _lg in _LEAGUE_CODE:
        _LEAGUE_CODE[_lg] = _code


@app.callback(
    Output("plb_table",      "children"),
    Output("plb_pagination", "children"),
    Input("plb_league",      "value"),
    Input("plb_position",    "value"),
    Input("plb_nat",         "value"),
    Input("plb_team",        "value"),
    Input("plb_yob_min",     "value"),
    Input("plb_yob_max",     "value"),
    Input("plb_sort",        "value"),
    Input("plb_s1",          "value"),
    Input("plb_s2",          "value"),
    Input("plb_s3",          "value"),
    Input("plb_page",        "data"),
)
def update_player_leaderboard(league, position, nat, team,
                               yob_min, yob_max, sort_val,
                               s1, s2, s3, page):
    sub = _apply_plb_filters(league, position, nat, team, yob_min, yob_max)

    # Parse sort value "col|dir"
    sort_col, direction = None, "desc"
    if sort_val and "|" in sort_val:
        sort_col, direction = sort_val.rsplit("|", 1)

    if sort_col and sort_col in sub.columns:
        sub = sub.sort_values(sort_col, ascending=(direction == "asc"))

    total      = len(sub)
    page       = page or 1
    total_pages = max(1, -(-total // PAGE_SIZE))  # ceiling division
    page       = min(page, total_pages)
    start      = (page - 1) * PAGE_SIZE
    page_sub   = sub.iloc[start: start + PAGE_SIZE]

    # Columns
    fixed  = ["player", "team", "league", "new_position", "Birth Year", "Minutes Played"]
    extras = [c for c in [s1, s2, s3] if c and c in sub.columns]
    all_c  = list(dict.fromkeys(fixed + [c for c in extras if c not in fixed]))
    all_c  = [c for c in all_c if c in sub.columns]
    rename = {
        "player": "Player", "team": "Team", "league": "League",
        "new_position": "Position", "Birth Year": "Born", "Minutes Played": "Min",
    }
    hdrs   = [rename.get(c, c) for c in all_c]

    # Styles
    base_td = {
        "fontFamily": FONT_BODY, "padding": "9px 14px",
        "borderBottom": f"1px solid {SURFACE2}", "fontSize": "13px",
        "color": TEXT, "textAlign": "left", "verticalAlign": "middle",
    }
    mid_td  = {**base_td, "color": MUTED, "textAlign": "center"}
    num_td  = {**mid_td, "fontFamily": FONT_DATA, "color": GOLD_L}
    rank_td = {**mid_td, "color": GOLD, "fontFamily": FONT_DATA, "fontWeight": "700", "width": "36px"}

    def _fmt(val):
        if isinstance(val, float) and not pd.isna(val):
            return f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
        if pd.isna(val):
            return "--"
        return str(val)

    rows = []
    for rank, (_, row) in enumerate(page_sub.iterrows(), start=start + 1):
        cells = [html.Td(rank, style=rank_td)]
        for c, h in zip(all_c, hdrs):
            raw = row.get(c, "")

            # Team cell: name on top, country code below
            if c == "team":
                lg_val  = row.get("league", "")
                code    = _LEAGUE_CODE.get(lg_val, lg_val[:3].upper() if lg_val else "")
                cell_content = html.Div([
                    html.Span(str(raw), style={"display": "block", "color": TEXT, "fontSize": "13px"}),
                    html.Span(code, style={
                        "display": "block", "color": MUTED,
                        "fontSize": "10px", "fontFamily": FONT_DATA,
                        "letterSpacing": "0.8px", "marginTop": "1px",
                    }),
                ])
                style = {**base_td, "minWidth": "110px"}
                if c == sort_col:
                    style = {**style, "backgroundColor": GOLD + "12"}
                cells.append(html.Td(cell_content, style=style))
                continue

            # League column: skip (info already in team cell)
            if c == "league":
                continue

            # Player name
            if c == "player":
                style = {**base_td, "fontWeight": "500", "minWidth": "120px"}
                if c == sort_col:
                    style = {**style, "backgroundColor": GOLD + "12"}
                cells.append(html.Td(_fmt(raw), style=style))
                continue

            # Stat columns (highlighted)
            if c in extras:
                style = {**num_td}
                if c == sort_col:
                    style = {**style, "backgroundColor": GOLD + "12", "color": GOLD_L}
                cells.append(html.Td(_fmt(raw), style=style))
                continue

            # Everything else (Position, Born, Min)
            cells.append(html.Td(_fmt(raw), style=mid_td))

        rows.append(html.Tr(cells))

    # Build header -- skip "League" since it's folded into Team
    display_hdrs = ["#"] + [h for h, c in zip(hdrs, all_c) if c != "league"]
    center_hdrs  = ["#", "Position", "Born", "Min"] + [rename.get(c, c) for c in extras]
    table = html.Table(
        [_table_header(display_hdrs, center_cols=center_hdrs), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": SURFACE},
    )

    # Pagination bar
    btn_base = {
        "fontFamily": FONT_DATA, "fontSize": "11px", "letterSpacing": "0.5px",
        "padding": "5px 12px", "border": f"1px solid {BORDER}",
        "borderRadius": "4px", "cursor": "pointer",
        "backgroundColor": SURFACE2, "color": MUTED,
        "display": "inline-block", "margin": "0 2px",
    }
    btn_active = {**btn_base, "backgroundColor": GOLD, "color": BG,
                  "border": f"1px solid {GOLD}", "fontWeight": "700"}
    btn_disabled = {**btn_base, "opacity": "0.35", "cursor": "default"}

    page_info = html.Span(
        f"Page {page} of {total_pages} - 50 per page",
        style={"fontFamily": FONT_DATA, "fontSize": "11px",
               "color": MUTED, "letterSpacing": "0.5px"},
    )

    def _pgbtn(label, target_page, disabled=False, active=False):
        style = btn_active if active else (btn_disabled if disabled else btn_base)
        if disabled or active:
            return html.Span(label, style=style)
        return html.Span(label, id={"type": "plb_pgbtn", "page": target_page},
                         n_clicks=0, style=style)

    # Build page number buttons (show up to 5 around current page)
    pg_buttons = [_pgbtn("Prev", page - 1, disabled=(page <= 1))]
    pages_to_show = sorted(set(
        [1] +
        list(range(max(2, page - 1), min(total_pages, page + 2))) +
        ([total_pages] if total_pages > 1 else [])
    ))
    prev_p = None
    for p in pages_to_show:
        if prev_p and p - prev_p > 1:
            pg_buttons.append(html.Span("...", style={
                "color": MUTED, "fontFamily": FONT_DATA,
                "fontSize": "11px", "padding": "0 4px",
            }))
        pg_buttons.append(_pgbtn(str(p), p, active=(p == page)))
        prev_p = p
    pg_buttons.append(_pgbtn("Next", page + 1, disabled=(page >= total_pages)))

    pagination = html.Div([
        page_info,
        html.Div(pg_buttons, style={"display": "inline-flex", "alignItems": "center",
                                    "gap": "2px", "marginLeft": "auto"}),
    ], style={"display": "flex", "alignItems": "center"})

    return table, pagination


@app.callback(
    Output("plb_page", "data", allow_duplicate=True),
    Input({"type": "plb_pgbtn", "page": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def change_plb_page(n_clicks_list):
    if not ctx.triggered_id or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    return ctx.triggered_id["page"]

# -----------------------------------------------------------------------------
# 12. Player scatter callback
# -----------------------------------------------------------------------------
@app.callback(
    Output("ps_scatter", "figure"),
    Input("ps_league",   "value"),
    Input("ps_position", "value"),
    Input("ps_x",        "value"),
    Input("ps_y",        "value"),
)
def update_player_scatter(league, position, x_col, y_col):
    if not (x_col and y_col):
        return empty_fig()
    sub = players_filtered.copy()
    if league   != "ALL": sub = sub[sub["league"]       == league]
    if position != "ALL": sub = sub[sub["new_position"] == position]
    sub = sub.dropna(subset=[x_col, y_col])
    fig = go.Figure()
    for pos in POSITIONS_LIST:
        mask = sub["new_position"] == pos
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=sub.loc[mask, x_col], y=sub.loc[mask, y_col],
            mode="markers", name=pos,
            marker=dict(size=8, opacity=0.8,
                        color=POS_COLORS.get(pos, MUTED),
                        line=dict(width=0.5, color=BG)),
            text=sub.loc[mask, "player"],
            customdata=np.stack([sub.loc[mask, "team"],
                                  sub.loc[mask, "new_position"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>Team: %{customdata[0]}<br>"
                f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>"
            ),
        ))
    fig.update_layout(**FIG_BASE,
                      xaxis=dict(title=x_col, **AXIS_BASE),
                      yaxis=dict(title=y_col, **AXIS_BASE))
    return fig

# -----------------------------------------------------------------------------
# 13. Radar callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("radar_players", "options"),
    Output("radar_players", "value"),
    Input("radar_position", "value"),
)
def update_radar_players(position):
    sub  = df_pct[df_pct["new_position"] == position] if position else df_pct
    opts = [{"label": f"{r['player']} ({r['team']})", "value": r["player"]}
            for _, r in sub[["player","team"]].drop_duplicates().sort_values("player").iterrows()]
    return opts, []

@app.callback(
    Output("radar_chart",   "figure"),
    Input("radar_profile",  "value"),
    Input("radar_position", "value"),
    Input("radar_players",  "value"),
)
def update_radar(profile, position, players):
    players = (players or [])[:3]
    if not players:
        return empty_fig("Select up to 3 players above")
    feats   = [f for f in FEATURE_SETS.get(profile, []) if f in df_pct.columns]
    sub     = df_pct[df_pct["new_position"] == position] if position else df_pct
    palette = [GOLD, "#2f81f7", "#3fb950", "#f85149"]
    fig     = go.Figure()
    for i, player in enumerate(players):
        row = sub[sub["player"] == player]
        if row.empty:
            continue
        vals  = row[feats].mean().values.tolist()
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=feats + [feats[0]],
            fill="toself", name=player,
            line=dict(color=color, width=2),
            fillcolor=_hex_rgba(color, 0.16),
        ))
    fig.update_layout(
        **FIG_BASE,
        polar=dict(
            bgcolor=SURFACE2,
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(color=MUTED, size=9),
                            gridcolor=BORDER, linecolor=BORDER),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                             tickfont=dict(family=FONT_BODY, color=TEXT, size=10)),
        ),
    )
    return fig

# -----------------------------------------------------------------------------
# 14. Player Rating Leaderboard callback
# -----------------------------------------------------------------------------
@app.callback(
    Output("lb_summary",  "children"),
    Output("lb_table",    "children"),
    Input("lb_league",    "value"),
    Input("lb_position",  "value"),
    Input("lb_sort",      "value"),
    Input("lb_direction", "value"),
    Input("lb_n",         "value"),
    Input("lb_min_atk",   "value"),
    Input("lb_min_prg",   "value"),
    Input("lb_min_def",   "value"),
)
def update_rating_leaderboard(league, position, sort_col, direction, n,
                               min_atk, min_prg, min_def):
    sub = scores_df.copy()
    if league   != "ALL": sub = sub[sub["league"]       == league]
    if position != "ALL": sub = sub[sub["new_position"] == position]
    sub = sub[
        (sub["attack_score"]      >= min_atk) &
        (sub["progression_score"] >= min_prg) &
        (sub["defense_score"]     >= min_def)
    ]
    sub = sub.sort_values(sort_col, ascending=(direction == "asc")).head(n)

    active = []
    if league   != "ALL": active.append(league)
    if position != "ALL": active.append(position)
    if min_atk  > 0:      active.append(f"Atk >= {min_atk}")
    if min_prg  > 0:      active.append(f"Prg >= {min_prg}")
    if min_def  > 0:      active.append(f"Def >= {min_def}")
    summary = f"{len(sub)} players" + (" -- " + " -- ".join(active) if active else "")

    hdrs    = ["#","Player","Team","League","Position","Birth Year",
               "Min","Attack","Prg","Defense","Composite"]
    center  = ["League","Position","Birth Year","Min","Attack","Prg","Defense","Composite"]

    base_td = {"fontFamily": FONT_BODY, "padding": "10px 14px",
               "borderBottom": f"1px solid {SURFACE2}", "fontSize": "13px",
               "color": TEXT, "textAlign": "left"}
    mid_td  = {**base_td, "color": MUTED, "textAlign": "center"}

    def cs(row, col):
        base = {**mid_td, "fontWeight": "700",
                "fontFamily": FONT_DATA, "color": score_color(row[col])}
        if col == sort_col:
            base["backgroundColor"] = f"{GOLD}12"
        return base

    def fmt_by(val):
        try:
            v = int(float(val))
            return str(v) if v > 1900 else "--"
        except Exception:
            return "--"

    rows = []
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        rows.append(html.Tr([
            html.Td(rank,                              style={**mid_td, "color": GOLD, "fontFamily": FONT_DATA, "fontWeight": "700"}),
            html.Td(row["player"],                     style=base_td),
            html.Td(row["team"],                       style={**base_td, "color": "#cdd9e5"}),
            html.Td(row["league"],                     style=mid_td),
            html.Td(row["new_position"],               style=mid_td),
            html.Td(fmt_by(row.get("Birth Year", "")),  style=mid_td),
            html.Td(f"{row['Minutes Played']:.0f}",    style=mid_td),
            html.Td(f"{row['attack_score']:.1f}",      style=cs(row, "attack_score")),
            html.Td(f"{row['progression_score']:.1f}", style=cs(row, "progression_score")),
            html.Td(f"{row['defense_score']:.1f}",     style=cs(row, "defense_score")),
            html.Td(f"{row['composite_score']:.1f}",   style=cs(row, "composite_score")),
        ]))

    table = html.Table(
        [_table_header(hdrs, center_cols=center), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": SURFACE},
    )
    return summary, table

# -----------------------------------------------------------------------------
# 15. KNN callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("knn_team",    "options"),
    Output("knn_team",    "value"),
    Input("knn_league",   "value"),
    Input("knn_position", "value"),
)
def update_knn_teams(league, position):
    sub = knn_base.copy()
    if position:        sub = sub[sub["new_position"] == position]
    if league != "ALL": sub = sub[sub["league"]       == league]
    opts = [{"label": "All teams", "value": "ALL"}] + \
           [{"label": t, "value": t} for t in sorted(sub["team"].unique())]
    return opts, "ALL"

@app.callback(
    Output("knn_player",  "options"),
    Output("knn_player",  "value"),
    Input("knn_position", "value"),
    Input("knn_league",   "value"),
    Input("knn_team",     "value"),
)
def update_knn_players(position, league, team):
    sub = knn_base.copy()
    if position:        sub = sub[sub["new_position"] == position]
    if league != "ALL": sub = sub[sub["league"]       == league]
    if team   != "ALL": sub = sub[sub["team"]         == team]
    opts = [{"label": p, "value": p} for p in sorted(sub["player"].unique())]
    return opts, None

@app.callback(
    Output("knn_header", "children"),
    Output("knn_radar",  "figure"),
    Output("knn_table",  "children"),
    Input("knn_player",  "value"),
    Input("knn_position","value"),
    Input("knn_n",       "value"),
)
def update_knn(player, position, n_neighbors):
    if not player or not position:
        return "Select a player above.", empty_fig(), ""

    pos_df = knn_base[knn_base["new_position"] == position].copy().reset_index(drop=True)
    if player not in pos_df["player"].values:
        return f"'{player}' not found in {position}.", empty_fig(), ""

    pos_matrix         = _scaler.transform(pos_df[ALL_KNN_FEATURES])
    k                  = min(n_neighbors + 1, len(pos_df))
    knn_model          = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_model.fit(pos_matrix)
    player_idx         = pos_df[pos_df["player"] == player].index[0]
    distances, indices = knn_model.kneighbors(pos_matrix[player_idx].reshape(1, -1))

    similar = pos_df.iloc[indices[0]].copy()
    similar["similarity_distance"] = distances[0]
    similar = similar[similar["player"] != player].head(n_neighbors)

    radar_df = pos_df.copy()
    for feat in ALL_KNN_FEATURES:
        if feat in radar_df.columns:
            radar_df[feat] = radar_df.groupby("new_position")[feat].rank(pct=True)

    def get_pct(name):
        rows = radar_df[radar_df["player"] == name]
        return None if rows.empty else rows[ALL_KNN_FEATURES].mean()

    palette = [GOLD, "#2f81f7", "#3fb950", "#f85149"]
    fig     = go.Figure()
    q_pct   = get_pct(player)
    if q_pct is not None:
        fig.add_trace(go.Scatterpolar(
            r=q_pct.values.tolist() + [q_pct.values[0]],
            theta=ALL_KNN_FEATURES + [ALL_KNN_FEATURES[0]],
            fill="toself", name=player,
            line=dict(color=palette[0], width=2.5), opacity=0.9,
            fillcolor=_hex_rgba(palette[0], 0.16),
        ))
    for i, (_, sim_row) in enumerate(similar.head(3).iterrows()):
        sim_pct = get_pct(sim_row["player"])
        if sim_pct is None:
            continue
        fig.add_trace(go.Scatterpolar(
            r=sim_pct.values.tolist() + [sim_pct.values[0]],
            theta=ALL_KNN_FEATURES + [ALL_KNN_FEATURES[0]],
            fill="toself", name=sim_row["player"],
            line=dict(color=palette[i + 1], width=1.5), opacity=0.65,
            fillcolor=_hex_rgba(palette[i + 1], 0.12),
        ))
    fig.update_layout(
        **FIG_BASE,
        polar=dict(
            bgcolor=SURFACE2,
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(color=MUTED, size=9),
                            gridcolor=BORDER, linecolor=BORDER),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                             tickfont=dict(family=FONT_BODY, color=TEXT, size=9)),
        ),
        title=dict(text=f"{player} vs. Top 3 Similar {position}s",
                   font=dict(family=FONT_HEAD, color=TEXT, size=14, weight=700)),
    )

    query_row   = knn_base[knn_base["player"] == player].iloc[0]
    header_text = f"Players most similar to {player} ({query_row['team']} -- {query_row['league']})"

    base_td = {"fontFamily": FONT_BODY, "padding": "10px 14px",
               "borderBottom": f"1px solid {SURFACE2}", "fontSize": "13px"}
    table_rows = []
    for rank, (_, row) in enumerate(similar.iterrows(), start=1):
        sim_score = max(0, round((1 - row["similarity_distance"] / 10) * 100, 1))
        sc = "#3fb950" if sim_score >= 80 else (GOLD if sim_score >= 60 else "#f85149")
        age_display = str(row["Age"]).split("-")[0]
        table_rows.append(html.Tr([
            html.Td(f"#{rank}", style={**base_td, "color": GOLD,
                                       "fontFamily": FONT_DATA, "fontWeight": "700"}),
            html.Td(row["player"],                 style={**base_td, "color": TEXT}),
            html.Td(row["team"],                   style={**base_td, "color": MUTED}),
            html.Td(row["league"],                 style={**base_td, "color": MUTED, "textAlign": "center"}),
            html.Td(age_display,                   style={**base_td, "color": MUTED, "textAlign": "center"}),
            html.Td(f"{row['Minutes Played']:.0f}",style={**base_td, "color": MUTED, "textAlign": "center"}),
            html.Td(f"{sim_score}%",               style={**base_td, "color": sc,
                                                           "fontFamily": FONT_DATA,
                                                           "fontWeight": "700", "textAlign": "center"}),
        ], style={"borderBottom": f"1px solid {SURFACE2}"}))

    table = html.Table([
        _table_header(["Rank","Player","Team","League","Age","Min","Similarity"],
                      center_cols=["League","Age","Min","Similarity"]),
        html.Tbody(table_rows),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "fontFamily": FONT_BODY, "fontSize": "13px", "backgroundColor": SURFACE})

    return header_text, fig, table

# -----------------------------------------------------------------------------
# 16. Player COG scatter callback
# -----------------------------------------------------------------------------
@app.callback(
    Output("cog_scatter", "figure"),
    Input("cog_color",    "value"),
    Input("cog_league",   "value"),
    Input("cog_team",     "value"),
)
def update_cog_scatter(color_by, league, team):
    df = players_filtered.copy()
    if league != "ALL": df = df[df["league"] == league]
    if team   != "ALL": df = df[df["team"]   == team]
    df = df.dropna(subset=["Touch COG", "Tackle COG"])
    fig = go.Figure()
    if color_by == "position":
        pos_numeric = df["new_position"].map({"Defender": 0, "Wingback": 1, "Midfielder": 2, "Forward": 3})
        fig.add_trace(go.Scatter(
            x=df["Touch COG"], y=df["Tackle COG"], mode="markers",
            marker=dict(color=pos_numeric, **MARKER_STYLE),
            text=df["player"],
            customdata=np.stack([df["team"], df["new_position"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>Team: %{customdata[0]}<br>"
                "Position: %{customdata[1]}<br>"
                "Touch COG: %{x:.2f}<br>Tackle COG: %{y:.2f}<extra></extra>"
            ),
        ))
    else:
        palette = px.colors.qualitative.Plotly
        for i, t in enumerate(df["team"].unique()):
            mask = df["team"] == t
            fig.add_trace(go.Scatter(
                x=df.loc[mask, "Touch COG"], y=df.loc[mask, "Tackle COG"],
                mode="markers", name=t,
                marker=dict(size=8, opacity=0.8, color=palette[i % len(palette)],
                            line=dict(width=0.5, color=BG)),
                text=df.loc[mask, "player"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Touch COG: %{x:.2f}<br>Tackle COG: %{y:.2f}<extra></extra>"
                ),
            ))
    fig.update_layout(
        **FIG_BASE,
        xaxis=dict(title="Touch Center of Gravity", **COG_AXIS),
        yaxis=dict(title="Tackle Center of Gravity", **COG_AXIS,
                   scaleanchor="x", scaleratio=1),
    )
    fig.add_shape(type="line", x0=-1, x1=1, y0=0, y1=0,
                  line=dict(color=BORDER, width=1, dash="dot"))
    fig.add_shape(type="line", x0=0, x1=0, y0=-1, y1=1,
                  line=dict(color=BORDER, width=1, dash="dot"))
    return fig

# -----------------------------------------------------------------------------
# 17. Standard Team Leaderboard callback
# -----------------------------------------------------------------------------
@app.callback(
    Output("tlb_summary", "children"),
    Output("tlb_table",   "children"),
    Input("tlb_league",   "value"),
    Input("tlb_sort",     "value"),
    Input("tlb_dir",      "value"),
    Input("tlb_n",        "value"),
)
def update_team_leaderboard(league, sort_col, direction, n):
    sub = team_df.copy()
    if league != "ALL": sub = sub[sub["league"] == league]
    if sort_col not in sub.columns:
        return "Column not found.", html.P("No data.", style={"color": MUTED})
    sub = sub.sort_values(sort_col, ascending=(direction == "asc")).head(n)
    summary = f"{len(sub)} teams" + (f" -- {league}" if league != "ALL" else "")

    display_map = {
        "team": "Team", "league": "League", "goals": "Goals", "xg": "xG",
        "minutes_mad": "Min MAD", "team_touch_cog": "Touch COG",
        "team_tackle_cog": "Tackle COG", "attacking_compactness": "Compactness",
        "mistakes": "Mistakes", "yellow_cards": "Yellow Cards", "fouls_committed": "Fouls",
    }
    show_cols = ["team", "league", "goals", "xg", "minutes_mad", "team_touch_cog",
                 "team_tackle_cog", "attacking_compactness", "mistakes",
                 "yellow_cards", "fouls_committed"]
    show_cols = [c for c in show_cols if c in sub.columns]
    hdrs      = [display_map.get(c, c) for c in show_cols]
    center_h  = [h for h in hdrs if h not in ["Team", "League"]]

    base_td = {"fontFamily": FONT_BODY, "padding": "10px 14px",
               "borderBottom": f"1px solid {SURFACE2}", "fontSize": "13px",
               "color": TEXT, "textAlign": "left"}
    mid_td  = {**base_td, "color": MUTED, "textAlign": "center"}

    rows = []
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        cells = [html.Td(rank, style={**mid_td, "color": GOLD,
                                      "fontFamily": FONT_DATA, "fontWeight": "700"})]
        for c, h in zip(show_cols, hdrs):
            val = row.get(c, "")
            if isinstance(val, float):
                val = f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}"
            elif isinstance(val, (int, np.integer)):
                val = str(int(val))
            style = base_td if h == "Team" else ({**base_td, "color": "#cdd9e5"} if h == "League" else mid_td)
            if c == sort_col:
                style = {**style, "backgroundColor": f"{GOLD}12", "color": GOLD_L}
            cells.append(html.Td(str(val), style=style))
        rows.append(html.Tr(cells))

    table = html.Table(
        [_table_header(["#"] + hdrs, center_cols=center_h), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": SURFACE,
               "overflowX": "auto", "display": "block"},
    )
    return summary, table

# -----------------------------------------------------------------------------
# 18. Advanced Team callbacks
# -----------------------------------------------------------------------------
@app.callback(Output("tc_scatter", "figure"), Input("tc_league", "value"))
def update_tc_scatter(league):
    sub = team_df.dropna(subset=["team_touch_cog", "team_tackle_cog", "xg"]).copy()
    if not league:
        return empty_fig()
    sub = sub[sub["league"] == league].copy()
    if sub.empty:
        return empty_fig("No data for this league")
    sub["touch_z"]  = (sub["team_touch_cog"]  - sub["team_touch_cog"].mean())  / sub["team_touch_cog"].std()
    sub["tackle_z"] = (sub["team_tackle_cog"] - sub["team_tackle_cog"].mean()) / sub["team_tackle_cog"].std()
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=BORDER, width=1, dash="dot"))
    for qx, qy, label in [
        ( 1.2,  1.2, "High Touch / High Tackle"),
        (-1.2,  1.2, "Low Touch / High Tackle"),
        ( 1.2, -1.2, "High Touch / Low Tackle"),
        (-1.2, -1.2, "Low Touch / Low Tackle"),
    ]:
        fig.add_annotation(x=qx, y=qy, text=label, showarrow=False, align="center",
                           font=dict(family=FONT_DATA, size=9, color=BORDER))
    fig.add_trace(go.Scatter(
        x=sub["touch_z"], y=sub["tackle_z"],
        mode="markers+text", text=sub["team"], textposition="top center",
        textfont=dict(family=FONT_BODY, size=10, color=MUTED),
        marker=dict(size=13, color=sub["xg"], colorscale="YlOrBr", showscale=True,
                    line=dict(width=0.5, color=BG),
                    colorbar=dict(title=dict(text="xG", font=dict(color=MUTED, size=11)),
                                  tickfont=dict(color=MUTED, size=10),
                                  bgcolor=SURFACE, bordercolor=BORDER)),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Touch COG z: %{x:.2f}<br>Tackle COG z: %{y:.2f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **FIG_BASE,
        xaxis=dict(title="Touch COG (z-score, league-relative)", **AXIS_BASE),
        yaxis=dict(title="Tackle COG (z-score, league-relative)", **AXIS_BASE),
    )
    return fig


@app.callback(
    Output("mv_title",   "children"),
    Output("mv_scatter", "figure"),
    Input("mv_x",        "value"),
    Input("mv_y",        "value"),
    Input("mv_league",   "value"),
)
def update_mv_scatter(x_label, y_label, league):
    x_col = X_OPTIONS.get(x_label)
    y_col = Y_OPTIONS.get(y_label)
    if not (x_col and y_col):
        return "", empty_fig()
    title = f"Effect of {x_label} on {y_label}"
    sub   = team_df.dropna(subset=[x_col, y_col]).copy()
    if league != "ALL": sub = sub[sub["league"] == league]
    palette = px.colors.qualitative.Plotly
    fig     = go.Figure()
    for i, lg in enumerate(sub["league"].unique()):
        lg_sub = sub[sub["league"] == lg]
        color  = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=lg_sub[x_col], y=lg_sub[y_col],
            mode="markers+text", name=lg,
            text=lg_sub["team"], textposition="top center",
            textfont=dict(family=FONT_BODY, size=9, color=MUTED),
            marker=dict(size=10, color=color, line=dict(width=0.5, color=BG)),
            hovertemplate=(
                f"<b>%{{text}}</b><br>{x_label}: %{{x:.3f}}<br>"
                f"{y_label}: %{{y:.2f}}<extra></extra>"
            ),
        ))
    if len(sub) >= 2:
        z  = np.polyfit(sub[x_col].astype(float), sub[y_col].astype(float), 1)
        xr = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=xr, y=np.poly1d(z)(xr), mode="lines", name="Trend", showlegend=True,
            line=dict(color=_hex_rgba(GOLD, 0.67), width=2, dash="dot"),
            hoverinfo="skip",
        ))
    fig.update_layout(**FIG_BASE,
                      xaxis=dict(title=x_label, **AXIS_BASE),
                      yaxis=dict(title=y_label, **AXIS_BASE))
    return title, fig


@app.callback(
    Output("pitch_map",   "figure"),
    Input("pitch_team_a", "value"),
    Input("pitch_team_b", "value"),
)
def update_pitch_map(team_a, team_b):
    pitch_df = team_df.dropna(subset=list(COG_COL_MAP.values())).copy()
    if not team_a or not team_b:
        return empty_fig("Select two teams above")
    row_a = pitch_df[pitch_df["team"] == team_a]
    row_b = pitch_df[pitch_df["team"] == team_b]
    if row_a.empty or row_b.empty:
        return empty_fig("Team data not available")
    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    W        = PITCH_WIDTH
    gap      = 24
    offset_b = W + gap
    pos_colors_pitch = {
        "Defender": "#f85149", "Wingback": "#d29922",
        "Midfielder": "#3fb950", "Forward": "#2f81f7",
    }

    def pitch_shapes(x_offset):
        cx = x_offset + CENTER_X
        L  = PITCH_LENGTH
        def s(t, **kw):
            return dict(type=t, layer="below", **kw)
        return [
            s("rect", x0=x_offset, y0=0, x1=x_offset+W, y1=L,
              fillcolor="#1a3d1a", line=dict(width=0)),
            # Pitch lines
            s("rect", x0=x_offset, y0=0, x1=x_offset+W, y1=L,
              line=dict(color="rgba(255,255,255,0.6)", width=1.5), fillcolor="rgba(0,0,0,0)"),
            s("line", x0=x_offset, y0=L/2, x1=x_offset+W, y1=L/2,
              line=dict(color="rgba(255,255,255,0.6)", width=1.5)),
            s("circle", x0=cx-10, y0=L/2-10, x1=cx+10, y1=L/2+10,
              line=dict(color="rgba(255,255,255,0.5)", width=1.5), fillcolor="rgba(0,0,0,0)"),
            s("circle", x0=cx-0.8, y0=L/2-0.8, x1=cx+0.8, y1=L/2+0.8,
              fillcolor="rgba(255,255,255,0.6)", line=dict(width=0)),
            s("rect", x0=x_offset+18, y0=0, x1=x_offset+W-18, y1=18,
              line=dict(color="rgba(255,255,255,0.5)", width=1.5), fillcolor="rgba(0,0,0,0)"),
            s("rect", x0=x_offset+18, y0=L-18, x1=x_offset+W-18, y1=L,
              line=dict(color="rgba(255,255,255,0.5)", width=1.5), fillcolor="rgba(0,0,0,0)"),
            s("rect", x0=x_offset+30, y0=0, x1=x_offset+W-30, y1=6,
              line=dict(color="rgba(255,255,255,0.4)", width=1), fillcolor="rgba(0,0,0,0)"),
            s("rect", x0=x_offset+30, y0=L-6, x1=x_offset+W-30, y1=L,
              line=dict(color="rgba(255,255,255,0.4)", width=1), fillcolor="rgba(0,0,0,0)"),
        ]

    def pos_traces(row, x_offset, show_legend):
        traces = []
        added  = set()
        positions = [
            ("Defender",   [x_offset + CENTER_X - SPACING/2, x_offset + CENTER_X + SPACING/2],
             [cog_to_y(row["cog_defender"])] * 2),
            ("Wingback",   [x_offset + 10, x_offset + W - 10],
             [cog_to_y(row["cog_wingback"])] * 2),
            ("Midfielder", [x_offset + CENTER_X - SPACING, x_offset + CENTER_X, x_offset + CENTER_X + SPACING],
             [cog_to_y(row["cog_midfielder"])] * 3),
            ("Forward",    [x_offset + CENTER_X - SPACING, x_offset + CENTER_X, x_offset + CENTER_X + SPACING],
             [cog_to_y(row["cog_forward"])] * 3),
        ]
        for group, xs, ys in positions:
            color = pos_colors_pitch[group]
            show  = show_legend and group not in added
            added.add(group)
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                name=group,
                legendgroup=group,
                showlegend=show,
                marker=dict(size=20, color=color,
                            line=dict(width=2, color="white"), opacity=0.92),
                hovertemplate=(
                    f"<b>{group}</b><br>"
                    f"COG: {row[f'cog_{group.lower()}']:.3f}<br>"
                    f"Team: {row['team']}<extra></extra>"
                ),
            ))
        return traces

    all_shapes = pitch_shapes(0) + pitch_shapes(offset_b)
    all_traces = pos_traces(row_a, 0, True) + pos_traces(row_b, offset_b, False)

    # COG value labels
    cog_annotations = []
    for pos, col in COG_COL_MAP.items():
        if col not in row_a.index or col not in row_b.index:
            continue
        color = pos_colors_pitch[pos]
        cog_annotations += [
            dict(x=W/2,             y=cog_to_y(row_a[col]) - 7,
                 text=f"{row_a[col]:.3f}", showarrow=False,
                 font=dict(family=FONT_DATA, color=color, size=9), xanchor="center"),
            dict(x=offset_b + W/2,  y=cog_to_y(row_b[col]) - 7,
                 text=f"{row_b[col]:.3f}", showarrow=False,
                 font=dict(family=FONT_DATA, color=color, size=9), xanchor="center"),
        ]

    fig = go.Figure(data=all_traces)
    _base = {k: v for k, v in FIG_BASE.items() if k != "legend"}
    fig.update_layout(
        shapes=all_shapes,
        **_base,
        height=700,
        xaxis=dict(range=[-4, offset_b + W + 4], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(range=[-10, PITCH_LENGTH + 12], showgrid=False,
                   zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        annotations=[
            dict(x=W/2,            y=PITCH_LENGTH + 6, text=f"<b>{team_a}</b>",
                 showarrow=False, font=dict(family=FONT_HEAD, color=TEXT, size=15),
                 xanchor="center"),
            dict(x=offset_b + W/2, y=PITCH_LENGTH + 6, text=f"<b>{team_b}</b>",
                 showarrow=False, font=dict(family=FONT_HEAD, color=TEXT, size=15),
                 xanchor="center"),
        ] + cog_annotations,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.02,
            font=dict(family=FONT_BODY, size=11, color=TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig

# -----------------------------------------------------------------------------
# 19. Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
