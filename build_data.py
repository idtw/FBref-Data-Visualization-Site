# =============================================================================
# build_data.py -- FBref Memoriam: One-Time Data Pipeline
# =============================================================================
# Run this ONCE locally from your project root:
#     python build_data.py
#
# Outputs:
#     data/players_filtered.parquet  -- 1,500+ outfield players, fully featured
#     data/scores_df.parquet         -- subset of players_filtered with scores
#     data/team_df.parquet           -- 96 team-season rows with COG + metrics
#
# After running, commit the data/ folder to GitHub.
# The web app (app.py) reads these files directly and never scrapes FBref.
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

from functools import reduce

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import soccerdata as sd

os.makedirs("data", exist_ok=True)

# -----------------------------------------------------------------------------
# Constants (mirrors notebook 1b)
# -----------------------------------------------------------------------------
position_map = {"Defender": 0, "Wingback": 1, "Midfielder": 2, "Forward": 3}

feature_sets = {
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

attack_features = {
    "Non-Penalty Expected Goals per 90":        0.25,
    "Expected Assisted Goals per 90":           0.20,
    "Goal-Creating Actions per 90":             0.20,
    "Non-Penalty xG Overperformance":           0.15,
    "Shots on Target %":                        0.10,
    "Passes into Penalty Area per 100 Touches": 0.10,
}
progression_features = {
    "Progressive Passes per 100 Touches":            0.25,
    "Progressive Carries per 100 Touches":           0.25,
    "Take-On Success %":                             0.15,
    "Take-Ons Attempted per 90":                     0.15,
    "Progressive Passing Distance per 100 Touches":  0.10,
    "Progressive Carrying Distance per 100 Touches": 0.10,
}
defense_features = {
    "Tackles per 90":            0.20,
    "Interceptions per 90":      0.20,
    "Ball Recoveries per 90":    0.20,
    "Dribblers Tackled %":       0.15,
    "Aerial Win %":              0.15,
    "Turnovers per 100 Touches": 0.10,
}
invert_stats = {"Turnovers per 100 Touches"}
composite_weights = {
    "Forward":    {"attack": 0.55, "progression": 0.30, "defense": 0.15},
    "Midfielder": {"attack": 0.30, "progression": 0.40, "defense": 0.30},
    "Wingback":   {"attack": 0.25, "progression": 0.40, "defense": 0.35},
    "Defender":   {"attack": 0.15, "progression": 0.25, "defense": 0.60},
}

cog_col_map = {
    "Defender":   "cog_defender",
    "Wingback":   "cog_wingback",
    "Midfielder": "cog_midfielder",
    "Forward":    "cog_forward",
}

rotation_metric = "minutes_mad"

# -----------------------------------------------------------------------------
# Chapter 1 -- Pull from FBref
# -----------------------------------------------------------------------------
print("Connecting to FBref via soccerdata...")

seasons    = ["2025-2026"]
leagues    = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
stat_types = [
    "standard", "shooting", "passing", "passing_types",
    "goal_shot_creation", "defense", "possession", "misc",
]

fb  = sd.FBref(leagues=leagues, seasons=seasons)
dfs = []
for st in stat_types:
    print(f"  Loading {st}...", flush=True)
    dfs.append(fb.read_player_season_stats(stat_type=st))
    print(f"  {st} loaded: {len(dfs[-1])} rows", flush=True)

standard, shooting, passing, passing_types, \
    goal_shot_creation, defense, possession, misc = dfs

# -----------------------------------------------------------------------------
# Chapter 2a -- Flatten, deduplicate, merge
# -----------------------------------------------------------------------------
print("\nFlattening and merging stat tables...")

def flatten_columns(df):
    df2 = df.copy()
    df2.columns = [
        "_".join(str(c) for c in col if c) if isinstance(col, tuple) else str(col)
        for col in df2.columns
    ]
    return df2

df_list = [
    ("standard",           flatten_columns(standard)),
    ("shooting",           flatten_columns(shooting)),
    ("passing",            flatten_columns(passing)),
    ("passing_types",      flatten_columns(passing_types)),
    ("goal_shot_creation", flatten_columns(goal_shot_creation)),
    ("defense",            flatten_columns(defense)),
    ("possession",         flatten_columns(possession)),
    ("misc",               flatten_columns(misc)),
]

key_cols    = ["league", "season", "team", "player"]
seen_nonkey = set()
cleaned     = []
for name, df in df_list:
    df2     = df.copy()
    to_drop = [c for c in df2.columns if c not in key_cols and c in seen_nonkey]
    seen_nonkey.update(c for c in df2.columns if c not in key_cols)
    df2 = df2.drop(columns=to_drop, errors="ignore")
    print(f"  {name}: dropped {len(to_drop)} duplicate cols -> shape {df2.shape}")
    cleaned.append(df2)

players_all = reduce(
    lambda left, right: left.merge(right, on=key_cols, how="inner"),
    cleaned,
)
print(f"Final merged shape: {players_all.shape}")

# Promote index if needed
if set(key_cols).issubset(set(players_all.index.names)):
    players_all = players_all.reset_index(drop=False)

# -----------------------------------------------------------------------------
# Chapter 2a -- Rename columns
# -----------------------------------------------------------------------------
rename_map = {
    "nation": "Nationality", "pos": "Position", "age": "Age", "born": "Birth Year",
    "Playing Time_MP": "Matches Played", "Playing Time_Starts": "Matches Started",
    "Playing Time_Min": "Minutes Played", "Playing Time_90s": "90s Played",
    "Performance_Gls": "Goals", "Performance_Ast": "Assists",
    "Performance_G+A": "Goals + Assists", "Performance_G-PK": "Non-Penalty Goals",
    "Performance_PK": "Penalties Scored", "Performance_PKatt": "Penalties Attempted",
    "Performance_CrdY": "Yellow Cards", "Performance_CrdR": "Red Cards",
    "Expected_xG": "Expected Goals", "Expected_npxG": "Non-Penalty Expected Goals",
    "Expected_xAG": "Expected Assisted Goals",
    "Expected_npxG+xAG": "Non-Penalty Expected Goals + Expected Assisted Goals",
    "Progression_PrgC": "Progressive Carries", "Progression_PrgP": "Progressive Passes",
    "Progression_PrgR": "Progressive Passes Received",
    "Standard_Gls": "Goals", "Standard_Sh": "Shots", "Standard_SoT": "Shots on Target",
    "Standard_SoT%": "Shots on Target %", "Standard_Sh/90": "Shots p90",
    "Standard_SoT/90": "Shots on Target p90", "Standard_G/Sh": "Goals per Shot",
    "Standard_G/SoT": "Goals per Shot on Target", "Standard_Dist": "Average Shot Distance",
    "Standard_FK": "Free Kicks Scored", "Standard_PK": "Penalties Scored",
    "Standard_PKatt": "Penalties Attempted",
    "Expected_npxG/Sh": "Expected Non-Penalty Goals per Shot",
    "Expected_G-xG": "xG Overperformance", "Expected_np:G-xG": "Non-Penalty xG Overperformance",
    "Total_Cmp": "Total Passes Completed", "Total_Att": "Total Passes Attempted",
    "Total_Cmp%": "Total Pass Completion %", "Total_TotDist": "Total Passing Distance",
    "Total_PrgDist": "Progressive Passing Distance",
    "Short_Cmp": "Short Passes Completed", "Short_Att": "Short Passes Attempted",
    "Short_Cmp%": "Short Pass Completion %", "Medium_Cmp": "Medium Passes Completed",
    "Medium_Att": "Medium Passes Attempted", "Medium_Cmp%": "Medium Pass Completion %",
    "Long_Cmp": "Long Passes Completed", "Long_Att": "Long Passes Attempted",
    "Long_Cmp%": "Long Pass Completion %", "Ast": "Assists", "xAG": "Expected Assisted Goals",
    "Expected_xA": "Expected Assists", "Expected_A-xAG": "xAG Overperformance",
    "KP": "Key Passes", "1/3": "Passes into Final Third",
    "PPA": "Passes into Penalty Area", "CrsPA": "Crosses into Penalty Area",
    "PrgP": "Progressive Passes", "Att": "Total Passes Attempted",
    "Pass Types_Live": "Live Ball Passes", "Pass Types_Dead": "Dead Ball Passes",
    "Pass Types_FK": "Free Kick Passes", "Pass Types_TB": "Through Balls",
    "Pass Types_Sw": "Switches", "Pass Types_Crs": "Crosses",
    "Pass Types_TI": "Throw-Ins", "Pass Types_CK": "Corner Kicks",
    "Corner Kicks_In": "Inswinging Corner Kicks", "Corner Kicks_Out": "Outswinging Corner Kicks",
    "Corner Kicks_Str": "Straight Corner Kicks",
    "Outcomes_Cmp": "Passes Completed", "Outcomes_Off": "Passes Offsides",
    "Outcomes_Blocks": "Passes Blocked",
    "SCA_SCA": "Shot-Creating Actions", "SCA_SCA90": "Shot-Creating Actions p90",
    "SCA Types_PassLive": "Live Ball Pass SCAs", "SCA Types_PassDead": "Dead Ball Pass SCAs",
    "SCA Types_TO": "Take-On SCAs", "SCA Types_Sh": "Shot SCAs",
    "SCA Types_Fld": "Drawn Foul SCAs", "SCA Types_Def": "Defensive SCAs",
    "GCA_GCA": "Goal-Creating Actions", "GCA_GCA90": "Goal-Creating Actions p90",
    "GCA Types_PassLive": "Live Ball Pass GCAs", "GCA Types_PassDead": "Dead Ball Pass GCAs",
    "GCA Types_TO": "Take-On GCAs", "GCA Types_Sh": "Shot GCAs",
    "GCA Types_Fld": "Drawn Foul GCAs", "GCA Types_Def": "Defensive GCAs",
    "Tackles_Tkl": "Tackles", "Tackles_TklW": "Possession-Winning Tackles",
    "Tackles_Def 3rd": "Defensive 3rd Tackles", "Tackles_Mid 3rd": "Midfield 3rd Tackles",
    "Tackles_Att 3rd": "Attacking 3rd Tackles",
    "Challenges_Tkl": "Dribbles Tackled", "Challenges_Att": "Attempted Dribbles Tackled",
    "Challenges_Tkl%": "Dribblers Tackled %", "Challenges_Lost": "Challenges Lost",
    "Blocks_Blocks": "Blocks", "Blocks_Sh": "Blocked Shots", "Blocks_Pass": "Blocked Passes",
    "Int": "Interceptions", "Tkl+Int": "Tackles + Interceptions",
    "Clr": "Clearances", "Err": "Errors",
    "Touches_Touches": "Touches", "Touches_Def Pen": "Touches in Defensive Penalty Box",
    "Touches_Def 3rd": "Touches in Defensive 3rd", "Touches_Mid 3rd": "Touches in Midfield 3rd",
    "Touches_Att 3rd": "Touches in Attacking 3rd",
    "Touches_Att Pen": "Touches in Attacking Penalty Box",
    "Touches_Live": "Live Ball Touches",
    "Take-Ons_Att": "Take-Ons Attempted", "Take-Ons_Succ": "Successful Take-Ons",
    "Take-Ons_Succ%": "Take-On Success %", "Take-Ons_Tkld": "Times Tackled During Take-On",
    "Take-Ons_Tkld%": "Tackled During Take-On %",
    "Carries_Carries": "Carries", "Carries_TotDist": "Total Carrying Distance",
    "Carries_PrgDist": "Progressive Carrying Distance", "Carries_PrgC": "Progressive Carries",
    "Carries_1/3": "Carries into Final Third", "Carries_CPA": "Carries into Penalty Area",
    "Carries_Mis": "Miscontrols", "Carries_Dis": "Dispossessed",
    "Receiving_Rec": "Passes Received", "Receiving_PrgR": "Progressive Passes Received",
    "Performance_2CrdY": "Second Yellow Cards", "Performance_Fls": "Fouls Committed",
    "Performance_Fld": "Fouls Won", "Performance_Off": "Offsides",
    "Performance_Crs": "Crosses", "Performance_Int": "Interceptions",
    "Performance_TklW": "Tackles Won", "Performance_PKwon": "Penalties Won",
    "Performance_PKcon": "Penalties Conceded", "Performance_OG": "Own Goals",
    "Performance_Recov": "Ball Recoveries",
    "Aerial Duels_Won": "Aerial Duels Won", "Aerial Duels_Lost": "Aerial Duels Lost",
    "Aerial Duels_Won%": "Aerial Win %",
}

players_all = players_all.rename(columns=rename_map)
players_all = players_all.loc[:, ~players_all.columns.duplicated(keep="first")]

# Drop pre-existing p90 cols -- we generate our own
p90_cols_to_drop = [col for col in players_all.columns if "p90" in col]
players_all = players_all.drop(columns=p90_cols_to_drop)

# Clean numeric ID index
players_all["ID"] = range(1, len(players_all) + 1)
players_all = players_all.set_index("ID")

# Generate per-90 columns for all eligible features
exclude_columns = [
    "league", "season", "team", "player", "Nationality", "Position",
    "Age", "Birth Year", "Matches Played", "Matches Started",
    "Minutes Played", "90s Played",
]
exclude_patterns = ["_total", "COG", "per 100", "%", "p90", "Overperformance", "Average"]

per90_feature_list = []
for col in players_all.columns:
    if col in exclude_columns:
        continue
    if any(pattern in col for pattern in exclude_patterns):
        continue
    if not pd.api.types.is_numeric_dtype(players_all[col]):
        continue
    per90_feature_list.append(col)

for col in per90_feature_list:
    players_all[col + " per 90"] = players_all[col] / players_all["90s Played"]

# -----------------------------------------------------------------------------
# Chapter 2a -- Calculated fields
# -----------------------------------------------------------------------------
print("Building calculated fields...")

players_all["Possession-Winning Tackle %"] = (
    players_all["Possession-Winning Tackles"] / players_all["Tackles"]
)
players_all["Non-Penalty Goals per Shot"] = (
    players_all["Non-Penalty Goals"] / players_all["Shots"]
)
players_all["Total Progression Distance"] = (
    players_all["Progressive Passing Distance"] + players_all["Progressive Carrying Distance"]
)
players_all["Miscontrols per 100 Touches"] = (
    players_all["Miscontrols"] / players_all["Touches"] * 100
)
players_all["Turnovers per 100 Touches"] = (
    (players_all["Miscontrols"] + players_all["Dispossessed"] + players_all["Errors"])
    / players_all["Touches"] * 100
)
players_all["Switches per 100 Touches"] = (
    players_all["Switches"] / players_all["Touches"] * 100
)
players_all["Progressive Passes per 100 Touches"] = (
    players_all["Progressive Passes"] / players_all["Touches"] * 100
)
players_all["Progressive Carries per 100 Touches"] = (
    players_all["Progressive Carries"] / players_all["Touches"] * 100
)
players_all["Progressive Passing Distance per 100 Touches"] = (
    players_all["Progressive Passing Distance"] / players_all["Touches"] * 100
)
players_all["Progressive Carrying Distance per 100 Touches"] = (
    players_all["Progressive Carrying Distance"] / players_all["Touches"] * 100
)
players_all["Passes into Penalty Area per 100 Touches"] = (
    players_all["Passes into Penalty Area"] / players_all["Touches"] * 100
)
players_all["Progressive Receptions per 100 Touches"] = (
    players_all["Progressive Passes Received"] / players_all["Touches"] * 100
)
players_all["Blocks + Clearances"] = (
    players_all["Blocks"] + players_all["Clearances"]
)

# -----------------------------------------------------------------------------
# Chapter 2b -- Touch COG and Tackle COG
# -----------------------------------------------------------------------------
print("Computing Touch COG and Tackle COG...")

players_all["touches_total"] = (
    players_all["Touches in Defensive Penalty Box"] +
    players_all["Touches in Defensive 3rd"] +
    players_all["Touches in Midfield 3rd"] +
    players_all["Touches in Attacking 3rd"] +
    players_all["Touches in Attacking Penalty Box"]
)
players_all["Touch COG"] = (
    players_all["Touches in Defensive Penalty Box"] * -1.0 +
    players_all["Touches in Defensive 3rd"]         * -0.6 +
    players_all["Touches in Midfield 3rd"]           *  0.0 +
    players_all["Touches in Attacking 3rd"]          *  0.6 +
    players_all["Touches in Attacking Penalty Box"]  *  1.0
) / players_all["touches_total"]

players_all["tackles_total"] = (
    players_all["Defensive 3rd Tackles"] +
    players_all["Midfield 3rd Tackles"] +
    players_all["Attacking 3rd Tackles"]
).astype(float)
players_all["Tackle COG"] = (
    players_all["Defensive 3rd Tackles"] * -1.0 +
    players_all["Midfield 3rd Tackles"]   *  0.0 +
    players_all["Attacking 3rd Tackles"]  *  1.0
).astype(float) / players_all["tackles_total"]

# -----------------------------------------------------------------------------
# Chapter 2 -- players_filtered
# -----------------------------------------------------------------------------
print("Filtering to outfield players with 270+ minutes...")

players_filtered = (
    players_all
    .fillna(0)
    .loc[
        (players_all["Minutes Played"] >= 270) &
        (players_all["Position"] != "GK") &
        (players_all["Birth Year"].notna())
    ]
    .copy()
)
print(f"players_filtered: {players_filtered.shape[0]} players")

# -----------------------------------------------------------------------------
# Chapter 2c -- K-Means positional clustering
# -----------------------------------------------------------------------------
print("Running K-Means position clustering...")

players_filtered["Positional COG"] = (
    players_filtered.groupby("team")["Touch COG"]
    .transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5)
)

def assign_roles(group):
    X          = group[["Positional COG"]].values
    km         = KMeans(n_clusters=3, random_state=42, n_init="auto")
    cluster_ids = km.fit_predict(X)
    centers    = km.cluster_centers_.flatten()
    order      = np.argsort(centers)
    role_map   = {order[0]: "Defender", order[1]: "Midfielder", order[2]: "Forward"}
    return pd.Series([role_map[c] for c in cluster_ids], index=group.index, name="kmeans_position")

players_filtered["kmeans_position"] = (
    players_filtered.groupby("team", group_keys=False).apply(assign_roles)
)

# Wingback detection via throw-in frequency
print("Detecting Wingbacks via throw-in frequency...")

# Build a working copy with kmeans_position for groupby
df = players_filtered.copy()
width_labels = []

for team, group in df.groupby("team"):
    ti90  = group["Throw-Ins per 90"]
    width = pd.Series("", index=group.index, dtype=object)

    zero_mask = ti90 == 0
    width.loc[zero_mask] = "Central"

    nonzero_mask = ~zero_mask
    group_nz     = group[nonzero_mask]

    if len(group_nz) >= 2:
        km_w      = KMeans(n_clusters=2, random_state=42, n_init="auto")
        cl_ids    = km_w.fit_predict(group_nz[["Throw-Ins per 90"]].values)
        order_w   = np.argsort(km_w.cluster_centers_.flatten())
        width_map = {order_w[0]: "", order_w[1]: "Wide"}
        width.loc[group_nz.index] = [width_map[c] for c in cl_ids]

    width_labels.append(width.to_frame(name="kmeans_width"))

players_filtered["kmeans_width"] = pd.concat(width_labels)["kmeans_width"]

# Derive final new_position (including Wingback)
def derive_new_position(row):
    pos     = row["Position"]
    knn_pos = row["kmeans_position"]
    width   = row["kmeans_width"]

    if pos == "GK":
        new_pos = "Goalkeeper"
    elif pos == "DF":
        new_pos = "Defender"
    elif pos == "MF":
        new_pos = "Midfielder"
    elif pos == "FW":
        new_pos = "Forward"
    elif isinstance(pos, str) and len(pos) > 3:
        codes   = [p.strip() for p in pos.replace(",", " ").split()]
        new_pos = "Forward" if "FW" in codes else knn_pos
    else:
        new_pos = knn_pos

    if knn_pos == "Midfielder" and width == "Wide":
        new_pos = "Wingback"

    return new_pos

players_filtered["new_position"] = players_filtered.apply(derive_new_position, axis=1)
print(players_filtered["new_position"].value_counts().to_string())

# -----------------------------------------------------------------------------
# Chapter 2g -- Scores
# -----------------------------------------------------------------------------
print("\nComputing player scores...")

all_score_features = (
    list(attack_features) + list(progression_features) + list(defense_features)
)
available_features = [f for f in all_score_features if f in players_filtered.columns]

scores_df = (
    players_filtered
    .dropna(subset=available_features)
    [["player", "team", "league", "season", "new_position", "Minutes Played", "Age"]
     + available_features]
    .copy()
    .reset_index(drop=True)
)

def compute_category_score(df, feature_weights, invert=set()):
    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)
    for feat, weight in feature_weights.items():
        if feat not in df.columns:
            continue
        ascending   = feat not in invert
        percentile  = df.groupby("new_position")[feat].rank(pct=True, ascending=ascending)
        weighted_sum += percentile * weight
        total_weight += weight
    return ((weighted_sum / total_weight) * 100).round(1)

scores_df["attack_score"]      = compute_category_score(scores_df, attack_features,      invert_stats)
scores_df["progression_score"] = compute_category_score(scores_df, progression_features, invert_stats)
scores_df["defense_score"]     = compute_category_score(scores_df, defense_features,     invert_stats)

scores_df["Age"] = scores_df["Age"].astype(str).str.split("-").str[0]

def composite_score(row):
    w = composite_weights.get(row["new_position"], {"attack": 0.33, "progression": 0.34, "defense": 0.33})
    return round(
        row["attack_score"]      * w["attack"] +
        row["progression_score"] * w["progression"] +
        row["defense_score"]     * w["defense"], 1,
    )

scores_df["composite_score"] = scores_df.apply(composite_score, axis=1)
print(f"Scores computed for {len(scores_df)} players.")

# -----------------------------------------------------------------------------
# Chapter 3a -- Team DataFrame
# -----------------------------------------------------------------------------
print("\nBuilding team_df...")

team_df = (
    players_all
    .groupby(["team", "league", "season"], as_index=False)
    .agg(
        nineties_played = ("90s Played",            "sum"),
        minutes_played  = ("Minutes Played",         "sum"),
        pens_conceded   = ("Penalties Conceded",     "sum"),
        errors          = ("Errors",                 "sum"),
        miscontrols     = ("Miscontrols",            "sum"),
        dispossessed    = ("Dispossessed",           "sum"),
        offsides        = ("Offsides",               "sum"),
        own_goals       = ("Own Goals",              "sum"),
        yellow_cards    = ("Yellow Cards",           "sum"),
        red_cards       = ("Red Cards",              "sum"),
        fouls_committed = ("Fouls Committed",        "sum"),
        goals           = ("Goals",                  "sum"),
        xg              = ("Expected Goals",         "sum"),
        npxg_overperf   = ("Non-Penalty xG Overperformance", "sum"),
        minutes_mad     = ("Minutes Played", lambda x: (x - x.mean()).abs().mean()),
    )
)

# -----------------------------------------------------------------------------
# Chapter 3b -- Team-level Touch COG and Tackle COG
# -----------------------------------------------------------------------------
print("Computing team COG metrics...")

overall_touch_cog = (
    players_filtered
    .assign(touch_cog_weighted=lambda x: x["Touch COG"] * x["Minutes Played"])
    .groupby(["team", "league", "season"], as_index=False)
    .agg(s=("touch_cog_weighted", "sum"), m=("Minutes Played", "sum"))
    .assign(team_touch_cog=lambda x: x["s"] / x["m"])
    [["team", "league", "season", "team_touch_cog"]]
)

overall_tackle_cog = (
    players_filtered
    .dropna(subset=["Tackle COG"])
    .assign(tackle_cog_weighted=lambda x: x["Tackle COG"] * x["Minutes Played"])
    .groupby(["team", "league", "season"], as_index=False)
    .agg(s=("tackle_cog_weighted", "sum"), m=("Minutes Played", "sum"))
    .assign(team_tackle_cog=lambda x: x["s"] / x["m"])
    [["team", "league", "season", "team_tackle_cog"]]
)

team_df = team_df.merge(overall_touch_cog,  on=["team", "league", "season"], how="left")
team_df = team_df.merge(overall_tackle_cog, on=["team", "league", "season"], how="left")

# Positional COG pivot
team_pos_cog_pivot = (
    players_filtered
    .assign(touch_cog_weighted=players_filtered["Touch COG"] * players_filtered["Minutes Played"])
    .groupby(["team", "league", "season", "new_position"], as_index=False)
    .agg(sw=("touch_cog_weighted", "sum"), sm=("Minutes Played", "sum"))
    .assign(avg=lambda x: x["sw"] / x["sm"])
    .pivot_table(index=["team", "league", "season"], columns="new_position", values="avg")
    .rename(columns=lambda c: f"cog_{c.lower()}")
    .reset_index()
)
team_df = team_df.merge(team_pos_cog_pivot, on=["team", "league", "season"], how="left")

# Wingback fallback
if "cog_wingback" in team_df.columns and team_df["cog_wingback"].isnull().any():
    wide_defender_cog = (
        players_filtered[
            (players_filtered["new_position"] == "Defender") &
            (players_filtered["kmeans_width"] == "Wide")
        ]
        .assign(touch_cog_weighted=lambda x: x["Touch COG"] * x["Minutes Played"])
        .groupby(["team", "league", "season"], as_index=False)
        .agg(sw=("touch_cog_weighted", "sum"), sm=("Minutes Played", "sum"))
        .assign(cog_wingback_fallback=lambda x: x["sw"] / x["sm"])
        [["team", "league", "season", "cog_wingback_fallback"]]
    )
    team_df = team_df.merge(wide_defender_cog, on=["team", "league", "season"], how="left")
    team_df["cog_wingback"] = team_df["cog_wingback"].fillna(team_df["cog_wingback_fallback"])
    team_df = team_df.drop(columns=["cog_wingback_fallback"])

# Generic fallback for any remaining null positional COG
cog_cols = [c for c in team_df.columns if c.startswith("cog_")]
pos_map  = {
    "cog_defender":   "Defender",
    "cog_midfielder": "Midfielder",
    "cog_forward":    "Forward",
    "cog_wingback":   "Wingback",
}
for col in cog_cols:
    if not team_df[col].isnull().any():
        continue
    position = pos_map.get(col)
    if position is None:
        continue
    pos_fallback = (
        players_filtered[players_filtered["new_position"] == position]
        .assign(touch_cog_weighted=lambda x: x["Touch COG"] * x["Minutes Played"])
        .groupby(["team", "league", "season"], as_index=False)
        .agg(sw=("touch_cog_weighted", "sum"), sm=("Minutes Played", "sum"))
        .assign(**{f"{col}_fallback": lambda x: x["sw"] / x["sm"]})
        [["team", "league", "season", f"{col}_fallback"]]
    )
    team_df = team_df.merge(pos_fallback, on=["team", "league", "season"], how="left")
    team_df[col] = team_df[col].fillna(team_df[f"{col}_fallback"])
    team_df = team_df.drop(columns=[f"{col}_fallback"])

# Attacking compactness and mistakes
team_df["attacking_compactness"] = team_df["cog_forward"] - team_df["cog_defender"]
team_df["mistakes"] = (
    team_df["errors"] + team_df["miscontrols"] +
    team_df["dispossessed"] + team_df["pens_conceded"]
)

print(f"team_df: {team_df.shape[0]} team-seasons")

# -----------------------------------------------------------------------------
# Export to parquet
# -----------------------------------------------------------------------------
print("\nWriting parquet files...")

players_filtered.to_parquet("data/players_filtered.parquet")
scores_df.to_parquet("data/scores_df.parquet")
team_df.to_parquet("data/team_df.parquet")

print("Done!")
print("  data/players_filtered.parquet")
print("  data/scores_df.parquet")
print("  data/team_df.parquet")
print("\nCommit the data/ folder to GitHub, then deploy app.py.")
