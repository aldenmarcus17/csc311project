"""
clean.py 
this code cleans and normalizes certain values, see report for more detail.
this code was created with the help of claude, then edited and adjusted by us.

Usage:  python3 clean.py [input.csv] [output_dir]

steps
  1. Rename columns, split by respondent id (each id answered all 3 sports,
     so splitting by row would leak a person across train/test).
  2. Fit every statistic (medians, means/stds) on TRAIN only.
  3. Build numeric features for train and test using those fitted values.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/csc311_project_training_202609.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
SEED = 67
TEST_FRAC = 0.15
VAL_FRAC = 0.15

NAMES = ["id", "ease", "strategy", "injury", "brilliance", "where", "seasons",
         "music", "food", "colour", "song", "feel", "pay", "label"]

# LIKERT SCALE
LIKERT = {"strongly disagree": 1, "disagree": 2, "neutral": 3,
          "agree": 4, "strongly agree": 5}
LIKERT_COLS = ["strategy", "injury", "brilliance"]

# MULTI-SELECT OPTIONS
WHERE = ["Indoor facility", "Outdoor public space", "School or university facility",
         "Community recreation facility", "Private sports facility",
         "At home or nearby", "Others"]
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
MUSIC = ["Hip-hop or rap", "Latin", "Folk or traditional", "Pop", "Rock",
         "Electronic or dance", "No particular type", "Classical or jazz",
         "Country", "Other"]

# KEYWORD DICTIONARIES
# food dictionary (also catch the other things that are really broad like none)
FOODS = {
    "hotdog":       r"hot\s*-?\s*dog",
    "popcorn":      r"pop\s*corn",
    "beer":         r"\bbeers?\b|\balcohol\b",
    "soda":         r"\bsoda|\bcoke\b|\bcola\b|\bpepsi\b|\bsprite\b|soft drink|\bpop\b(?!\s*corn)",
    "pizza":        r"pizza",
    "fries":        r"\bfries\b|french fr|\bfires\b",
    "chips":        r"\bchips\b|crisps",
    "nachos":       r"nacho",
    "burger":        r"burger|sliders?",
    "wings":        r"\bwings?\b",
    "chicken":      r"chicken(?!\s*wing)",
    "peanuts":      r"peanut|cracker\s*jack|\bnuts\b",
    "sunflower":    r"sunflower|\bseeds\b",
    "pretzel":      r"pretzel",
    "water":        r"\bwater\b",
    "sports_drink": r"gatorade|gaderade|powerade|sports? drink|energy drink|electrolyte",
    "poutine":      r"poutine",
    "sandwich":     r"sandwich",
    "fish":         r"\bfish\b",
    "taco":         r"\btacos?\b",
    "bbq":          r"bbq|barbecue|\bribs\b|skewer|kebab",
    "sausage":      r"sausage|bratwurst",
    "snacks":       r"snack|fast food|pub food|finger food|junk food",
    "none":         r"\bn/?a\b|\bnone\b|\bnothing\b|unsure|\bidk\b|don.?t know|no particular",
}
# colours (based on data)
COLOURS = {
    "red":    r"\bred\b|maroon|crimson",
    "blue":   r"\bblue\b",
    "green":  r"\bgreen\b|绿",
    "white":  r"\bwhite\b",
    "black":  r"\bblack\b",
    "orange": r"\borange\b",
    "yellow": r"yellow|\bgold\b",
    "brown":  r"\bbrown\b",
    "purple": r"purple",
    "grey":   r"\bgr[ae]y\b|silver",
}
# songs are too sparse, can't vectorize
# most songs picked are covered in these three categories
SONGS = {
    "song_waka":     r"waka|shakira|wavin|k.?naan|we are one|ole ola",
    "song_baseball": r"take me out|ball ?game|centerfield|center field|sweet caroline|fogerty|talkin.{0,3} baseball",
    "song_hiphop":   r"drake|kanye|tupac|travis|jay.?z|kendrick|\blil\b|eminem|sicko|all i do is win|hall of fame|see you again|hit them up",
}
# feeling dictionary
FEELS = {
    "excited":     r"excit|exicted|exited|thrill|energi[sz]|energetic|hyped|exhilarat|pumped|motivated|amazed|amazing|ecstatic|passion|impress|anticipat|fires me up",
    "bored":       r"bor(?:ed|ing)|dull|sleep|tedious|monotonous",
    "tense":       r"tense|tension|nervous|anxious|anxiety|stress|on edge|\bedge\b|suspense|scared|worried",
    "happy":       r"happy|\bfun\b|entertain|enjoy|\bjoy|\bgood\b|pleas|cheerful",
    "calm":        r"relax|calm|peace|chill|patient|content|comfortable",
    "indifferent": r"indifferent|neutral|don.?t care|\bmeh\b|apath|\bfine\b|alright|decent|complacent|not much",
    "curious":     r"interest|intrigu|curious|engag|focus|captivat|invest",
    "confused":    r"confus|clueless|don.?t understand|do not understand",
    "none":        r"don.?t know|no idea|\bn/?a\b|\bnothing\b|never watch|don.?t watch|do not watch",
}
NO_SONG = re.compile(r"^\s*(none|no|nothing|idk|no idea|n/?a|na|no particular song|not sure|unsure|"
                     r"don.?t know|i don.?t know)\b|\[a ", re.I)
NO_PAY = re.compile(r"\b(none|nothing|unwilling|not (really |too )?willing|no willing|wouldn.?t|"
                    r"would not|won.?t|no i don|do not watch|not too much)\b|need to pay me", re.I)


def normalize_lower(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def multihot(series, options, prefix):
    sets = series.fillna("").apply(lambda v: {o.strip() for o in v.split(",")})
    return pd.DataFrame({f"{prefix}_{normalize_lower(o)}": sets.apply(lambda s, o=o: int(o in s)) for o in options},
                        index=series.index)


def keyword_flags(series, patterns, prefix=""):
    text = series.fillna("").astype(str).str.lower()
    return pd.DataFrame({f"{prefix}{k}": text.str.contains(p, regex=True).astype(int)
                         for k, p in patterns.items()}, index=series.index)


def parse_pay(s):
    """number -> number; several numbers/range -> their mean; refusal text -> 0; else NaN."""
    if pd.isna(s):
        return np.nan
    t = str(s).lower().replace("\xa0", " ").replace(",", "")
    nums = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(k\b)?", t):
        v = float(m.group(1))
        nums.append(v * 1000 if m.group(2) else v)
    if nums:
        return float(np.mean(nums))
    if NO_PAY.search(t):
        return 0.0
    return np.nan


def fit_state(train):
    """Everything learned from train only."""
    pay = train["pay"].apply(parse_pay)
    ease = pd.to_numeric(train["ease"], errors="coerce").clip(0, 10)
    return {
        "ease_median": float(ease.median()),
        "pay_median": float(pay.median()),
        "pay_log_mean": None, "pay_log_std": None,   # filled below
        "ease_mean": float(ease.fillna(ease.median()).mean()),
        "ease_std": float(ease.fillna(ease.median()).std()),
        "keywords": {"food": FOODS, "colour": COLOURS, "song": SONGS, "feel": FEELS},
    }


def build(df, st):
    X = pd.DataFrame(index=df.index)
    X["id"] = df["id"]

    ease = pd.to_numeric(df["ease"], errors="coerce").clip(0, 10).fillna(st["ease_median"])
    X["ease"] = ease
    for c in LIKERT_COLS:
        X[c] = df[c].astype(str).str.strip().str.lower().map(LIKERT).fillna(3)   # missing -> neutral

    X = X.join(multihot(df["where"], WHERE, "where"))
    X = X.join(multihot(df["seasons"], SEASONS, "season"))
    X = X.join(multihot(df["music"], MUSIC, "music"))
    X = X.join(keyword_flags(df["food"], FOODS, "food_"))
    X = X.join(keyword_flags(df["colour"], COLOURS, "colour_"))

    X = X.join(keyword_flags(df["song"], SONGS))
    X["song_named"] = (~df["song"].fillna("none").astype(str).apply(lambda s: bool(NO_SONG.search(s)))).astype(int)

    pay = df["pay"].apply(parse_pay).fillna(st["pay_median"])
    X["pay_log"] = np.log1p(pay)   # log transform only (no cap, raw value not kept)

    X = X.join(keyword_flags(df["feel"], FEELS, "feel_"))

    X["label"] = df["label"]
    return X.copy()


def pay_kind(s):
    if pd.isna(s):
        return "blank (median)"
    t = str(s).lower().replace("\xa0", " ")
    if re.search(r"\d", t):
        return "numeric"
    return "refusal (0)" if NO_PAY.search(t) else "unparsed text (median)"


def coverage_report(parts, path):
    """How much of the data each rule set covers. The rules were written from TRAIN;
    the val/train+val columns only show how well they carry over (nothing is tuned on
    validation), and the 'unmatched answers' lists use TRAIN rows only. The test set
    is intentionally excluded from this report."""
    parts = dict(parts)
    parts["train+val"] = (pd.concat([r for r, _ in parts.values()]),
                          pd.concat([x for _, x in parts.values()]))
    xcols = list(parts["train+val"][1].columns)

    def pref(p):
        return [c for c in xcols if c.startswith(p)]

    groups = {"food (>=1 item)": pref("food_"), "colour (>=1)": pref("colour_"),
              "feel (>=1)": pref("feel_"),
              "song (recognised)": ["song_waka", "song_baseball", "song_hiphop"],
              "song (named any)": ["song_named"]}
    t1 = pd.DataFrame({n: {g: 100 * (X[c].sum(axis=1) > 0).mean() for g, c in groups.items()}
                       for n, (r, X) in parts.items()})
    flag_cols = pref("food_") + pref("colour_") + pref("feel_") + pref("song_")
    t2 = pd.DataFrame({n: X[flag_cols].mean() * 100 for n, (r, X) in parts.items()})
    t3 = pd.DataFrame({n: r["pay"].apply(pay_kind).value_counts(normalize=True) * 100
                       for n, (r, X) in parts.items()}).fillna(0)
    raw_cols = ["ease", "strategy", "injury", "brilliance", "where", "seasons",
                "music", "food", "colour", "song", "feel", "pay"]
    t4 = pd.DataFrame({n: r[raw_cols].isna().mean() * 100 for n, (r, X) in parts.items()})

    lines = ["# Coverage report", "",
             "Rows per split: " + ", ".join(f"{n}={len(r)}" for n, (r, X) in parts.items()), ""]

    def block(title, df):
        lines.extend([f"## {title}", "```", df.round(1).to_string(), "```", ""])

    block("% of rows with at least one keyword matched", t1)
    block("% of rows setting each feature", t2)
    block("Pay answers: how each was handled (% of rows)", t3)
    block("Blank (missing) raw answers (% of rows)", t4)
    rtr, Xtr_ = parts["train"]
    for col, p in [("food", "food_"), ("colour", "colour_"), ("feel", "feel_")]:
        miss = Xtr_[pref(p)].sum(axis=1) == 0
        top = (rtr.loc[miss, col].fillna("<blank>").astype(str).str.lower().str.strip()
               .value_counts().head(15))
        block(f"TRAIN answers with no {col} match (top 15)", top.to_frame("count"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines).encode("ascii", errors="replace").decode("ascii"))


def main():
    df = pd.read_csv(SRC)
    df.columns = NAMES

    # split by respondant id
    ids = np.array(sorted(df["id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * TEST_FRAC))
    n_val = int(round(len(ids) * VAL_FRAC))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    test = df[df["id"].isin(test_ids)].copy()
    val = df[df["id"].isin(val_ids)].copy()
    train = df[~df["id"].isin(test_ids | val_ids)].copy()

    st = fit_state(train)                      # fitted on train only
    Xtr, Xva, Xte = build(train, st), build(val, st), build(test, st)

    # standardisation stats for the neural-net path (train only)
    st["pay_log_mean"] = float(Xtr["pay_log"].mean())
    st["pay_log_std"] = float(Xtr["pay_log"].std())

    os.makedirs(OUT, exist_ok=True)
    Xtr.to_csv(os.path.join(OUT, "train_clean.csv"), index=False)
    Xva.to_csv(os.path.join(OUT, "val_clean.csv"), index=False)
    Xte.to_csv(os.path.join(OUT, "test_clean.csv"), index=False)
    with open(os.path.join(OUT, "state.json"), "w") as f:
        json.dump({k: v for k, v in st.items() if not k.startswith("_")}, f, indent=1)

    print(f"train: {Xtr.shape}  ({train['id'].nunique()} respondents)")
    print(f"val:   {Xva.shape}  ({val['id'].nunique()} respondents)")
    print(f"test:  {Xte.shape}  ({test['id'].nunique()} respondents) - do not explore this file")
    print("train class counts:", Xtr["label"].value_counts().to_dict())

    coverage_report({"train": (train, Xtr), "val": (val, Xva)},
                    os.path.join(OUT, "coverage_report.md"))


if __name__ == "__main__":
    main()