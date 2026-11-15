"""Play-call predictor: from the pre-snap look, run vs pass and (for runs) direction L/R.

XGBoost with native categoricals on a generic snaps CSV (schema in docs/playcall.md).
Needs the `[playcall]` extra (xgboost, pandas). Evaluation rules:

- Split by game, never by snap: the latest `test_games` games are test, the game
  before them validation (early stopping only), the rest train.
- Baselines on the same test snaps: "always the most common play" and a
  down x distance-bucket pass-rate table, both built from train + validation.
- Confident-snap accuracy: share of test snaps where the model is >= 70% / >= 80%
  one way, and how often those calls are right.
- Leave-one-game-out is reported too, but it lets later games predict earlier
  ones, so it is optimistic; the test games are the headline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

NUM = ["down", "dist", "yard_ln", "qtr", "prev_gain", "box"]
CAT = ["hash", "off_form", "off_str", "backfield", "qb_set", "motion_dir", "prev_type"]
FEATS = NUM + CAT
REQUIRED = ["game", "run_pass"]
MAX_ROUNDS = 400
PATIENCE = 40
PARAMS = dict(objective="binary:logistic", eval_metric=["error", "logloss"], eta=0.05, max_depth=3, min_child_weight=2,
              subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0, max_cat_to_onehot=16, tree_method="hist", seed=7, nthread=2)
CONF = (0.7, 0.8)


def load(path, labelled: bool = True) -> pd.DataFrame:
    """Snaps CSV -> frame. Missing feature columns are allowed (treated as blank); blank categoricals are their own category."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False).apply(lambda s: s.str.strip())
    missing = [c for c in (REQUIRED if labelled else []) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    for c in FEATS + ["game", "game_date", "play_no", "run_pass", "play_dir"]:
        if c not in df:
            df[c] = ""
    df["run_pass"] = df["run_pass"].str.title()
    if labelled:
        df = df[df["run_pass"].isin(["Run", "Pass"])].copy()
    for c in NUM + ["play_no"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT:
        df[c] = df[c].replace("", "blank")
    df["play_dir"] = df["play_dir"].str.upper()
    df["gid"] = (df["game_date"] + " " + df["game"]).str.strip()  # date + name: a rematch is its own game
    df["y"] = (df["run_pass"] == "Pass").astype(int)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_games: int = 2) -> dict[str, list[str]]:
    """Games in date order (file order when there are no dates): latest `test_games` test, the one before validation, rest train."""
    order = df.drop_duplicates("gid").sort_values("game_date", kind="stable")["gid"].tolist()
    if len(order) < test_games + 2:
        raise ValueError(f"need at least {test_games + 2} games (train, validation, {test_games} test), found {len(order)}")
    return {"train": order[:-test_games - 1], "val": [order[-test_games - 1]], "test": order[-test_games:]}


def matrix(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    """Feature frame with a fixed category vocabulary (unseen values become missing)."""
    X = df[FEATS].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c].where(X[c].isin(cats[c])), categories=cats[c])
    return X


def dist_bucket(d: pd.Series) -> pd.Series:
    return pd.cut(d, [0, 3, 6, 10, np.inf], labels=["1-3", "4-6", "7-10", "11+"]).astype(str).replace("nan", "?")


def scores(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, int)
    mp, hit = np.maximum(p, 1 - p), (p >= 0.5) == y
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    conf = [{"t": t, "n": int((mp >= t).sum()), "share": float((mp >= t).mean()),
             "acc": float(hit[mp >= t].mean()) if (mp >= t).any() else None} for t in CONF]
    return {"n": len(y), "acc": float(hit.mean()), "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), "conf": conf}


def tendency(fit: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Pass rate by down x distance bucket; cells shrink toward the overall rate (2 pseudo-snaps)."""
    prior = float(fit["y"].mean())
    t = fit.assign(b=dist_bucket(fit["dist"])).groupby(["down", "b"])["y"].agg(passes="sum", n="count").reset_index()
    t["p"] = (t["passes"] + 2 * prior) / (t["n"] + 2)
    return t, prior


def tendency_p(t: pd.DataFrame, prior: float, rows: pd.DataFrame) -> np.ndarray:
    key = rows.assign(b=dist_bucket(rows["dist"]))[["down", "b"]]
    return key.merge(t[["down", "b", "p"]], on=["down", "b"], how="left")["p"].fillna(prior).to_numpy()


def _dm(X, y=None):
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def _fit(dtr, dva):
    stop = xgb.callback.EarlyStopping(rounds=PATIENCE, metric_name="logloss", data_name="val", save_best=True)
    return xgb.train(PARAMS, dtr, MAX_ROUNDS, evals=[(dva, "val")], verbose_eval=False, callbacks=[stop])


def train(data, out_dir, test_games: int = 2) -> dict:
    """Train run/pass (+ run direction) on a snaps CSV, score it, save model + metrics in out_dir. Returns the metrics."""
    df = load(data)
    sp = split(df, test_games)
    cats = {c: sorted(df[c].unique()) for c in CAT}
    X = matrix(df, cats)
    part = {k: df[df["gid"].isin(v)] for k, v in sp.items()}
    model = _fit(*(_dm(X.loc[part[k].index], part[k]["y"]) for k in ("train", "val")))
    rounds = model.best_iteration + 1

    te, fit = part["test"], pd.concat([part["train"], part["val"]])
    p = model.predict(_dm(X.loc[te.index]))
    tt, prior = tendency(fit)
    imp = model.get_score(importance_type="total_gain")
    tot = sum(imp.values()) or 1
    ev = {"model": scores(p, te["y"]),
          "majority": scores(np.full(len(te), prior), te["y"]) | {"pick": "Pass" if prior >= 0.5 else "Run"},
          "tendency": scores(tendency_p(tt, prior, te), te["y"]),
          "importances": sorted(({"feature": k, "share": v / tot} for k, v in imp.items()), key=lambda r: -r["share"])}

    # leave-one-game-out (optimistic: later games help predict earlier ones)
    hits = maj = tab = 0
    for g in df["gid"].unique():
        inn, out = df[df["gid"] != g], df[df["gid"] == g]
        b = xgb.train({k: v for k, v in PARAMS.items() if k != "eval_metric"}, _dm(X.loc[inn.index], inn["y"]), rounds)
        y = out["y"].to_numpy()
        t_, p_ = tendency(inn)
        hits += ((b.predict(_dm(X.loc[out.index])) >= 0.5) == y).sum()
        maj += (y == int(inn["y"].mean() >= 0.5)).sum()
        tab += ((tendency_p(t_, p_, out) >= 0.5) == y).sum()
    ev["logo"] = {"acc": hits / len(df), "majority_acc": maj / len(df), "tendency_acc": tab / len(df), "n": len(df), "optimistic": True}

    # run direction L/R on Run snaps only, same split
    runs = df[(df["y"] == 0) & df["play_dir"].isin(["L", "R"])]
    rp = {k: runs[runs["gid"].isin(v)] for k, v in sp.items()}
    dmodel = None
    if len(rp["train"]) < 15 or rp["train"]["play_dir"].nunique() < 2 or not len(rp["test"]) or len(rp["val"]) < 5:
        ev["direction"] = {"skipped": f"not enough Run snaps with a direction (train {len(rp['train'])}, val {len(rp['val'])}, test {len(rp['test'])})"}
    else:
        yd = {k: (v["play_dir"] == "R").astype(int) for k, v in rp.items()}
        dmodel = _fit(*(_dm(X.loc[rp[k].index], yd[k]) for k in ("train", "val")))
        right = float(pd.concat([yd["train"], yd["val"]]).mean())
        ev["direction"] = {"model": scores(dmodel.predict(_dm(X.loc[rp["test"].index])), yd["test"]),
                           "majority": scores(np.full(len(rp["test"]), right), yd["test"]) | {"pick": "R" if right >= 0.5 else "L"}}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "model.json")
    if dmodel is not None:
        dmodel.save_model(out_dir / "direction_model.json")
    metrics = {"data": {"rows": len(df), "games": int(df["gid"].nunique()), "pass_rate": float(df["y"].mean()),
                        **{k: len(v) for k, v in part.items()}, **{f"{k}_games": v for k, v in sp.items()}},
               "rounds": rounds, "eval": ev}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out_dir / "categories.json").write_text(json.dumps(cats))
    return metrics


def predict(model_dir, data) -> pd.DataFrame:
    """Score snaps (labels not needed) with a trained model dir: adds p_pass, pick and, if trained, p_right."""
    model_dir = Path(model_dir)
    df = load(data, labelled=False)
    X = matrix(df, json.loads((model_dir / "categories.json").read_text()))
    m = xgb.Booster()
    m.load_model(model_dir / "model.json")
    out = df.drop(columns=["gid", "y"]).copy()
    out[CAT] = out[CAT].replace("blank", "")
    out["p_pass"] = m.predict(_dm(X)).round(3)
    out["pick"] = np.where(out["p_pass"] >= 0.5, "Pass", "Run")
    if (model_dir / "direction_model.json").exists():
        d = xgb.Booster()
        d.load_model(model_dir / "direction_model.json")
        out["p_right"] = d.predict(_dm(X)).round(3)   # "if it's a run, which way"
    return out


def summary(m: dict) -> str:
    """Human-readable lines for a metrics dict."""
    e, d = m["eval"], m["data"]
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    lines = [f"{d['rows']} snaps, {d['games']} games; test = {', '.join(d['test_games'])} ({d['test']} snaps)",
             f"test accuracy      model {pct(e['model']['acc'])}   always {e['majority']['pick']} {pct(e['majority']['acc'])}"
             f"   down x distance table {pct(e['tendency']['acc'])}"]
    lines += [f"confident >= {c['t']:.0%}   {pct(c['share'])} of snaps, right {pct(c['acc'])}" for c in e["model"]["conf"]]
    lines.append(f"leave-one-game-out {pct(e['logo']['acc'])} (optimistic; majority {pct(e['logo']['majority_acc'])}, table {pct(e['logo']['tendency_acc'])})")
    dr = e["direction"]
    lines.append(dr["skipped"] if "skipped" in dr else
                 f"run direction      model {pct(dr['model']['acc'])}   always {dr['majority']['pick']} {pct(dr['majority']['acc'])}  ({dr['model']['n']} test runs)")
    return "\n".join(lines)
