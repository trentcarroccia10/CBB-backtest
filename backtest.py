"""Backtest the playbook's college basketball rules on past games.

For each game and each market (moneyline, spread, total) the model is priced against the
DraftKings line, with the playbook's rules:
  * edge = blended probability - no-vig market probability (blend 50/50 with the market)
  * bet only if edge >= threshold (3%) AND expected value >= +2% at the price
  * edge above 10% = DATA CHECK, never bet (tracked separately so we can see if they'd have won)
  * one bet per game (the biggest edge), flat 1 unit; 1/4 Kelly capped at 2 units also reported
  * tiers: A >= 6%, B 4-6%, C 3-4%
Bets are placed at the OPENING line (closest to the board's 5pm price that ESPN keeps) and
closing line value (CLV) is measured against the DraftKings close. A second run bets at the
close, which is the harder test (no line-movement help).

Run: python backtest.py   -> results/summary.json, results/bets_*.csv, results/data_checks.csv
Applies the addendum's screening rule: on days with 100+ games, only games with
an ACC / Big East / Big Ten / Big 12 / SEC team (or Notre Dame) are considered.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

CBB = True  # this repository is college basketball only
from model_cbb import SIGMA_MARGIN, SIGMA_TOTAL, build, implied, novig, phi

ROOT = Path(__file__).parent
RES = ROOT / "results"
N = NormalDist()


@dataclass(frozen=True)
class Rules:
    threshold: float = 0.03
    min_ev: float = 0.02
    data_check: float = 0.10
    blend: str = "always"          # always | none | early (blend only while either team has < 20 games)
    price: str = "open"            # open | close : which line the bet is placed at
    markets: tuple = ("ml", "spread", "total")
    one_per_game: bool = True
    playoffs: bool = False
    max_fav: float = -1000         # skip moneyline favourites shorter than this
    screen: bool = True            # college only: 100+ game days -> power-conference games only
    early_n: int = 10 if CBB else 20   # "early season" = either team has played fewer games than this


# ESPN occasionally stores a price (e.g. -110) where the spread or total number belongs.
SPREAD_MAX = 60 if CBB else 30
TOTAL_RANGE = (90, 200) if CBB else (170, 280)


def ok_spread(x) -> bool:
    return x is not None and not pd.isna(x) and abs(x) <= SPREAD_MAX


def ok_total(x) -> bool:
    return x is not None and not pd.isna(x) and TOTAL_RANGE[0] <= x <= TOTAL_RANGE[1]


def payout(o: float) -> float:
    return o / 100 if o > 0 else 100 / -o


def candidates(g: dict, r: Rules) -> list[dict]:
    """Every side of every market for one game, priced at r.price."""
    w = r.price
    out = []
    m, t = g["proj_margin"], g["proj_total"]
    early = min(g["n_home"], g["n_away"]) < r.early_n
    blend_w = {"always": 0.5, "none": 0.0, "early": 0.5 if early else 0.0}[r.blend]

    def add(market, side, line, px, nv, p_model, won, push, clv):
        if px is None or nv is None or pd.isna(px) or pd.isna(nv):
            return
        p = (1 - blend_w) * p_model + blend_w * nv
        out.append({"market": market, "side": side, "line": line, "price": px, "nv": nv, "p_model": p_model,
                    "p": p, "edge": p - nv, "ev": p * payout(px) - (1 - p), "won": won, "push": push, "clv": clv})

    margin = g["home_pts"] - g["away_pts"]
    total = g["home_pts"] + g["away_pts"]
    have_close = w == "open"

    if "ml" in r.markets:
        nv_h = novig(g[f"{w}_ml_h"], g[f"{w}_ml_a"])
        nvc = novig(g["close_ml_h"], g["close_ml_a"]) if have_close else None
        if nv_h is not None:
            ph = phi(m / SIGMA_MARGIN)
            add("ml", "home", None, g[f"{w}_ml_h"], nv_h, ph, margin > 0, False,
                None if nvc is None else nvc - nv_h)
            add("ml", "away", None, g[f"{w}_ml_a"], 1 - nv_h, 1 - ph, margin < 0, False,
                None if nvc is None else (1 - nvc) - (1 - nv_h))

    if "spread" in r.markets and ok_spread(g[f"{w}_sp_h"]):
        L = g[f"{w}_sp_h"]
        hp, ap = g[f"{w}_sp_h_px"], g[f"{w}_sp_a_px"]
        hp = -110.0 if pd.isna(hp) else hp
        ap = -110.0 if pd.isna(ap) else ap
        nv_h = novig(hp, ap)
        ph = phi((m + L) / SIGMA_MARGIN)
        clv_h = None
        if have_close and ok_spread(g["close_sp_h"]):
            chp = -110.0 if pd.isna(g["close_sp_h_px"]) else g["close_sp_h_px"]
            cap = -110.0 if pd.isna(g["close_sp_a_px"]) else g["close_sp_a_px"]
            nvc = novig(chp, cap)
            # close probability translated to the line we bet (normal approximation)
            pc_at_L = N.cdf(N.inv_cdf(min(max(nvc, 1e-4), 1 - 1e-4)) + (L - g["close_sp_h"]) / SIGMA_MARGIN)
            clv_h = pc_at_L - nv_h
        push = (margin + L) == 0
        add("spread", "home", L, hp, nv_h, ph, margin + L > 0, push, clv_h)
        add("spread", "away", -L, ap, 1 - nv_h, 1 - ph, margin + L < 0, push,
            None if clv_h is None else -clv_h)

    if "total" in r.markets and ok_total(g[f"{w}_tot"]):
        T = g[f"{w}_tot"]
        op, up = g[f"{w}_ov_px"], g[f"{w}_un_px"]
        op = -110.0 if pd.isna(op) else op
        up = -110.0 if pd.isna(up) else up
        nv_o = novig(op, up)
        po = phi((t - T) / SIGMA_TOTAL)
        clv_o = None
        if have_close and ok_total(g["close_tot"]):
            cop = -110.0 if pd.isna(g["close_ov_px"]) else g["close_ov_px"]
            cup = -110.0 if pd.isna(g["close_un_px"]) else g["close_un_px"]
            nvc = novig(cop, cup)
            pc_at_T = N.cdf(N.inv_cdf(min(max(nvc, 1e-4), 1 - 1e-4)) - (T - g["close_tot"]) / SIGMA_TOTAL)
            clv_o = pc_at_T - nv_o
        push = total == T
        add("total", "over", T, op, nv_o, po, total > T, push, clv_o)
        add("total", "under", T, up, 1 - nv_o, 1 - po, total < T, push, None if clv_o is None else -clv_o)
    return out


def tier(e: float) -> str:
    return "A" if e >= 0.06 else "B" if e >= 0.04 else "C"


def run(games: pd.DataFrame, r: Rules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (bets, data_checks)."""
    bets, checks = [], []
    g_iter = games if r.playoffs else games[games.stype == 2]
    if CBB and r.screen:
        g_iter = g_iter[(g_iter.day_games < 100) | g_iter.power.astype(bool)]
    for g in g_iter.to_dict("records"):
        c = candidates(g, r)
        if not c:
            continue
        ok, dc = [], []
        for x in c:
            if x["market"] == "ml" and x["price"] < 0 and x["price"] < r.max_fav:
                continue
            if x["edge"] > r.data_check:
                dc.append(x)
            elif x["edge"] >= r.threshold and x["ev"] >= r.min_ev:
                ok.append(x)
        if r.one_per_game and ok:
            ok = [max(ok, key=lambda x: x["edge"])]
        meta = {"date": g["date"], "season": g["season"], "game": f'{g["away"]} @ {g["home"]}',
                "proj_margin": g["proj_margin"], "proj_total": g["proj_total"]}
        bets += [meta | x for x in ok]
        checks += [meta | x for x in dc]
    return pd.DataFrame(bets), pd.DataFrame(checks)


def settle(b: pd.DataFrame) -> pd.DataFrame:
    if b.empty:
        return b
    b = b.copy()
    b["units"] = np.where(b["push"], 0.0, np.where(b["won"], b["price"].map(payout), -1.0))
    f = ((b["price"].map(payout) * b["p"] - (1 - b["p"])) / b["price"].map(payout)).clip(lower=0)
    b["kelly_stake"] = np.minimum(2.0, 0.25 * f * 100)          # 100-unit bankroll, 1/4 Kelly, cap 2u
    b["kelly_units"] = b["kelly_stake"] * b["units"]
    b["tier"] = b["edge"].map(tier)
    return b.sort_values("date")


def stats(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0}
    s = b[~b["push"]]
    curve = b["units"].cumsum()
    dd = float((curve.cummax() - curve).max())
    streak = cur = 0
    for u in b["units"]:
        cur = cur + 1 if u < 0 else 0 if u > 0 else cur
        streak = max(streak, cur)
    roi = b["units"].sum() / len(b)
    se = b["units"].std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else np.nan
    clv = b["clv"].dropna() if "clv" in b else pd.Series(dtype=float)
    return {
        "bets": int(len(b)), "wins": int(s["won"].sum()), "losses": int((~s["won"]).sum()), "pushes": int(b["push"].sum()),
        "win_pct": round(100 * s["won"].mean(), 1) if len(s) else None,
        "units": round(float(b["units"].sum()), 2), "roi_pct": round(100 * roi, 2),
        "roi_ci95": [round(100 * (roi - 1.96 * se), 1), round(100 * (roi + 1.96 * se), 1)] if se == se else None,
        "kelly_units": round(float(b["kelly_units"].sum()), 2),
        "avg_edge_pct": round(100 * b["edge"].mean(), 2), "avg_price": round(float(b["price"].mean()), 1),
        "clv_beat_pct": round(100 * (clv > 0).mean(), 1) if len(clv) else None,
        "avg_clv_pct": round(100 * clv.mean(), 2) if len(clv) else None,
        "max_drawdown_units": round(dd, 2), "longest_losing_streak": int(streak),
        "profit_usd_at_25": round(25 * float(b["units"].sum()), 0),
    }


def model_quality(g: pd.DataFrame) -> dict:
    """How good are the raw projections compared with the DraftKings closing line?"""
    g = g[g.stype == 2].copy()
    g["margin"] = g.home_pts - g.away_pts
    g["total"] = g.home_pts + g.away_pts
    g["hw"] = (g.margin > 0).astype(float)
    g["nv_close"] = [novig(a, b) for a, b in zip(g.close_ml_h, g.close_ml_a)]
    q = g.dropna(subset=["nv_close"])
    eps = 1e-6

    def brier(p):
        return float(np.mean((p - q.hw) ** 2))

    def ll(p):
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(q.hw * np.log(p) + (1 - q.hw) * np.log(1 - p)))

    blend = 0.5 * q.p_home_win + 0.5 * q.nv_close
    gap = g[g.close_sp_h.map(ok_spread)]
    gap = gap.assign(d=gap.proj_margin + gap.close_sp_h, cover=(gap.margin + gap.close_sp_h) > 0)
    hi_q = gap[gap.d >= gap.d.quantile(0.8)]
    lo_q = gap[gap.d <= gap.d.quantile(0.2)]
    # side the model likes most: home in the top fifth, away in the bottom fifth
    top_cover = float((hi_q.cover.sum() + (~lo_q.cover).sum()) / (len(hi_q) + len(lo_q))) if len(hi_q) else None
    sp = g[g.close_sp_h.map(ok_spread)]
    to = g[g.close_tot.map(ok_total)]
    by_season = {}
    for s, x in sp.groupby("season"):
        by_season[int(s)] = {"model_margin_mae": round(float((x.proj_margin - x.margin).abs().mean()), 2),
                             "close_spread_mae": round(float((-x.close_sp_h - x.margin).abs().mean()), 2)}
    return {
        "games": int(len(q)),
        "winner": {"model_brier": round(brier(q.p_home_win), 4), "market_brier": round(brier(q.nv_close), 4),
                   "blend_brier": round(brier(blend), 4),
                   "model_logloss": round(ll(q.p_home_win), 4), "market_logloss": round(ll(q.nv_close), 4),
                   "blend_logloss": round(ll(blend), 4)},
        "margin": {"model_mae": round(float((sp.proj_margin - sp.margin).abs().mean()), 2),
                   "close_spread_mae": round(float((-sp.close_sp_h - sp.margin).abs().mean()), 2),
                   "corr_model_vs_close": round(float(np.corrcoef(sp.proj_margin, -sp.close_sp_h)[0, 1]), 3)},
        "total": {"model_mae": round(float((to.proj_total - to.total).abs().mean()), 2),
                  "close_total_mae": round(float((to.close_tot - to.total).abs().mean()), 2),
                  "corr_model_vs_close": round(float(np.corrcoef(to.proj_total, to.close_tot)[0, 1]), 3)},
        "margin_by_season": by_season,
        "biggest_disagreement_cover_pct": round(100 * top_cover, 1) if top_cover is not None else None,
    }


def breakdown(b: pd.DataFrame, col: str) -> dict:
    return {str(k): stats(v) for k, v in b.groupby(col)} if not b.empty else {}


def main() -> None:
    RES.mkdir(exist_ok=True)
    games = build()
    games["date"] = pd.to_datetime(games["date"], utc=True)
    have = games.dropna(subset=["open_ml_h", "open_sp_h"], how="all")
    summary: dict = {"games_total": int(len(games)), "games_with_open_lines": int(len(have)),
                     "seasons": sorted(int(s) for s in games.season.unique()),
                     "books": games.book.value_counts().to_dict(),
                     "model_quality": model_quality(games)}

    base = Rules()
    # first season has no prior-season ratings unless 2023 was downloaded; skip seasons with no prior
    first = int(games.season.min())
    test = games[games.season > first]
    summary["test_seasons"] = sorted(int(s) for s in test.season.unique())

    b, dc = run(test, base)
    b, dc = settle(b), settle(dc)
    b.to_csv(RES / "bets_playbook_open.csv", index=False)
    dc.to_csv(RES / "data_checks.csv", index=False)
    summary["playbook"] = {"rules": base.__dict__, "overall": stats(b),
                           "by_season": breakdown(b, "season"), "by_market": breakdown(b, "market"),
                           "by_tier": breakdown(b, "tier"),
                           "by_side": breakdown(b.assign(ms=b.market + ":" + b.side), "ms") if not b.empty else {},
                           "by_month": breakdown(b.assign(mo=b.date.dt.tz_convert("America/Chicago").dt.strftime("%m-%b")), "mo") if not b.empty else {},
                           "data_checks": stats(dc)}

    bc, _ = run(test, replace(base, price="close"))
    bc = settle(bc)
    bc.to_csv(RES / "bets_playbook_close.csv", index=False)
    summary["playbook_at_close"] = {"overall": stats(bc), "by_market": breakdown(bc, "market")}

    # threshold sweep x blend mode (bet at open)
    sweep = []
    for blend in ("always", "early", "none"):
        for th in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            r = replace(base, threshold=th, blend=blend, min_ev=0.0 if th == 0 else base.min_ev)
            x = settle(run(test, r)[0])
            st = stats(x)
            sweep.append({"blend": blend, "threshold_pct": th * 100, **{k: st.get(k) for k in
                          ("bets", "win_pct", "units", "roi_pct", "roi_ci95", "clv_beat_pct", "avg_clv_pct")}})
            for mk in ("ml", "spread", "total"):
                st = stats(x[x.market == mk]) if not x.empty else {"bets": 0}
                sweep.append({"blend": blend, "threshold_pct": th * 100, "market": mk,
                              **{k: st.get(k) for k in ("bets", "win_pct", "units", "roi_pct", "clv_beat_pct", "avg_clv_pct")}})
    summary["sweep"] = sweep

    # variants
    variants = {}
    for name, r in {
        "all_markets_per_game": replace(base, one_per_game=False),
        "no_blend": replace(base, blend="none"),
        "with_playoffs": replace(base, playoffs=True),
        **({"no_screen_all_games": replace(base, screen=False)} if CBB else {}),
        "no_ml_fav_shorter_than_-200": replace(base, max_fav=-200),
    }.items():
        variants[name] = stats(settle(run(test, r)[0]))
    summary["variants"] = variants

    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("games_total", "games_with_open_lines", "test_seasons", "model_quality")}, indent=1))
    print("PLAYBOOK @open", json.dumps(summary["playbook"]["overall"]))
    print("PLAYBOOK @close", json.dumps(summary["playbook_at_close"]["overall"]))


if __name__ == "__main__":
    main()
