"""Rebuild the playbook's college basketball model as it would have looked before every past game.

The playbook uses BartTorvik's adjusted efficiency (AdjOE / AdjDE / AdjT). Torvik doesn't
publish day-by-day history, so this rebuilds the same kind of number from box scores:
  * Opponent-adjusted offense, defense and tempo, fitted each day from ONLY the games played
    before that day (a ridge regression, so a team's rating accounts for who it played).
  * MB-2 regression with k = 8 games: each team's rating is pulled toward its prior as if it
    had 8 extra games at the prior level. The prior is the team's final rating from last
    season (0 = league average for teams with no history).
  * Home edge 3.3 points per game (none at neutral sites), split evenly between the teams.
  * Projected score = tempo x adjusted efficiency; margin spread 10.5, total spread 13.

Writes data/games_model.csv, one row per game, plus conference columns
and the number of games played that day (for the addendum's 100+ game screening rule).
"""
from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
K_GAMES = 8.0          # MB-2 regression strength for college basketball
HOME_EDGE = 3.3        # points per game
SIGMA_MARGIN = 10.5
SIGMA_TOTAL = 13.0
POWER_CONFS = {"2", "4", "7", "8", "23"}   # ACC, Big East, Big Ten, Big 12, SEC
NOTRE_DAME = "87"


def phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def possessions(b: dict) -> float:
    return b["fga"] - b["oreb"] + b["tov"] + 0.475 * b["fta"]   # 0.475 is the usual college FT factor


def load_games() -> pd.DataFrame:
    rows = [json.loads(line) for line in (DATA / "games.jsonl").open()]
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["local_day"] = df["date"].dt.tz_convert("America/New_York").dt.date
    df["day_games"] = df.groupby("local_day")["id"].transform("size")
    return df.sort_values("date").reset_index(drop=True)


class SeasonFit:
    """Accumulates normal equations so each day's fit is one small linear solve.

    Efficiency model, one row per team-game:  eff_off = mu + o[off] + d[def] +/- h
    Tempo model, one row per game:           tempo   = muT + p[home] + p[away]
    o > 0 = good offence; d > 0 = bad defence (allows more); p > 0 = plays fast.
    """

    def __init__(self, teams: list[str], prior: dict, lg_prior: dict):
        self.ix = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.n = n
        # efficiency unknowns: [mu, o_0..o_n-1, d_0..d_n-1]; tempo unknowns: [muT, p_0..p_n-1]
        self.Me = np.zeros((1 + 2 * n, 1 + 2 * n))
        self.ve = np.zeros(1 + 2 * n)
        self.Mt = np.zeros((1 + n, 1 + n))
        self.vt = np.zeros(1 + n)
        self.games = np.zeros(n)
        self.lg_eff = []
        self.lg_tempo = []
        # MB-2 priors as k pseudo-games (ridge toward last season's rating)
        mu0, t0 = lg_prior["eff"], lg_prior["tempo"]
        self._ridge(self.Me, self.ve, 0, mu0, 20.0)
        self._ridge(self.Mt, self.vt, 0, t0, 20.0)
        for t, i in self.ix.items():
            pr = prior.get(t, {"o": 0.0, "d": 0.0, "p": 0.0})
            self._ridge(self.Me, self.ve, 1 + i, pr["o"], K_GAMES)
            self._ridge(self.Me, self.ve, 1 + n + i, pr["d"], K_GAMES)
            self._ridge(self.Mt, self.vt, 1 + i, pr["p"], K_GAMES)
        self.sol_e = None
        self.sol_t = None

    @staticmethod
    def _ridge(M, v, j, target, w):
        M[j, j] += w
        v[j] += w * target

    @staticmethod
    def _row(M, v, cols, y):
        for a in cols:
            v[a] += y
            for b in cols:
                M[a, b] += 1.0

    def add(self, g: dict) -> None:
        b = g["box"]
        poss = (possessions(b["home"]) + possessions(b["away"])) / 2
        minutes = 40 + 5 * max(0, g["periods"] - 2)
        tempo = poss * 40 / minutes
        h = 0.0 if g["neutral"] else HOME_EDGE / 2 / (tempo / 100)   # home edge in points per 100
        ih, ia = self.ix[g["home"]], self.ix[g["away"]]
        n = self.n
        eff_h = 100 * g["home_pts"] / poss
        eff_a = 100 * g["away_pts"] / poss
        self._row(self.Me, self.ve, (0, 1 + ih, 1 + n + ia), eff_h - h)
        self._row(self.Me, self.ve, (0, 1 + ia, 1 + n + ih), eff_a + h)
        self._row(self.Mt, self.vt, (0, 1 + ih, 1 + ia), tempo)
        self.games[ih] += 1
        self.games[ia] += 1
        self.lg_eff += [eff_h, eff_a]
        self.lg_tempo.append(tempo)

    def solve(self) -> None:
        self.sol_e = np.linalg.solve(self.Me, self.ve)
        self.sol_t = np.linalg.solve(self.Mt, self.vt)

    def team(self, t: str) -> dict:
        i, n = self.ix[t], self.n
        return {"o": self.sol_e[1 + i], "d": self.sol_e[1 + n + i], "p": self.sol_t[1 + i], "n": int(self.games[i])}

    def league(self) -> tuple[float, float]:
        return float(self.sol_e[0]), float(self.sol_t[0])

    def final_priors(self) -> tuple[dict, dict]:
        self.solve()
        pri = {t: {k: float(v) for k, v in self.team(t).items() if k != "n"} for t in self.ix}
        mu, mut = self.league()
        return pri, {"eff": mu, "tempo": mut}


def implied(o):
    if o is None or pd.isna(o) or abs(o) < 100:   # American odds are never between -100 and +100
        return None
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def novig(a, b):
    ia, ib = implied(a), implied(b)
    if ia is None or ib is None:
        return None
    return ia / (ia + ib)


def build() -> pd.DataFrame:
    df = load_games()
    df = df[df["box"].notna()]
    out = []
    prior, lg_prior = {}, {"eff": 104.0, "tempo": 68.0}
    for season, sg in df.groupby("season", sort=True):
        teams = sorted(set(sg["home"]) | set(sg["away"]))
        fit = SeasonFit(teams, prior, lg_prior)
        for _, day in sg.groupby("local_day", sort=True):
            fit.solve()
            mu, mut = fit.league()
            day_rows = day.to_dict("records")
            for g in day_rows:
                h, a = fit.team(g["home"]), fit.team(g["away"])
                tempo = mut + h["p"] + a["p"]
                edge = 0.0 if g["neutral"] else HOME_EDGE / 2
                hp = tempo * (mu + h["o"] + a["d"]) / 100 + edge
                ap = tempo * (mu + a["o"] + h["d"]) / 100 - edge
                margin, total = hp - ap, hp + ap
                power = bool({g.get("home_conf"), g.get("away_conf")} & POWER_CONFS) or NOTRE_DAME in (g["home"], g["away"])
                row = {
                    "id": g["id"], "date": g["date"], "season": season, "stype": g["stype"],
                    "home": g.get("home_name") or g["home"], "away": g.get("away_name") or g["away"],
                    "home_conf": g.get("home_conf"), "away_conf": g.get("away_conf"), "power": power,
                    "day_games": int(g["day_games"]), "neutral": g["neutral"],
                    "home_pts": g["home_pts"], "away_pts": g["away_pts"],
                    "n_home": h["n"], "n_away": a["n"], "h_b2b": False, "a_b2b": False,
                    "proj_home": round(hp, 2), "proj_away": round(ap, 2),
                    "proj_margin": round(margin, 2), "proj_total": round(total, 2),
                    "p_home_win": phi(margin / SIGMA_MARGIN),
                }
                o = g.get("odds") or {}
                row["book"] = o.get("book")
                for when in ("open", "close"):
                    q = (o.get(when) or {})
                    for k in ("ml_h", "ml_a", "sp_h", "sp_h_px", "sp_a_px", "tot", "ov_px", "un_px"):
                        row[f"{when}_{k}"] = q.get(k)
                out.append(row)
            for g in day_rows:          # results only enter the ratings after the whole day is projected
                fit.add(g)
        prior, lg_prior = fit.final_priors()
    res = pd.DataFrame(out)
    res.to_csv(DATA / "games_model.csv", index=False)
    return res


if __name__ == "__main__":
    r = build()
    print(len(r), "games projected ->", DATA / "games_model.csv")
    print(r.groupby("season").size())
