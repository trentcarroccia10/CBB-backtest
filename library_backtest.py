"""Run the moneyline side through the sports-betting library's own bettors and backtest().

Two bettors, both scored by sportsbet.evaluation.backtest with forward-in-time folds:
  1. PlaybookBettor  - your rules as a library bettor: blended model probability, 3% edge,
                       +2% EV, edges above 10% skipped (DATA CHECK).
  2. ClassifierBettor - the library's standard wrapper around a scikit-learn model: a logistic
                       regression that LEARNS from the model's projections, rest and the
                       market price, retrained on each fold.

Run: python library_backtest.py   (needs data/games_model.csv from backtest.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sportsbet.evaluation import BaseBettor, ClassifierBettor, backtest

from model_cbb import implied

ROOT = Path(__file__).parent
CBB = True  # this repository is college basketball only
DATA = ROOT / "data"
OUT = ROOT / "results"
MARKETS = ["home_win", "away_win"]


def to_decimal(a):
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def library_frames(price: str = "open"):
    g = pd.read_csv(DATA / "games_model.csv", parse_dates=["date"])
    g = g[(g.stype == 2) & (g.season > g.season.min())].dropna(subset=[f"{price}_ml_h", f"{price}_ml_a"])
    if CBB:  # same screening rule as backtest.py
        g = g[(g.day_games < 100) | g.power.astype(bool)]
    ih = g[f"{price}_ml_h"].map(implied)
    ia = g[f"{price}_ml_a"].map(implied)
    idx = pd.DatetimeIndex(g["date"], name="date")
    X = pd.DataFrame({
        "p_model_home": g["p_home_win"].to_numpy(),
        "proj_margin": g["proj_margin"].to_numpy(),
        "h_b2b": g["h_b2b"].astype(float).to_numpy(),
        "a_b2b": g["a_b2b"].astype(float).to_numpy(),
        "nv_home": (ih / (ih + ia)).to_numpy(),
    }, index=idx)
    Y = pd.DataFrame({
        "home_win__postplay__0min": (g.home_pts > g.away_pts).astype(int).to_numpy(),
        "away_win__postplay__0min": (g.home_pts < g.away_pts).astype(int).to_numpy(),
    }, index=idx)
    O = pd.DataFrame({
        "dk__home_win__preplay__0min": to_decimal(g[f"{price}_ml_h"].to_numpy()),
        "dk__away_win__preplay__0min": to_decimal(g[f"{price}_ml_a"].to_numpy()),
    }, index=idx)
    return X, Y, O


class PlaybookBettor(BaseBettor):
    """The playbook's moneyline rules as a sports-betting bettor."""

    def __init__(self, betting_markets=None, init_cash=None, stake=None,
                 threshold=0.03, min_ev=0.02, data_check=0.10, blend=0.5):
        super().__init__(betting_markets=betting_markets, init_cash=init_cash, stake=stake)
        self.threshold = threshold
        self.min_ev = min_ev
        self.data_check = data_check
        self.blend = blend

    def _fit(self, X, Y, O):
        return self  # rule-based: the ratings are already built from past games only

    def _predict_proba(self, X):
        ph = (1 - self.blend) * X["p_model_home"].to_numpy() + self.blend * X["nv_home"].to_numpy()
        return np.column_stack([ph, 1 - ph])

    def bet(self, X, O):
        P = self.predict_proba(X)
        nv = np.column_stack([X["nv_home"].to_numpy(), 1 - X["nv_home"].to_numpy()])
        dec = O[["dk__home_win__preplay__0min", "dk__away_win__preplay__0min"]].to_numpy()
        edge, ev = P - nv, P * dec - 1
        ok = (edge >= self.threshold) & (edge <= self.data_check) & (ev >= self.min_ev)
        best = np.argmax(np.where(ok, edge, -np.inf), axis=1)
        out = np.zeros_like(ok)
        rows = np.where(ok.any(axis=1))[0]
        out[rows, best[rows]] = True
        return out


def main():
    X, Y, O = library_frames()
    cv = TimeSeriesSplit(5)
    playbook = PlaybookBettor(betting_markets=MARKETS, init_cash=2500.0, stake=25.0)
    res_pb = backtest(playbook, X, Y, O, cv=cv, n_jobs=1)

    clf = make_pipeline(StandardScaler(), MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    learned = ClassifierBettor(clf, betting_markets=MARKETS, init_cash=2500.0, stake=25.0)
    res_lr = backtest(learned, X, Y, O, cv=cv, n_jobs=1)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    res_pb.to_csv(out / "library_backtest_playbook.csv")
    res_lr.to_csv(out / "library_backtest_logistic.csv")
    print("== PlaybookBettor (your rules) via sportsbet.backtest ==")
    print(res_pb)
    print("\n== ClassifierBettor (logistic regression) via sportsbet.backtest ==")
    print(res_lr)


if __name__ == "__main__":
    main()
