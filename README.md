# College basketball backtest for the Daily Betting Board

Replays the playbook's men's college basketball game-line rules (moneyline, spread, total) on
every regular-season Division I game since the 2023-24 season, using ratings rebuilt from box
scores with only pre-game information, and grades the bets against sportsbook opening/closing
lines from ESPN. Findings are in `PROJECT_SUMMARY.md`. The NBA version lives in
[nba-backtest](https://github.com/trentcarroccia10/nba-backtest).

## Run it

```bash
pip install -r requirements.txt     # sports-betting, pandas, numpy, scikit-learn
python fetch_data.py                 # ESPN: games, box scores, open/close lines -> data/games.jsonl (re-run = only new games)
python fetch_data.py 2023            # 2022-23 season, used only as last-season priors
python backtest.py                   # playbook rules -> results/summary.json, bets_*.csv, data_checks.csv
python library_backtest.py           # same moneyline rules + a logistic model through sportsbet.evaluation.backtest
python make_report.py                # results/report.html
```

`data/` and `results/` are committed, so `python backtest.py` works straight after cloning.
Downloading from scratch needs network access to `site.api.espn.com` and
`sports.core.api.espn.com` and takes about 30 minutes (about 6,000 games a season).

## Files

| File | What it does |
|---|---|
| `fetch_data.py` | Downloads completed Division I games, team box scores, conferences and the opening/closing lines (DraftKings or ESPN BET, whichever ESPN carried). |
| `model_cbb.py` | Opponent-adjusted offense, defense and tempo, refitted every day from earlier games only (in place of BartTorvik's ratings, which have no day-by-day history). MB-2 regression with k = 8 toward last season, home edge 3.3, σ 10.5 / 13, conference flags for the screening rule. |
| `backtest.py` | Playbook rules: 50/50 market blend, 3% edge, +2% EV, >10% = DATA CHECK, one bet per game, the addendum's 100+ game power-conference screen, tiers, flat and ¼-Kelly stakes, CLV, threshold sweep, variants. Edit `Rules` to test a change. |
| `library_backtest.py` | `PlaybookBettor`, a `sportsbet` `BaseBettor` subclass, and a `ClassifierBettor`, both scored with `sportsbet.evaluation.backtest`. |
| `make_report.py` | Builds the HTML report from the results. |

## Not covered

BartTorvik's own ratings (rebuilt here, close but not identical), injuries and suspensions, the
five-reason deep dive, Pinnacle fair prices (MB-11; needs a paid Odds API plan), and the NCAA
tournament.
