# College basketball backtest: findings for the Sports quant project

Code and data: https://github.com/trentcarroccia10/cbb-backtest
Full report: https://claude.ai/artifact/S4CruQi3YBVJWQwzegHqj6
NBA version: https://github.com/trentcarroccia10/nba-backtest
Run date: November 18, 2026

## What was tested

The playbook's men's college basketball game-line rules (moneyline, spread, total) replayed on
18,051 regular-season Division I games with sportsbook lines (ESPN BET or DraftKings, whichever
ESPN carried), the 2023-24 season to November 17, 2026. The 2022-23 season gave each team its
starting rating. The addendum's screening rule was applied: on days with 100+ games, only games
with an ACC, Big East, Big Ten, Big 12 or SEC team (or Notre Dame).

Model: BartTorvik doesn't publish day-by-day history, so opponent-adjusted offense, defense and
tempo were rebuilt from box scores and refitted every day from earlier games only. MB-2
regression with k = 8 games toward last season, home edge 3.3 points, margin spread 10.5, total
spread 13. Rules: 50/50 blend with the market, 3% edge, +2% EV, edges over 10% treated as DATA
CHECK and never bet, one bet per game, flat 1 unit ($25).

## Results

| | Result |
|---|---|
| Bets | 9,789 |
| Bet at the opening line | -378.9 units, -$9,472, -3.9% per bet (likely range -5.9% to -1.9%) |
| Bet at the closing line | -527.3 units, -$13,183, -5.4% per bet |
| Win rate | 50.4% (break-even at -110 is about 52.4%) |
| Moneyline / spread / total | -12.0% / -3.2% / -1.5% per bet |
| Tiers A / B / C | -3.0% / -3.7% / -8.3% |
| Beat the closing line | 53.4% of bets, by about 1.1 points with the cut removed |

- Road moneyline underdogs: 801 bets, won 32%, -24.0% per bet (range -35.2% to -12.8%). The clearest loser.
- Home moneylines: -0.5% per bet (range -6.5% to +5.5%).
- By month: November -6.5%, December -2.8%, January -1.9%, February -0.3%, March -8.2% per bet.
- Higher thresholds don't fix it: 4% -3.3%, 5% -3.3%, 6% -3.0%, 8% -4.5% per bet.
- The 50/50 market blend helps: without it, -6.8% per bet.
- The screening rule barely matters: all games -3.5% per bet vs -3.9% screened.
- DATA CHECK edges (over 10%), if they had been bet: -3.1% per bet.
- The sports-betting library's bettor and backtest() agree: the moneyline rules lose in four of
  five forward-in-time windows, and the library's logistic model loses in all five.

## Why

The model tracks the closing line closely (correlation 0.87 on margins) but is less accurate:
average miss on the margin is 9.5 points vs 8.9 for the closing spread, and on the total 13.4 vs
13.0. Where the model disagreed most with the spread, the side it liked covered 50.3%, short of
the 52.4% needed at -110. Bets beat the close by about 1.1 points on average, less than the
bookmaker's cut of about 2.4 points per side at -110.

## Not tested

BartTorvik's own ratings (rebuilt here, close but not identical), injuries and suspensions, the
five-reason deep dive and PENDING holds, Pinnacle fair prices (MB-11), and the NCAA tournament.

## Suggested playbook changes for college

1. A ratings-only college edge is a lean at most. A bet also needs news the price hasn't caught up with.
2. Don't bet road moneyline underdogs on ratings alone.
3. Be most careful in November, and keep the 50/50 market blend.
4. In the CLV check, count a bet as beating the close only if it beat it by more than the cut
   (about 2.4 points at -110).
5. Rerun the backtest (`python backtest.py`) before adopting any rule change.
