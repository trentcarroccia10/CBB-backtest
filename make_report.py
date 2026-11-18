"""Turn the results into report.html (the published page).

  python make_report.py   -> results/report.html
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
CBB = True  # this repository is college basketball only
RES = ROOT / "results"

LEAGUE = {
    False: {
        "title": "NBA Model Backtest", "eyebrow": "NBA game lines", "rules_name": "NBA",
        "games_desc": "regular-season games with DraftKings lines",
        "warmup": "The 2022-23 season was used only to give each team a starting rating.",
        "screen": "",
        "cant_see": ["Injury adjustments for the 20–30 players who move lines (MB-5), and late lineup news the market hasn't priced yet.",
                     "The five-reason deep dive and PENDING holds. A human filter could screen out many of these bets.",
                     "Pinnacle as the fair price (MB-11). ESPN only keeps DraftKings' history, so getting Pinnacle's old prices needs a paid Odds API plan.",
                     "Player props, NHL, NFL and college games."],
        "model_note": "Model: pace × efficiency, regressed with k = 15 games, home edge 2.5 points, 1.5 points off for the second night of a back-to-back, spread σ 12, total σ 18.",
        "sources": "ESPN scoreboard, box score and odds feeds (DraftKings opening and closing lines; ESPN BET or another book where DraftKings wasn't listed).",
        "changes": ["Treat a ratings-only edge as a lean at most. Only call something a bet when there's news the price hasn't caught up with yet.",
                    "Keep logging closing prices, but only count a beat when it's bigger than the cut (about 2.4 points at −110).",
                    "Rerun this backtest after any rule change. It takes about five seconds once the data is downloaded (python backtest.py)."],
    },
    True: {
        "title": "College Hoops Backtest", "eyebrow": "Men's college basketball game lines", "rules_name": "college basketball",
        "games_desc": "regular-season Division I games with sportsbook lines",
        "warmup": "The 2022-23 season was used only to give each team a starting rating.",
        "screen": "Following the addendum, on days with 100+ games only games with an ACC, Big East, Big Ten, Big 12 or SEC team (or Notre Dame) are considered.",
        "cant_see": ["BartTorvik's own ratings. Torvik doesn't publish day-by-day history, so the model rebuilds opponent-adjusted efficiency and tempo from box scores the same way. It's close, but not identical.",
                     "Injuries, suspensions and late lineup news, which matter more in college because rotations are short.",
                     "The five-reason deep dive and PENDING holds.",
                     "Pinnacle as the fair price (MB-11), and the NCAA tournament (regular season only)."],
        "model_note": "Model: opponent-adjusted offense, defense and tempo, refitted daily from past games only, regressed with k = 8 games toward last season, home edge 3.3 points, spread σ 10.5, total σ 13.",
        "sources": "ESPN scoreboard, box score and odds feeds (ESPN BET or DraftKings opening and closing lines, whichever ESPN carried).",
        "changes": ["Treat a ratings-only college edge as a lean at most, the same as the NBA. Only call something a bet when there's news the price hasn't caught up with yet.",
                    "Don't bet road moneyline underdogs on ratings alone. They won 32% of the time and lost about 24¢ per dollar over 801 bets, the clearest loser in the test.",
                    "Be most careful in November. Early-season bets lost about 6.5% per bet, while January and February came close to breaking even. Keep the 50/50 market blend: without it, losses grew from 3.9% to 6.8% per bet.",
                    "Keep logging closing prices, but only count a beat when it's bigger than the cut (about 2.4 points at −110).",
                    "Rerun this backtest after any rule change (python backtest.py, about 30 seconds)."],
    },
}[CBB]


def money(u: float) -> str:
    d = round(25 * u)
    return ("+$" if d >= 0 else "−$") + f"{abs(d):,}"


def pct(x, signed=True) -> str:
    if x is None:
        return "—"
    return (f"{x:+.1f}%" if signed else f"{x:.1f}%").replace("-", "−")


def curve(path: Path) -> list[tuple[str, float]]:
    b = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    b["day"] = b["date"].dt.tz_convert("America/Chicago").dt.date
    daily = b.groupby("day")["units"].sum().cumsum()
    return [(str(d), round(float(v), 2)) for d, v in daily.items()]


def bar_rows(rows: list[dict], label_key: str) -> str:
    out = []
    lo = min(min(r["roi_pct"] for r in rows), -1)
    hi = max(max(r["roi_pct"] for r in rows), 1)
    span = hi - lo
    zero = (0 - lo) / span * 100
    for r in rows:
        v = r["roi_pct"]
        left = min(zero, (v - lo) / span * 100)
        width = abs(v) / span * 100
        cls = "pos" if v > 0 else "neg"
        ci = r.get("roi_ci95")
        ci_txt = f'range {pct(ci[0])} to {pct(ci[1])}' if ci else ""
        out.append(
            f'<div class="brow"><div class="blabel">{escape(str(r[label_key]))}</div>'
            f'<div class="btrack"><span class="bzero" style="left:{zero:.2f}%"></span>'
            f'<span class="bbar {cls}" style="left:{left:.2f}%;width:{max(width, 0.4):.2f}%"></span></div>'
            f'<div class="bval mono">{pct(v)}</div><div class="bmeta">{r["bets"]:,} bets · {ci_txt}</div></div>')
    return "\n".join(out)


def main() -> None:
    s = json.loads((RES / "summary.json").read_text())
    p, pc, q = s["playbook"], s["playbook_at_close"], s["model_quality"]
    o, oc = p["overall"], pc["overall"]
    lo, hi = o["roi_ci95"]
    cents = abs(o["roi_pct"])
    if hi < 0:
        verdict = f"No. As written, the rules would have lost about {cents:.1f}¢ on every dollar bet."
        luck = f"The likely range for the true return is {pct(lo)} to {pct(hi)} per bet, so this isn't bad luck: the whole range is below zero."
    elif lo > 0:
        verdict = f"Yes. As written, the rules would have made about {cents:.1f}¢ on every dollar bet."
        luck = f"The likely range for the true return is {pct(lo)} to {pct(hi)} per bet. The whole range is above zero, so it's unlikely to be luck."
    else:
        verdict = f"Not clearly. The rules would have {'lost' if o['roi_pct'] < 0 else 'made'} about {cents:.1f}¢ per dollar bet, but that's within the luck range."
        luck = f"The likely range for the true return is {pct(lo)} to {pct(hi)} per bet. That range includes zero, so this sample can't separate skill from luck."
    mm, cm = q["margin"]["model_mae"], q["margin"]["close_spread_mae"]
    if mm > cm:
        why_head = "Why: the model knows less than the market"
        acc = f"It's just a little less accurate than the closing line:"
    else:
        why_head = "How the model compares with the market"
        acc = "It's about as accurate as the closing line, or better:"
    cover = q.get("biggest_disagreement_cover_pct")
    cover_line = (f"In the 40% of games where the model disagreed most with the spread (20% each way), the side it liked covered "
                  f"<b>{cover:.1f}%</b> of the time. Break-even at −110 is 52.4%.") if cover is not None else ""
    sweep_all = [r for r in s["sweep"] if r["blend"] == "always" and "market" not in r and r["threshold_pct"] > 0]
    if all(r["roi_pct"] < 0 for r in sweep_all):
        sweep_head, sweep_note = ("Raising the edge threshold doesn't fix it",
                                  "If the edges were real, a higher bar would mean fewer but better bets. Instead the return stays negative at every level.")
    else:
        best = max(sweep_all, key=lambda r: r["roi_pct"])
        sweep_head = "Results at different edge thresholds"
        sweep_note = (f"The best level was {best['threshold_pct']:g}% ({pct(best['roi_pct'])} per bet on {best['bets']:,} bets). "
                      "Check its range before trusting it: picking the best of several thresholds flatters the result.")
    mk = p["by_market"]
    best_m = max(mk, key=lambda k: mk[k]["roi_pct"])
    worst_m = min(mk, key=lambda k: mk[k]["roi_pct"])
    nm = {"ml": "Moneylines", "spread": "Spreads", "total": "Totals"}
    market_note = f"{nm[best_m]} did best and {nm[worst_m].lower()} did worst. Only a range that sits entirely above or below zero means much."
    clv_note = (f"Bets placed at the open beat the close {pct(o['clv_beat_pct'], False)} of the time, by about {o['avg_clv_pct']:.1f} percentage points "
                "on average with the bookmaker's cut removed. At −110 each side carries about 2.4 points of that cut, so a bet has to beat the close "
                "by more than that to make money over time. The Monday CLV report should count a bet as beating the close only when it beats it by more than the cut.")
    open_curve = curve(RES / "bets_playbook_open.csv")
    close_curve = curve(RES / "bets_playbook_close.csv")
    lib_pb = pd.read_csv(RES / "library_backtest_playbook.csv")
    lib_lr = pd.read_csv(RES / "library_backtest_logistic.csv")

    names = {"ml": "Moneyline", "spread": "Spread", "total": "Total"}
    mk_rows = [{"name": names[k], **v} for k, v in p["by_market"].items()]
    season_names = {"2024": "2023-24", "2025": "2024-25", "2026": "2025-26", "2027": "2026-27 so far"}
    season_rows = [{"name": season_names.get(k, k), **v} for k, v in p["by_season"].items()]
    sweep_rows = [{"name": f'{r["threshold_pct"]:g}% edge', **r} for r in s["sweep"]
                  if r["blend"] == "always" and "market" not in r and r["threshold_pct"] > 0]
    tier_rows = [{"name": {"A": "A (6%+)", "B": "B (4–6%)", "C": "C (3–4%)"}[k], **v} for k, v in sorted(p["by_tier"].items())]

    def lib_table(df: pd.DataFrame) -> str:
        rows = "".join(
            f'<tr><td>{r["Testing start"]} → {r["Testing end"]}</td><td class="mono">{int(r["Number of bets"])}</td>'
            f'<td class="mono">{pct(r["Yield percentage per bet"])}</td><td class="mono">{"−" if r["Final cash"] < 0 else ""}${abs(r["Final cash"]):,.0f}</td></tr>'
            for _, r in df.iterrows())
        return ('<div class="tw"><table><thead><tr><th>Test window</th><th>Bets</th><th>Return per bet</th>'
                f'<th>Bankroll after ($2,500 start)</th></tr></thead><tbody>{rows}</tbody></table></div>')

    dc = p["data_checks"]
    data = json.dumps({"open": open_curve, "close": close_curve})
    html = TEMPLATE
    for k, v in {
        "__TITLE__": LEAGUE["title"], "__EYEBROW__": LEAGUE["eyebrow"], "__RULES__": LEAGUE["rules_name"],
        "__GAMESDESC__": LEAGUE["games_desc"], "__WARMUP__": LEAGUE["warmup"], "__SCREEN__": LEAGUE["screen"],
        "__CANTSEE__": "".join(f"<li>{escape(x)}</li>" for x in LEAGUE["cant_see"]),
        "__CHANGES__": "".join(f"<li>{escape(x)}</li>" for x in LEAGUE["changes"]),
        "__MODELNOTE__": LEAGUE["model_note"], "__SOURCES__": LEAGUE["sources"],
        "__VERDICT__": verdict, "__LUCK__": luck, "__WHYHEAD__": why_head, "__ACC__": acc, "__COVERLINE__": cover_line,
        "__SWEEPHEAD__": sweep_head, "__SWEEPNOTE__": sweep_note, "__MARKETNOTE__": market_note, "__CLVNOTE__": clv_note,
        "__GAMES__": f'{s["games_with_open_lines"]:,}',
        "__BETS__": f'{o["bets"]:,}',
        "__UNITS__": f'{o["units"]:+.1f}'.replace("-", "−"),
        "__USD__": money(o["units"]),
        "__ROI__": pct(o["roi_pct"]),
        "__CI_LO__": pct(o["roi_ci95"][0]), "__CI_HI__": pct(o["roi_ci95"][1]),
        "__WIN__": pct(o["win_pct"], False),
        "__ROI_CLOSE__": pct(oc["roi_pct"]), "__USD_CLOSE__": money(oc["units"]),
        "__CLV_BEAT__": pct(o["clv_beat_pct"], False), "__CLV_AVG__": f'{o["avg_clv_pct"]:.1f}',
        "__DD__": f'{o["max_drawdown_units"]:.0f}', "__STREAK__": str(o["longest_losing_streak"]),
        "__MKT_BARS__": bar_rows(mk_rows, "name"), "__SEASON_BARS__": bar_rows(season_rows, "name"),
        "__SWEEP_BARS__": bar_rows(sweep_rows, "name"), "__TIER_BARS__": bar_rows(tier_rows, "name"),
        "__MODEL_MAE__": f'{q["margin"]["model_mae"]:.1f}', "__MKT_MAE__": f'{q["margin"]["close_spread_mae"]:.1f}',
        "__TMODEL_MAE__": f'{q["total"]["model_mae"]:.1f}', "__TMKT_MAE__": f'{q["total"]["close_total_mae"]:.1f}',
        "__CORR__": f'{q["margin"]["corr_model_vs_close"]:.2f}',
        "__DC_BETS__": f'{dc["bets"]:,}', "__DC_ROI__": pct(dc["roi_pct"]),
        "__DC_LO__": pct(dc["roi_ci95"][0]), "__DC_HI__": pct(dc["roi_ci95"][1]),
        "__NOBLEND__": pct(s["variants"]["no_blend"]["roi_pct"]),
        "__ALLMK__": pct(s["variants"]["all_markets_per_game"]["roi_pct"]),
        "__ALLMK_N__": f'{s["variants"]["all_markets_per_game"]["bets"]:,}',
        "__LIB_PB__": lib_table(lib_pb), "__LIB_LR__": lib_table(lib_lr),
        "__DATA__": data,
    }.items():
        html = html.replace(k, v)
    (RES / "report.html").write_text(html)
    print("wrote", RES / "report.html")


TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap">
<style>
/* Layout: one reading column, verdict first, then evidence sections, matching the Daily Betting Board's palette */
:root{
  --bg:#f6f8fa; --panel:#ffffff; --line:#d8dee4; --text:#1f2328; --muted:#5b6672;
  --good:#1a7f37; --bad:#cf222e; --accent:#0969da; --warn:#9a6700; --track:#eaeef2;
  --sans:Inter,system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0d1117; --panel:#161b22; --line:#2a3340; --text:#e6edf3; --muted:#8b98a7;
  --good:#3fb950; --bad:#f85149; --accent:#58a6ff; --warn:#d29922; --track:#232b36; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0d1117; --panel:#161b22; --line:#2a3340; --text:#e6edf3; --muted:#8b98a7;
  --good:#3fb950; --bad:#f85149; --accent:#58a6ff; --warn:#d29922; --track:#232b36; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font:15px/1.55 var(--sans)}
.wrap{max-width:880px;margin:0 auto;padding-block:28px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:28px}
h1{font-size:28px;font-weight:800;letter-spacing:-.02em;margin:0;text-wrap:balance}
h2{font-size:18px;font-weight:700;margin:0;text-wrap:balance}
p{margin:0;max-width:68ch}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.muted{color:var(--muted)}
.verdict{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--bad);border-radius:10px;padding:18px 20px;display:flex;flex-direction:column;gap:10px}
.verdict .big{font-size:20px;font-weight:700;text-wrap:balance}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
@media (max-width:700px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
.stat{display:flex;flex-direction:column;gap:2px;padding:12px 0;border-top:2px solid var(--line)}
.stat b{font:700 22px var(--mono);font-variant-numeric:tabular-nums}
.stat span{font-size:13px;color:var(--muted)}
.neg{color:var(--bad)} .pos{color:var(--good)}
section{display:flex;flex-direction:column;gap:12px}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
.chart svg{display:block;width:100%;height:auto;min-width:520px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
.bars{display:flex;flex-direction:column;gap:10px}
.brow{display:grid;grid-template-columns:120px minmax(0,1fr) 70px;grid-template-areas:"l t v" ". m m";column-gap:12px;align-items:center}
.blabel{grid-area:l;font-weight:600;font-size:14px}
.btrack{grid-area:t;position:relative;height:14px;background:var(--track);border-radius:3px}
.bzero{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--muted)}
.bbar{position:absolute;top:2px;bottom:2px;border-radius:2px}
.bbar.neg{background:var(--bad)} .bbar.pos{background:var(--good)}
.bval{grid-area:v;text-align:right;font-size:14px}
.bmeta{grid-area:m;font-size:12px;color:var(--muted)}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:600;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:8px;border-bottom:1px solid var(--line)}
ul{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px;max-width:70ch}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media (max-width:700px){.cols{grid-template-columns:1fr}}
.cols>div{min-width:0;display:flex;flex-direction:column;gap:8px}
h3{font-size:14px;margin:0;font-weight:700}
.note{font-size:13px;color:var(--muted)}
code{font-family:var(--mono);font-size:13px;background:var(--track);padding:1px 5px;border-radius:4px}
</style>

<div class="wrap">
  <header style="display:flex;flex-direction:column;gap:8px">
    <div class="eyebrow">Daily Betting Board · __EYEBROW__ · backtest</div>
    <h1>Would the playbook's __RULES__ rules have made money?</h1>
    <p class="muted">__GAMES__ __GAMESDESC__, the 2023-24 season to last night. Ratings rebuilt from box scores, using only what was known before tip-off. __WARMUP__ __SCREEN__</p>
  </header>

  <div class="verdict">
    <div class="big">__VERDICT__</div>
    <p>They would have placed __BETS__ bets (one per game, flat $25) and finished <b>__USD__</b> (__UNITS__ units). __LUCK__ The model wins __WIN__ of the time, and at normal −110 prices you need about 52.4% to break even.</p>
  </div>

  <div class="stats">
    <div class="stat"><b class="neg">__ROI__</b><span>return per bet, betting the opening line</span></div>
    <div class="stat"><b class="neg">__ROI_CLOSE__</b><span>return per bet at the closing line (__USD_CLOSE__)</span></div>
    <div class="stat"><b>__CLV_BEAT__</b><span>of bets beat the closing line, by __CLV_AVG__ points on average</span></div>
    <div class="stat"><b>__DD__ u</b><span>worst drop from a peak; longest losing run __STREAK__</span></div>
  </div>

  <section>
    <h2>Bankroll over time, 1 unit per bet</h2>
    <div class="chart"><svg id="curve" viewBox="0 0 800 300" role="img" aria-label="Cumulative units won or lost over time"></svg></div>
    <div class="legend"><span><i style="background:var(--accent)"></i>Bet at the opening line</span><span><i style="background:var(--warn)"></i>Bet at the closing line</span></div>
    <p class="note">The board bets around 5pm, which is somewhere between these two lines.</p>
  </section>

  <section>
    <h2>__WHYHEAD__</h2>
    <p>The model's projections track the closing line closely (correlation __CORR__), so it's a faithful copy of the playbook method. __ACC__</p>
    <ul>
      <li>Game margin: the model misses by <b>__MODEL_MAE__ points</b> on average. The closing spread misses by <b>__MKT_MAE__</b>.</li>
      <li>Game total: the model misses by <b>__TMODEL_MAE__</b>. The closing total misses by <b>__TMKT_MAE__</b>.</li>
      <li>__COVERLINE__</li>
    </ul>
  </section>

  <section>
    <h2>__SWEEPHEAD__</h2>
    <p class="note">__SWEEPNOTE__</p>
    <div class="bars">__SWEEP_BARS__</div>
  </section>

  <div class="cols">
    <section><h2>By market</h2><div class="bars">__MKT_BARS__</div>
      <p class="note">__MARKETNOTE__</p></section>
    <section><h2>By tier</h2><div class="bars">__TIER_BARS__</div>
      <p class="note">Tier A is 6%+ edge, B is 4–6%, C is 3–4%.</p></section>
  </div>

  <section>
    <h2>By season</h2>
    <div class="bars">__SEASON_BARS__</div>
  </section>

  <section>
    <h2>Closing line value</h2>
    <p>__CLVNOTE__</p>
  </section>

  <section>
    <h2>Other things tested</h2>
    <ul>
      <li><b>No 50/50 blend with the market:</b> __NOBLEND__ per bet.</li>
      <li><b>Every qualifying market in a game</b> instead of only the best one: __ALLMK_N__ bets, __ALLMK__ per bet.</li>
      <li><b>DATA CHECK edges (above 10%)</b>, which the rules never bet: __DC_BETS__ of them, __DC_ROI__ per bet if they had been bet (range __DC_LO__ to __DC_HI__).</li>
    </ul>
  </section>

  <section>
    <h2>Cross-check with the sports-betting library</h2>
    <p>The moneyline rules were also run through the library's own bettor and its <code>backtest()</code>, which walks forward through time in five windows and checks the moneyline results above with a second tool.</p>
    <h3>Playbook rules as a library bettor</h3>
    __LIB_PB__
    <h3>The library's learning model (a logistic regression trained on the projections, rest and price)</h3>
    __LIB_LR__
    <p class="note">This one learns its own weights from the same inputs, retrained before each window.</p>
  </section>

  <section>
    <h2>What this backtest can't see</h2>
    <p>This tests the numbers part of the playbook. These parts can't be replayed from box scores:</p>
    <ul>__CANTSEE__</ul>
  </section>

  <section>
    <h2>What I'd change on the board</h2>
    <ul>__CHANGES__</ul>
  </section>

  <p class="note">Sources: __SOURCES__ __MODELNOTE__ Rules: 3% edge, +2% EV, edges over 10% skipped, one bet per game. For research, not financial advice.</p>
</div>

<script>
const D = __DATA__;
(function(){
  const svg = document.getElementById("curve"); const W=800,H=300,L=56,R=52,T=14,B=34;
  const all = D.open.concat(D.close); const dates = [...new Set(all.map(d=>d[0]))].sort();
  const xi = Object.fromEntries(dates.map((d,i)=>[d,i]));
  const ys = all.map(d=>d[1]); let lo=Math.min(0,...ys), hi=Math.max(0,...ys);
  const step = (hi-lo)>200?100:50; lo=Math.floor(lo/step)*step; hi=Math.ceil(hi/step)*step;
  const x = i => L + i/(dates.length-1)*(W-L-R), y = v => T + (hi-v)/(hi-lo)*(H-T-B);
  const css = n => `var(--${n})`; let s="";
  for(let v=lo; v<=hi; v+=step){ s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="${css(v===0?'muted':'line')}" stroke-width="1"/>`;
    s+=`<text x="${L-8}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="${css('muted')}" font-family="JetBrains Mono,monospace">${v>0?'+':''}${v}u</text>`; }
  const seasons = [["2023-10-01","2023-24"],["2024-10-01","2024-25"],["2025-10-01","2025-26"],["2026-10-01","2026-27"]];
  for(const [start,label] of seasons){ const i = dates.findIndex(d=>d>=start); if(i<0) continue;
    s+=`<line x1="${x(i)}" x2="${x(i)}" y1="${T}" y2="${H-B}" stroke="${css('line')}" stroke-dasharray="3 4"/>`;
    s+=`<text x="${x(i)+4}" y="${H-B+18}" font-size="11" fill="${css('muted')}">${label}</text>`; }
  const path = arr => arr.map((d,k)=>(k?"L":"M")+x(xi[d[0]]).toFixed(1)+" "+y(d[1]).toFixed(1)).join(" ");
  s+=`<path d="${path(D.close)}" fill="none" stroke="${css('warn')}" stroke-width="2"/>`;
  s+=`<path d="${path(D.open)}" fill="none" stroke="${css('accent')}" stroke-width="2"/>`;
  for(const [arr,c] of [[D.open,'accent'],[D.close,'warn']]){ const e=arr[arr.length-1];
    s+=`<circle cx="${x(xi[e[0]])}" cy="${y(e[1])}" r="3.5" fill="${css(c)}"/>`;
    s+=`<text x="${x(xi[e[0]])-6}" y="${y(e[1])-8}" text-anchor="end" font-size="11" fill="${css(c)}" font-family="JetBrains Mono,monospace">${e[1].toFixed(0)}u</text>`; }
  svg.innerHTML = s;
})();
</script>
"""

if __name__ == "__main__":
    main()
