"""Collect games, box scores and DraftKings open/close lines from ESPN.

  python fetch_data.py                 men's college basketball (all of Division I), seasons 2024-2027
  python fetch_data.py 2023            one season (2023 = 2022-23, used for last-season priors)

Writes one compact record per game. Re-running only fetches games not already saved,
so it doubles as the daily/weekly updater.
"""
import calendar, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

CBB = True  # this repository is college basketball only
LEAGUE = "mens-college-basketball" if CBB else "nba"
SB = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{LEAGUE}/scoreboard?dates={{day}}" + (
    "&groups=50&limit=500" if CBB else "")
SUM = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{LEAGUE}/summary?event={{id}}"
ODDS = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/{LEAGUE}/events/{{id}}/competitions/{{id}}/odds"
OUT = Path(__file__).parent / "data" / "games.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
SEASONS = [int(a) for a in sys.argv[1:] if a.isdigit()] or [2024, 2025, 2026, 2027]  # named by the year it ends
WINDOW = ((11, 1), (4, 15)) if CBB else ((10, 1), (6, 30))  # season start (previous year), season end


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nba-backtest"}), timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _day(d, season):
    out = []
    for ev in get(SB.format(day=d.strftime("%Y%m%d"))).get("events", []):
        stype = ev.get("season", {}).get("type")
        comp = ev["competitions"][0]
        if stype not in (2, 3) or comp.get("type", {}).get("abbreviation") == "ALLSTAR":
            continue
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        side = {c["homeAway"]: c for c in comp["competitors"]}
        if CBB:  # team ids are the stable key in college; names and conferences ride along
            out.append({
                "id": ev["id"], "date": ev["date"], "season": season, "stype": stype,
                "neutral": bool(comp.get("neutralSite")),
                "home": side["home"]["team"]["id"], "away": side["away"]["team"]["id"],
                "home_name": side["home"]["team"].get("displayName"), "away_name": side["away"]["team"].get("displayName"),
                "home_conf": side["home"]["team"].get("conferenceId"), "away_conf": side["away"]["team"].get("conferenceId"),
                "home_pts": int(side["home"]["score"]), "away_pts": int(side["away"]["score"]),
                "periods": comp.get("status", {}).get("period", 2),
            })
            continue
        out.append({
            "id": ev["id"], "date": ev["date"], "season": season, "stype": stype,
            "neutral": bool(comp.get("neutralSite")),
            "home": side["home"]["team"]["abbreviation"], "away": side["away"]["team"]["abbreviation"],
            "home_pts": int(side["home"]["score"]), "away_pts": int(side["away"]["score"]),
            "periods": comp.get("status", {}).get("period", 4),
        })
    return out


def schedule(season):
    (sm, sd), (em, ed) = WINDOW
    d, end = date(season - 1, sm, sd), min(date(season, em, ed), date.today() - timedelta(days=1))
    days = []
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    games = []
    with ThreadPoolExecutor(12) as ex:
        for res in ex.map(lambda x: _day(x, season), days):
            games += res
    return games


def am(x):
    """American odds from an ESPN price object or number."""
    if x is None:
        return None
    if isinstance(x, dict):
        x = x.get("american") or x.get("alternateDisplayValue")
    try:
        return float(str(x).replace("+", "")) if str(x) not in ("", "EVEN", "OFF") else (100.0 if str(x) == "EVEN" else None)
    except ValueError:
        return None


def pt(x):
    if isinstance(x, dict):
        x = x.get("american") or x.get("alternateDisplayValue")
    try:
        return float(str(x).replace("+", "")) if x not in (None, "", "PK") else (0.0 if x == "PK" else None)
    except ValueError:
        return None


def detail(g):
    s = get(SUM.format(id=g["id"]))
    box = {}
    for t in s.get("boxscore", {}).get("teams", []):
        st = {x["name"]: x["displayValue"] for x in t.get("statistics", [])}
        try:
            fgm, fga = map(int, st["fieldGoalsMade-fieldGoalsAttempted"].split("-"))
            ftm, fta = map(int, st["freeThrowsMade-freeThrowsAttempted"].split("-"))
            tpm, tpa = map(int, st["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].split("-"))
            box[t["homeAway"]] = {"fga": fga, "fgm": fgm, "fta": fta, "ftm": ftm, "tpa": tpa, "tpm": tpm,
                                  "oreb": int(st["offensiveRebounds"]), "dreb": int(st["defensiveRebounds"]),
                                  "tov": int(st.get("totalTurnovers") or st.get("turnovers"))}
        except (KeyError, ValueError):
            pass
    g["box"] = box if len(box) == 2 else None
    g["odds"] = None
    try:
        items = get(ODDS.format(id=g["id"])).get("items", [])
    except Exception:  # noqa: BLE001
        items = []
    items.sort(key=lambda it: it.get("provider", {}).get("priority", 99))
    cands = []
    for it in items:
        h, a = it.get("homeTeamOdds", {}), it.get("awayTeamOdds", {})
        o = {"book": it.get("provider", {}).get("name")}
        for when in ("open", "close"):
            ho, ao, to = h.get(when, {}), a.get(when, {}), it.get(when, {})
            o[when] = {
                "ml_h": am(ho.get("moneyLine")), "ml_a": am(ao.get("moneyLine")),
                "sp_h": pt(ho.get("pointSpread")), "sp_h_px": am(ho.get("spread")), "sp_a_px": am(ao.get("spread")),
                "tot": pt(to.get("total")), "ov_px": am(to.get("over")), "un_px": am(to.get("under")),
            }
        c = o["close"]  # fill close from the headline fields if the close block is empty
        c["ml_h"] = c["ml_h"] if c["ml_h"] is not None else am(h.get("moneyLine"))
        c["ml_a"] = c["ml_a"] if c["ml_a"] is not None else am(a.get("moneyLine"))
        c["sp_h"] = c["sp_h"] if c["sp_h"] is not None else pt(it.get("spread"))
        c["tot"] = c["tot"] if c["tot"] is not None else pt(it.get("overUnder"))
        has_open = o["open"]["ml_h"] is not None or o["open"]["sp_h"] is not None
        if has_open or c["ml_h"] is not None:
            live = "live" in str(o["book"]).lower()
            cands.append(((live, not has_open), "draft" not in str(o["book"]).lower().replace(" ", ""), len(cands), o))
    if cands:
        g["odds"] = min(cands, key=lambda t: t[:3])[3]
    return g


def main():
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.open()}
    todo = []
    for s in SEASONS:
        sch = schedule(s)
        new = [g for g in sch if g["id"] not in done]
        print(f"season {s}: {len(sch)} completed games, {len(new)} new", flush=True)
        todo += new
    n = 0
    with ThreadPoolExecutor(12) as ex, OUT.open("a") as f:
        futs = [ex.submit(detail, g) for g in todo]
        for fu in as_completed(futs):
            try:
                f.write(json.dumps(fu.result()) + "\n")
            except Exception as e:  # noqa: BLE001
                print("skip:", e, file=sys.stderr)
            n += 1
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print("saved", n, "games ->", OUT)


if __name__ == "__main__":
    main()
