#!/usr/bin/env python3
"""
ryuuprediction_bot
===================
Fetches upcoming fixtures from football-data.org for several major leagues,
builds a Poisson / Dixon-Coles style attack-defense model from each team's
recent results, predicts the most likely scorelines, and sends the
predictions to the Football Prophet Telegram group.

Setup
-----
1. Get a free API key: https://www.football-data.org/client/register
2. Create a Telegram bot via @BotFather, get the bot token.
3. Get your chat ID: message your bot once, then visit
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   and read the "chat":{"id": ...} field.
4. Copy .env.example to .env and fill in your values.
5. Run:  python predictor.py
   Or use GitHub Actions for fully automated daily runs (see .github/workflows/).

Dependencies:  pip install -r requirements.txt
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

# Fix emoji printing on Windows terminals (cp1252 → utf-8)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load secrets from .env (ignored by git) or from environment variables
# (set in GitHub Actions secrets / your shell).
load_dotenv()

# ----------------------------------------------------------------------
# CONFIG — set these in .env or as GitHub Actions secrets
# ----------------------------------------------------------------------
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
TELEGRAM_BOT_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")

_chat_ids_env = os.environ.get("TELEGRAM_CHAT_IDS", "")
if _chat_ids_env:
    TELEGRAM_CHAT_IDS = [c.strip() for c in _chat_ids_env.split(",") if c.strip()]
else:
    TELEGRAM_CHAT_IDS = []

# Competitions to track (football-data.org codes). Free tier friendly set.
COMPETITIONS = {
    "WC":  "FIFA World Cup",
    "PL":  "Premier League",
    "PD":  "La Liga",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
    "CL":  "Champions League",
}

DAYS_AHEAD       = 7    # look for fixtures in the next N days
LOOKBACK_MATCHES = 10  # how many recent matches to use per team for form
MAX_GOALS        = 6   # cap on scoreline grid (0..MAX_GOALS for each side)

# Time-decay rate for recent-form weighting.
# A match 30 days ago gets weight ~0.86, 90 days ago ~0.64, 180 days ago ~0.41.
FORM_DECAY_RATE = 0.005

# League-specific average goals per team per game (goals/match ÷ 2).
# Used to normalise each team's attack/defence rates against a realistic baseline.
LEAGUE_AVG_GOALS = {
    "WC":  1.35,  # FIFA World Cup      ~2.70 goals/match
    "PL":  1.40,  # Premier League      ~2.80 goals/match
    "PD":  1.35,  # La Liga             ~2.70 goals/match
    "BL1": 1.55,  # Bundesliga          ~3.10 goals/match
    "SA":  1.30,  # Serie A             ~2.60 goals/match
    "FL1": 1.35,  # Ligue 1             ~2.70 goals/match
    "CL":  1.45,  # Champions League    ~2.90 goals/match
}

# Fixture IDs sent today are stored here to prevent duplicate messages.
SENT_FIXTURES_FILE = "sent_fixtures.json"
# How many days of history to keep in the sent-fixtures log.
SENT_FIXTURES_RETENTION_DAYS = 14

API_BASE = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}


# ----------------------------------------------------------------------
# Duplicate-prevention helpers
# ----------------------------------------------------------------------
def load_sent_fixtures() -> dict:
    """
    Load the sent-fixtures log from disk. Prunes entries older than
    SENT_FIXTURES_RETENTION_DAYS to keep the file small.
    Returns a dict keyed by ISO date string, values are lists of fixture IDs.
    """
    if not os.path.exists(SENT_FIXTURES_FILE):
        return {}
    try:
        with open(SENT_FIXTURES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=SENT_FIXTURES_RETENTION_DAYS)).date()
    return {date: ids for date, ids in data.items() if datetime.fromisoformat(date).date() >= cutoff}


def save_sent_fixtures(data: dict) -> None:
    """Persist the sent-fixtures log to disk."""
    with open(SENT_FIXTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_already_sent(fixture_id: int, sent_data: dict) -> bool:
    """Return True if this fixture ID was already sent on today's date."""
    today = datetime.now(timezone.utc).date().isoformat()
    return fixture_id in sent_data.get(today, [])


def mark_as_sent(fixture_id: int, sent_data: dict) -> None:
    """Record a fixture ID as sent for today."""
    today = datetime.now(timezone.utc).date().isoformat()
    sent_data.setdefault(today, [])
    if fixture_id not in sent_data[today]:
        sent_data[today].append(fixture_id)


# ----------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------
def api_get(path, params=None):
    url  = f"{API_BASE}{path}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    if resp.status_code == 429:
        # rate limited — wait and retry once
        time.sleep(6)
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_upcoming_fixtures():
    """Return list of upcoming matches across all tracked competitions."""
    now       = datetime.now(timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to   = (now + timedelta(days=DAYS_AHEAD)).strftime("%Y-%m-%d")

    fixtures = []
    for code, name in COMPETITIONS.items():
        try:
            data = api_get(
                f"/competitions/{code}/matches",
                params={"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED"},
            )
            for m in data.get("matches", []):
                home_id   = m["homeTeam"]["id"]
                home_name = m["homeTeam"]["name"]
                away_id   = m["awayTeam"]["id"]
                away_name = m["awayTeam"]["name"]
                # Skip fixtures where teams haven't been decided yet
                # (knockout placeholder slots return null from the API)
                if None in (home_id, home_name, away_id, away_name):
                    continue
                fixtures.append(
                    {
                        "id":               m["id"],
                        "competition":      name,
                        "competition_code": code,
                        "utc_date":         m["utcDate"],
                        "home_id":          home_id,
                        "home_name":        home_name,
                        "away_id":          away_id,
                        "away_name":        away_name,
                    }
                )
            time.sleep(6)  # free tier: ~10 req/min, stay safe
        except requests.HTTPError as e:
            print(f"[warn] could not fetch {name}: {e}", file=sys.stderr)
    return fixtures


def get_team_recent_matches(team_id, limit=LOOKBACK_MATCHES):
    """Fetch a team's recent finished matches."""
    try:
        data = api_get(
            f"/teams/{team_id}/matches",
            params={"status": "FINISHED", "limit": limit},
        )
        time.sleep(6)
        return data.get("matches", [])
    except requests.HTTPError as e:
        print(f"[warn] could not fetch team {team_id} history: {e}", file=sys.stderr)
        return []


# ----------------------------------------------------------------------
# Prediction model
# ----------------------------------------------------------------------
def team_goal_rates(team_id, matches):
    """
    Compute a team's time-weighted average goals scored and conceded,
    split by home/away. Recent matches are weighted more heavily via
    exponential decay (FORM_DECAY_RATE). Falls back to the overall
    weighted average if not enough home/away-specific data exists.
    """
    now = datetime.now(timezone.utc)

    scored_home_w,   conceded_home_w,   home_w   = 0.0, 0.0, 0.0
    scored_away_w,   conceded_away_w,   away_w   = 0.0, 0.0, 0.0

    for m in matches:
        home  = m["homeTeam"]["id"] == team_id
        score = m.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            continue

        # Exponential time-decay: newer matches count more
        try:
            match_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            days_ago = max(0, (now - match_dt).days)
        except (ValueError, TypeError, KeyError):
            days_ago = 30  # safe fallback
        w = math.exp(-FORM_DECAY_RATE * days_ago)

        if home:
            scored_home_w   += hg * w
            conceded_home_w += ag * w
            home_w          += w
        else:
            scored_away_w   += ag * w
            conceded_away_w += hg * w
            away_w          += w

    total_w = home_w + away_w
    if total_w == 0:
        # no usable data — fall back to neutral league-average rates
        return {"attack_home": 1.3, "defense_home": 1.1, "attack_away": 1.1, "defense_away": 1.3}

    avg_scored_w   = (scored_home_w   + scored_away_w)   / total_w
    avg_conceded_w = (conceded_home_w + conceded_away_w) / total_w

    return {
        "attack_home":  (scored_home_w   / home_w) if home_w else avg_scored_w,
        "defense_home": (conceded_home_w / home_w) if home_w else avg_conceded_w,
        "attack_away":  (scored_away_w   / away_w) if away_w else avg_scored_w,
        "defense_away": (conceded_away_w / away_w) if away_w else avg_conceded_w,
    }


def dixon_coles_adjustment(home_goals, away_goals, lambda_home, lambda_away, rho=-0.1):
    """
    Low-score correlation adjustment from Dixon & Coles (1997) — corrects
    the plain independent-Poisson assumption for 0-0, 1-0, 0-1, 1-1
    scorelines, which real football has more/fewer of than pure Poisson predicts.
    """
    if   home_goals == 0 and away_goals == 0: return 1 - (lambda_home * lambda_away * rho)
    elif home_goals == 0 and away_goals == 1: return 1 + (lambda_home * rho)
    elif home_goals == 1 and away_goals == 0: return 1 + (lambda_away * rho)
    elif home_goals == 1 and away_goals == 1: return 1 - rho
    return 1.0


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_match(home_stats, away_stats, competition_code=""):
    """
    Expected goals: blend each side's attack rate with the opponent's
    defensive weakness. Then build the full scoreline probability grid
    with the Dixon-Coles correlation tweak, and read off the most
    likely scoreline plus win/draw/loss probabilities.

    Uses a league-specific average goals baseline (LEAGUE_AVG_GOALS) so
    the normalisation reflects how open/defensive that competition tends to be.
    """
    league_avg_goals = LEAGUE_AVG_GOALS.get(competition_code, 1.35)

    lambda_home = home_stats["attack_home"] * (away_stats["defense_away"] / league_avg_goals)
    lambda_away = away_stats["attack_away"] * (home_stats["defense_home"] / league_avg_goals)

    # keep it sane
    lambda_home = max(0.3, min(lambda_home, 4.0))
    lambda_away = max(0.3, min(lambda_away, 4.0))

    grid       = {}
    total_prob = 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p  = poisson_pmf(hg, lambda_home) * poisson_pmf(ag, lambda_away)
            p *= dixon_coles_adjustment(hg, ag, lambda_home, lambda_away)
            p  = max(p, 0)
            grid[(hg, ag)] = p
            total_prob    += p

    # normalize (Dixon-Coles tweak can slightly distort total mass)
    for key in grid:
        grid[key] /= total_prob

    best_score    = max(grid, key=grid.get)
    home_win      = sum(p for (h, a), p in grid.items() if h > a)
    draw          = sum(p for (h, a), p in grid.items() if h == a)
    away_win      = sum(p for (h, a), p in grid.items() if h < a)

    return {
        "score":          best_score,
        "score_prob":     grid[best_score],
        "home_win_prob":  home_win,
        "draw_prob":      draw,
        "away_win_prob":  away_win,
        "lambda_home":    lambda_home,
        "lambda_away":    lambda_away,
    }


# ----------------------------------------------------------------------
# Telegram delivery
# ----------------------------------------------------------------------
def send_telegram_message(text):
    """Send a message to every configured chat/group ID."""
    url    = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    all_ok = True
    for chat_id in TELEGRAM_CHAT_IDS:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"[error] telegram send to {chat_id} failed: {resp.text}", file=sys.stderr)
            all_ok = False
        time.sleep(0.5)  # be gentle with Telegram's rate limits
    return all_ok


def format_prediction_message(fixture, prediction):
    dt       = datetime.fromisoformat(fixture["utc_date"].replace("Z", "+00:00"))
    date_str = dt.strftime("%a %d %b, %H:%M UTC")
    hg, ag   = prediction["score"]

    # Upset alert: flag when the away team is a clear favourite
    upset_line = ""
    if prediction["away_win_prob"] > 0.60:
        upset_line = (
            f"\n⚠️ <b>UPSET PICK</b> — {fixture['away_name']} are "
            f"strong favourites away from home"
        )

    return (
        f"⚽ <b>{fixture['competition']}</b>\n"
        f"{date_str}\n"
        f"<b>{fixture['home_name']} vs {fixture['away_name']}</b>\n\n"
        f"🔮 Predicted score: <b>{hg} - {ag}</b> "
        f"({prediction['score_prob']*100:.1f}% likelihood)\n"
        f"📊 Win probabilities: "
        f"{fixture['home_name']} {prediction['home_win_prob']*100:.0f}% | "
        f"Draw {prediction['draw_prob']*100:.0f}% | "
        f"{fixture['away_name']} {prediction['away_win_prob']*100:.0f}%"
        f"{upset_line}"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_once():
    if not FOOTBALL_DATA_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        print(
            "Missing configuration. Please set FOOTBALL_DATA_API_KEY, "
            "TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_IDS in your .env file "
            "or as environment variables."
        )
        return

    # Load duplicate-prevention state
    sent_data = load_sent_fixtures()

    print("Fetching upcoming fixtures...")
    fixtures = get_upcoming_fixtures()
    if not fixtures:
        print("No upcoming fixtures found in the next", DAYS_AHEAD, "days.")
        return

    # Filter out fixtures already sent today
    new_fixtures = [f for f in fixtures if not is_already_sent(f["id"], sent_data)]
    skipped      = len(fixtures) - len(new_fixtures)
    if skipped:
        print(f"Skipping {skipped} fixture(s) already sent today.")

    if not new_fixtures:
        print("All fixtures already sent today. Nothing new to send.")
        save_sent_fixtures(sent_data)
        return

    print(f"Found {len(new_fixtures)} new fixture(s). Building predictions...")

    # Cache team stats so we don't refetch the same team twice in one run
    stats_cache = {}

    for fixture in new_fixtures:
        for team_id in (fixture["home_id"], fixture["away_id"]):
            if team_id not in stats_cache:
                matches = get_team_recent_matches(team_id)
                stats_cache[team_id] = team_goal_rates(team_id, matches)

        home_stats = stats_cache[fixture["home_id"]]
        away_stats = stats_cache[fixture["away_id"]]
        prediction = predict_match(home_stats, away_stats, fixture["competition_code"])

        message = format_prediction_message(fixture, prediction)
        print("-" * 40)
        print(message)
        if send_telegram_message(message):
            mark_as_sent(fixture["id"], sent_data)
        time.sleep(1)

    # Persist the updated sent-fixtures log
    save_sent_fixtures(sent_data)
    print("Done. Predictions sent to Telegram.")


def loop_forever(interval_hours=24):
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[error] run failed: {e}", file=sys.stderr)
        print(f"Sleeping {interval_hours}h until next run...")
        time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ryuuprediction_bot → Telegram football score predictor")
    parser.add_argument("--loop",     action="store_true", help="run continuously instead of once")
    parser.add_argument("--interval", type=float, default=24, help="hours between runs in loop mode")
    args = parser.parse_args()

    if args.loop:
        loop_forever(args.interval)
    else:
        run_once()
