#!/usr/bin/env python3
"""
Football Prophet Predictor Bot
==============================
Fetches upcoming fixtures from football-data.org for major leagues,
builds a Poisson / Dixon-Coles model from team form, calculates win
probabilities, Over/Under 2.5, BTTS, and SportyBet/Stake parameters,
generates visual match preview cards, and sends picks to Telegram.

Also tracks prediction history and automatically resolves finished
matches to maintain empirical accuracy & hit-rate stats.
"""

import os
import sys
import json
import math
import time
import random
import argparse
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# Fix emoji printing on Windows terminals
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load secrets from .env or environment variables
load_dotenv()

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
TELEGRAM_BOT_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")

_chat_ids_env = os.environ.get("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [c.strip() for c in _chat_ids_env.split(",") if c.strip()] if _chat_ids_env else []

COMPETITIONS = {
    "WC":  "FIFA World Cup",
    "PL":  "Premier League",
    "PD":  "La Liga",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
    "CL":  "Champions League",
}

DAYS_AHEAD       = int(os.environ.get("DAYS_AHEAD", 7))  # configurable via GitHub Actions Variable
LOOKBACK_MATCHES = 10
MAX_GOALS        = 6
FORM_DECAY_RATE  = 0.005

LEAGUE_AVG_GOALS = {
    "WC":  1.35,
    "PL":  1.40,
    "PD":  1.35,
    "BL1": 1.55,
    "SA":  1.30,
    "FL1": 1.35,
    "CL":  1.45,
}

# Files managed & committed back by GitHub Actions on every run
SENT_FIXTURES_FILE           = "sent_fixtures.json"
SENT_FIXTURES_RETENTION_DAYS = 14
PREDICTION_HISTORY_FILE      = "prediction_history.json"
MATCH_CARDS_DIR              = "match_cards"

# Telegram caption hard limit is 1024 characters
TELEGRAM_CAPTION_LIMIT = 1000

API_BASE    = "https://api.football-data.org/v4"
HEADERS     = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
API_DELAY_S = 6   # free tier: ~10 req/min — 6s keeps us safely under


# ----------------------------------------------------------------------
# Season Hype Announcements — Randomized Dictionary
# ----------------------------------------------------------------------
HYPE_HEADER_PREFIXES = [
    "🚨 CLUB FOOTBALL IS BACK!",
    "🔥 THE WAIT IS OVER!",
    "⚡ COUNTDOWN ALERT!",
    "🏆 SEASON KICKOFF INCOMING!",
    "🎉 FOOTBALL RETURNS!",
    "💪 IT'S GAME TIME!",
    "🔔 MARK YOUR CALENDARS!",
    "🎊 THE BEAUTIFUL GAME RETURNS!",
]

HYPE_ACTION_SLOGANS = [
    "{comp} let's go! 🚀",
    "Time to cook — data-driven picks are coming! 📊",
    "Lock in your picks and let's get this bag! 💰",
    "No vibes, pure statistical firepower! 🔮",
    "Get your betslips ready — we are back in action! 🎯",
    "The journey starts now — follow every prediction! 🏆",
    "{comp} is back and so are we! 🔥",
    "New season, new opportunities. Let's cook! 📈",
]

HYPE_COUNTDOWN_PHRASES = {
    4: "Only 4 DAYS left until kickoff!",
    3: "Just 3 DAYS to go before Matchday 1!",
    2: "2 DAYS AWAY! Get ready!",
    1: "TOMORROW IS THE DAY! ⚽",
    0: "MATCHDAY 1 IS TODAY! LET'S GO! 🔥",
}

# Common long team name overrides for cleaner display
TEAM_NAME_OVERRIDES = {
    "Rayo Vallecano de Madrid":         "Rayo Vallecano",
    "Real Racing Club de Santander":     "Racing Santander",
    "Deportivo Alavés":                  "Alavés",
    "Tottenham Hotspur FC":              "Spurs",
    "Manchester United FC":              "Man Utd",
    "Manchester City FC":                "Man City",
    "AFC Bournemouth":                   "Bournemouth",
    "Wolverhampton Wanderers FC":        "Wolves",
    "Brighton & Hove Albion FC":         "Brighton",
    "Nottingham Forest FC":              "Nott'm Forest",
    "West Bromwich Albion FC":           "West Brom",
    "Queens Park Rangers FC":            "QPR",
    "Blackburn Rovers FC":               "Blackburn",
    "Club Atlético de Madrid":           "Atlético Madrid",
    "Paris Saint-Germain FC":            "PSG",
    "Club Brugge KV":                    "Club Brugge",
    "Real Sociedad de Fútbol":           "Real Sociedad",
    "Villarreal CF":                     "Villarreal",
}


# ----------------------------------------------------------------------
# Helper Utilities
# ----------------------------------------------------------------------
def clean_team_name(name: str) -> str:
    """Simplify team names for readability — checks overrides first, then strips suffixes."""
    if not name:
        return ""
    if name in TEAM_NAME_OVERRIDES:
        return TEAM_NAME_OVERRIDES[name]
    for token in [" FC", " AFC", " CF", " SD", " UD", " CD", " SC", " SV", " AC", " SSV", " AS", " OSG", " FK", " SK"]:
        if name.endswith(token):
            name = name[:-len(token)]
    return name.strip()


def clean_text_for_image(text: str) -> str:
    """Strip emoji so PIL can render text cleanly without glyph boxes."""
    for ch in [
        "🏆", "⚽", "📅", "⚔️", "🎯", "🔥", "⚡", "⚠️", "⭐", "💰",
        "🏷️", "📊", "📈", "💡", "🥅", "🔮", "🚨", "🎉", "💪", "🔔",
        "🎊", "🚀", "🔔", "🎊", "⏳",
    ]:
        text = text.replace(ch, "")
    return text.strip()


def truncate_caption(text: str, limit: int = TELEGRAM_CAPTION_LIMIT) -> str:
    """Ensure Telegram captions stay within the 1024-char API limit."""
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


# ----------------------------------------------------------------------
# Data Fetching — Resilient API Client
# ----------------------------------------------------------------------
def api_get(path: str, params: dict = None) -> dict:
    """Fetch a football-data.org endpoint with 3 retries and rate-limit awareness."""
    url = f"{API_BASE}{path}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                print("[warn] Rate limited — waiting 12s before retry...", file=sys.stderr)
                time.sleep(12)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                raise
            print(f"[warn] API request failed (attempt {attempt + 1}/3): {e}", file=sys.stderr)
            time.sleep(3)
    return {}


def get_upcoming_fixtures(days_ahead: int = None) -> list:
    if days_ahead is None:
        days_ahead = DAYS_AHEAD
    now       = datetime.now(timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to   = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

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
                if None in (home_id, home_name, away_id, away_name):
                    continue
                fixtures.append({
                    "id":               m["id"],
                    "competition":      name,
                    "competition_code": code,
                    "utc_date":         m["utcDate"],
                    "home_id":          home_id,
                    "home_name":        home_name,
                    "away_id":          away_id,
                    "away_name":        away_name,
                })
            time.sleep(API_DELAY_S)
        except Exception as e:
            print(f"[warn] could not fetch {name}: {e}", file=sys.stderr)
    return fixtures


def get_team_recent_matches(team_id: int, limit: int = LOOKBACK_MATCHES) -> list:
    """Fetch team's finished matches over the past 365 days across seasons."""
    now       = datetime.now(timezone.utc)
    date_to   = now.strftime("%Y-%m-%d")
    date_from = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    try:
        data = api_get(
            f"/teams/{team_id}/matches",
            params={"status": "FINISHED", "dateFrom": date_from, "dateTo": date_to, "limit": limit},
        )
        time.sleep(API_DELAY_S)
        matches = data.get("matches", [])
        if not matches:
            print(f"[warn] No recent matches found for team {team_id} — using fallback stats.", file=sys.stderr)
        return sorted(matches, key=lambda m: m.get("utcDate", ""), reverse=True)[:limit]
    except Exception as e:
        print(f"[warn] could not fetch team {team_id} history: {e}", file=sys.stderr)
        return []


# ----------------------------------------------------------------------
# Duplicate & State Persistence
# ----------------------------------------------------------------------
def load_sent_fixtures() -> dict:
    """
    Load the sent-fixtures log. Structure:
      {
        "fixtures": {"2026-07-23": [fixture_id, ...], ...},
        "announcements": {"announce_PL_2026-08-21_d4": true, ...}
      }
    Prunes fixture entries older than SENT_FIXTURES_RETENTION_DAYS.
    """
    if not os.path.exists(SENT_FIXTURES_FILE):
        return {"fixtures": {}, "announcements": {}}
    try:
        with open(SENT_FIXTURES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"fixtures": {}, "announcements": {}}

    # Support old flat format (migrate automatically)
    if not isinstance(data, dict) or "fixtures" not in data:
        data = {"fixtures": data, "announcements": {}}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=SENT_FIXTURES_RETENTION_DAYS)).date()
    data["fixtures"] = {
        date: ids for date, ids in data.get("fixtures", {}).items()
        if datetime.fromisoformat(date).date() >= cutoff
    }
    data.setdefault("announcements", {})
    return data


def save_sent_fixtures(data: dict) -> None:
    with open(SENT_FIXTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_fixture_sent(fixture_id: int, sent_data: dict) -> bool:
    """Check if a fixture was already sent within the past 7 days (handles midnight crossings)."""
    now = datetime.now(timezone.utc)
    for days_back in range(7):
        date_key = (now - timedelta(days=days_back)).date().isoformat()
        if fixture_id in sent_data.get("fixtures", {}).get(date_key, []):
            return True
    return False


def mark_fixture_sent(fixture_id: int, sent_data: dict) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    sent_data["fixtures"].setdefault(today, [])
    if fixture_id not in sent_data["fixtures"][today]:
        sent_data["fixtures"][today].append(fixture_id)


def is_announcement_sent(key: str, sent_data: dict) -> bool:
    return sent_data.get("announcements", {}).get(key, False)


def mark_announcement_sent(key: str, sent_data: dict) -> None:
    sent_data.setdefault("announcements", {})[key] = True


# ----------------------------------------------------------------------
# Prediction History & Accuracy Resolution
# ----------------------------------------------------------------------
def load_prediction_history() -> list:
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return []
    try:
        with open(PREDICTION_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_prediction_history(history: list) -> None:
    with open(PREDICTION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def record_prediction(fixture: dict, prediction: dict, history: list) -> None:
    for item in history:
        if item.get("fixture_id") == fixture["id"]:
            return  # already recorded
    history.append({
        "fixture_id":             fixture["id"],
        "competition":            fixture["competition"],
        "competition_code":       fixture.get("competition_code", ""),
        "home_name":              fixture["home_name"],
        "away_name":              fixture["away_name"],
        "utc_date":               fixture["utc_date"],
        "predicted_pick":         prediction["pick_outcome"],
        "pick_name":              prediction["pick_name"],
        "pick_prob":              prediction["pick_prob"],
        "confidence_level":       prediction["confidence_level"],
        "stake_units":            prediction["stake_units"],
        "fair_odds":              prediction["fair_odds"],
        "predicted_score":        list(prediction["score"]),
        "over_2_5_prob":          prediction["over_2_5_prob"],
        "btts_yes_prob":          prediction["btts_yes_prob"],
        "created_at":             datetime.now(timezone.utc).isoformat(),
        "status":                 "PENDING",
        "actual_score":           None,
        "actual_outcome":         None,
        "is_outcome_correct":     None,
        "is_exact_score_correct": None,
    })


def resolve_past_predictions(history: list) -> dict:
    """
    Checks all PENDING predictions whose match time + 3h has passed,
    fetches actual results from the API, and updates the history in-place.
    Returns accuracy summary stats.
    """
    pending = [item for item in history if item.get("status") == "PENDING"]
    now = datetime.now(timezone.utc)
    resolved_new = 0

    for item in pending:
        try:
            match_dt = datetime.fromisoformat(item["utc_date"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if now < match_dt + timedelta(hours=3):
            continue

        try:
            match_data = api_get(f"/matches/{item['fixture_id']}")
            time.sleep(1)  # lighter delay for resolution calls
            status = match_data.get("status", "")
            if status in ("FINISHED", "AWARDED"):
                score = match_data.get("score", {}).get("fullTime", {})
                hg, ag = score.get("home"), score.get("away")
                if hg is not None and ag is not None:
                    actual_outcome = "HOME_WIN" if hg > ag else ("AWAY_WIN" if ag > hg else "DRAW")
                    item["actual_score"]           = [hg, ag]
                    item["actual_outcome"]         = actual_outcome
                    item["is_outcome_correct"]     = (item["predicted_pick"] == actual_outcome)
                    item["is_exact_score_correct"] = (item["predicted_score"] == [hg, ag])
                    item["status"]                 = "RESOLVED"
                    resolved_new                  += 1
        except Exception as e:
            print(f"[warn] could not resolve fixture {item['fixture_id']}: {e}", file=sys.stderr)

    resolved  = [item for item in history if item.get("status") == "RESOLVED"]
    total     = len(resolved)
    if total == 0:
        return {"total": 0, "resolved_new": resolved_new}

    hits       = sum(1 for i in resolved if i.get("is_outcome_correct"))
    exact_hits = sum(1 for i in resolved if i.get("is_exact_score_correct"))

    high_conf = [i for i in resolved if i.get("confidence_level") == "High"]
    high_hits = sum(1 for i in high_conf if i.get("is_outcome_correct"))

    med_conf  = [i for i in resolved if i.get("confidence_level") == "Medium"]
    med_hits  = sum(1 for i in med_conf if i.get("is_outcome_correct"))

    return {
        "total":         total,
        "resolved_new":  resolved_new,
        "hits":          hits,
        "hit_rate":      hits / total * 100,
        "exact_hits":    exact_hits,
        "high_total":    len(high_conf),
        "high_hits":     high_hits,
        "high_hit_rate": (high_hits / len(high_conf) * 100) if high_conf else 0.0,
        "med_total":     len(med_conf),
        "med_hits":      med_hits,
        "med_hit_rate":  (med_hits / len(med_conf) * 100) if med_conf else 0.0,
    }


# ----------------------------------------------------------------------
# Prediction Model & Analytics
# ----------------------------------------------------------------------
def team_goal_rates(team_id: int, matches: list) -> dict:
    team_id = int(team_id)
    now = datetime.now(timezone.utc)

    scored_home_w, conceded_home_w, home_w = 0.0, 0.0, 0.0
    scored_away_w, conceded_away_w, away_w = 0.0, 0.0, 0.0

    for m in matches:
        home_id = m.get("homeTeam", {}).get("id")
        if home_id is None:
            continue
        home = (int(home_id) == team_id)

        score = m.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            continue

        try:
            match_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            days_ago = max(0, (now - match_dt).days)
        except (ValueError, TypeError, KeyError):
            days_ago = 30
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
        # Fallback: league-neutral rates, log warning
        print(f"[warn] team {team_id}: no usable match data — using fallback stats.", file=sys.stderr)
        return {"attack_home": 1.3, "defense_home": 1.1, "attack_away": 1.1, "defense_away": 1.3}

    avg_scored_w   = (scored_home_w + scored_away_w)   / total_w
    avg_conceded_w = (conceded_home_w + conceded_away_w) / total_w

    return {
        "attack_home":  (scored_home_w   / home_w) if home_w else avg_scored_w,
        "defense_home": (conceded_home_w / home_w) if home_w else avg_conceded_w,
        "attack_away":  (scored_away_w   / away_w) if away_w else avg_scored_w,
        "defense_away": (conceded_away_w / away_w) if away_w else avg_conceded_w,
    }


def dixon_coles_adjustment(home_goals: int, away_goals: int, lh: float, la: float, rho: float = -0.1) -> float:
    if   home_goals == 0 and away_goals == 0: return 1 - (lh * la * rho)
    elif home_goals == 0 and away_goals == 1: return 1 + (lh * rho)
    elif home_goals == 1 and away_goals == 0: return 1 + (la * rho)
    elif home_goals == 1 and away_goals == 1: return 1 - rho
    return 1.0


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_match(home_stats: dict, away_stats: dict, competition_code: str = "", fixture: dict = None) -> dict:
    league_avg = LEAGUE_AVG_GOALS.get(competition_code, 1.35)

    lambda_home = max(0.3, min(home_stats["attack_home"] * (away_stats["defense_away"] / league_avg), 4.0))
    lambda_away = max(0.3, min(away_stats["attack_away"] * (home_stats["defense_home"] / league_avg), 4.0))

    grid = {}
    total_prob = 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = max(0.0,
                poisson_pmf(hg, lambda_home)
                * poisson_pmf(ag, lambda_away)
                * dixon_coles_adjustment(hg, ag, lambda_home, lambda_away)
            )
            grid[(hg, ag)] = p
            total_prob    += p

    for key in grid:
        grid[key] /= total_prob

    home_win = sum(p for (h, a), p in grid.items() if h > a)
    draw     = sum(p for (h, a), p in grid.items() if h == a)
    away_win = sum(p for (h, a), p in grid.items() if h < a)
    over_2_5 = sum(p for (h, a), p in grid.items() if (h + a) > 2.5)
    btts_yes = sum(p for (h, a), p in grid.items() if h >= 1 and a >= 1)

    best_score = max(grid, key=grid.get)

    home_name = clean_team_name(fixture["home_name"]) if fixture else "Home"
    away_name = clean_team_name(fixture["away_name"]) if fixture else "Away"

    # Pick Determination
    if home_win > away_win and home_win >= 0.40:
        pick_outcome, pick_name, pick_prob = "HOME_WIN", f"{home_name} Win", home_win
    elif away_win > home_win and away_win >= 0.40:
        pick_outcome, pick_name, pick_prob = "AWAY_WIN", f"{away_name} Win", away_win
    elif draw >= 0.35 and draw > home_win and draw > away_win:
        pick_outcome, pick_name, pick_prob = "DRAW", "Draw", draw
    elif home_win >= away_win:
        pick_outcome, pick_name, pick_prob = "HOME_WIN", f"{home_name} Win", home_win
    else:
        pick_outcome, pick_name, pick_prob = "AWAY_WIN", f"{away_name} Win", away_win

    # Confidence & Stake
    if pick_prob >= 0.52:
        confidence_level, confidence_label, stake_units, confidence_stars = "High",   "High",     "3/3 Units", "⭐⭐⭐"
    elif pick_prob >= 0.42:
        confidence_level, confidence_label, stake_units, confidence_stars = "Medium", "Moderate", "2/3 Units", "⭐⭐"
    else:
        confidence_level, confidence_label, stake_units, confidence_stars = "Low",    "Risky",    "1/3 Units", "⭐"

    return {
        "score":             best_score,
        "score_prob":        grid[best_score],
        "home_win_prob":     home_win,
        "draw_prob":         draw,
        "away_win_prob":     away_win,
        "over_2_5_prob":     over_2_5,
        "btts_yes_prob":     btts_yes,
        "pick_outcome":      pick_outcome,
        "pick_name":         pick_name,
        "pick_prob":         pick_prob,
        "confidence_level":  confidence_level,
        "confidence_label":  confidence_label,
        "confidence_stars":  confidence_stars,
        "stake_units":       stake_units,
        "fair_odds":         round(1.0 / max(0.01, pick_prob), 2),
        "lambda_home":       lambda_home,
        "lambda_away":       lambda_away,
    }


# ----------------------------------------------------------------------
# Pillow Image Card Generator
# ----------------------------------------------------------------------
def create_match_card_image(fixture: dict, prediction: dict) -> str | None:
    """Generates an 800×450 dark-mode match card PNG. Returns path or None on failure."""
    try:
        os.makedirs(MATCH_CARDS_DIR, exist_ok=True)
        file_path = os.path.join(MATCH_CARDS_DIR, f"card_{fixture['id']}.png")

        W, H = 800, 450
        img  = Image.new("RGB", (W, H), color="#0f172a")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Outer border
        draw.rectangle([10, 10, W - 10, H - 10], outline="#38bdf8", width=2)

        # Competition header
        comp_text = clean_text_for_image(fixture["competition"]).upper()
        draw.rectangle([30, 25, 320, 60], fill="#1e293b", outline="#3b82f6", width=1)
        draw.text((45, 36), comp_text, fill="#f8fafc", font=font)

        # Date
        dt = datetime.fromisoformat(fixture["utc_date"].replace("Z", "+00:00"))
        draw.text((W - 220, 36), dt.strftime("%a %d %b, %H:%M UTC"), fill="#94a3b8", font=font)

        # Teams
        home_clean = clean_team_name(fixture["home_name"])
        away_clean = clean_team_name(fixture["away_name"])
        draw.text((40, 90), home_clean, fill="#ffffff", font=font)
        draw.text((W // 2 - 15, 90), "VS", fill="#38bdf8", font=font)
        draw.text((W - 300, 90), away_clean, fill="#ffffff", font=font)

        # Pick box
        draw.rectangle([40, 135, W - 40, 240], fill="#1e1b4b", outline="#6366f1", width=2)
        draw.text((60, 155),
                  f"PICK: {clean_text_for_image(prediction['pick_name']).upper()}",
                  fill="#fbbf24", font=font)
        draw.text((60, 195),
                  f"Stake: {prediction['stake_units']}  |  Confidence: {prediction['confidence_level']}  |  Model Odds: @{prediction['fair_odds']:.2f}",
                  fill="#e2e8f0", font=font)

        # 1X2 probability bar
        draw.text((40, 260), "Win Chances:", fill="#94a3b8", font=font)
        bx, by, bw, bh = 40, 285, 720, 30
        hw = int(bw * prediction["home_win_prob"])
        dw = int(bw * prediction["draw_prob"])
        draw.rectangle([bx, by, bx + hw, by + bh], fill="#3b82f6")
        draw.rectangle([bx + hw, by, bx + hw + dw, by + bh], fill="#64748b")
        draw.rectangle([bx + hw + dw, by, bx + bw, by + bh], fill="#ef4444")
        draw.text((40, 325),
                  f"{home_clean} {prediction['home_win_prob']*100:.0f}%   |   Draw {prediction['draw_prob']*100:.0f}%   |   {away_clean} {prediction['away_win_prob']*100:.0f}%",
                  fill="#cbd5e1", font=font)

        # Market signals
        hg, ag = prediction["score"]
        btts_label = "Yes" if prediction["btts_yes_prob"] >= 0.5 else "No"
        draw.rectangle([40, 360, W - 40, 410], fill="#1e293b", outline="#334155")
        draw.text((60, 376),
                  f"Over 2.5 Goals: {prediction['over_2_5_prob']*100:.0f}%  |  BTTS: {btts_label} ({prediction['btts_yes_prob']*100:.0f}%)  |  Expected Score: {hg}-{ag}",
                  fill="#38bdf8", font=font)

        # Footer
        draw.text((W - 190, 420), "RYUU PREDICTION AI", fill="#64748b", font=font)

        img.save(file_path)
        return file_path

    except Exception as e:
        print(f"[warn] Card image generation failed: {e} — falling back to text.", file=sys.stderr)
        return None


# ----------------------------------------------------------------------
# Telegram Delivery
# ----------------------------------------------------------------------
def send_telegram_message(text: str) -> bool:
    url    = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    all_ok = True
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"[error] telegram message to {chat_id} failed: {resp.text}", file=sys.stderr)
                all_ok = False
        except Exception as e:
            print(f"[error] telegram message exception: {e}", file=sys.stderr)
            all_ok = False
        time.sleep(0.5)
    return all_ok


def send_telegram_photo(photo_path: str, caption: str) -> bool:
    """Send a photo card with truncated caption, falling back to text if photo unavailable."""
    if not photo_path or not os.path.exists(photo_path):
        return send_telegram_message(caption)

    safe_caption = truncate_caption(caption)
    url    = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    all_ok = True
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            with open(photo_path, "rb") as f:
                resp = requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": safe_caption, "parse_mode": "HTML"},
                    files={"photo": f},
                    timeout=30,
                )
            if resp.status_code != 200:
                print(f"[warn] photo send failed to {chat_id}, falling back to text.", file=sys.stderr)
                send_telegram_message(caption)
                all_ok = False
        except Exception as e:
            print(f"[error] photo upload failed: {e}", file=sys.stderr)
            send_telegram_message(caption)
            all_ok = False
        time.sleep(0.5)
    return all_ok


# ----------------------------------------------------------------------
# Message Formatting
# ----------------------------------------------------------------------
def format_prediction_message(fixture: dict, prediction: dict) -> str:
    dt       = datetime.fromisoformat(fixture["utc_date"].replace("Z", "+00:00"))
    date_str = dt.strftime("%a %d %b, %H:%M UTC")
    hg, ag   = prediction["score"]
    home     = clean_team_name(fixture["home_name"])
    away     = clean_team_name(fixture["away_name"])
    btts     = "Yes" if prediction["btts_yes_prob"] >= 0.50 else "No"

    return (
        f"⚽ <b>{fixture['competition']}</b>\n"
        f"📅 {date_str}\n"
        f"⚔️ <b>{home} vs {away}</b>\n\n"
        f"🔥 <b>PICK: {prediction['pick_name']}</b>\n"
        f"⭐ Confidence: <b>{prediction['confidence_label']}</b> ({prediction['stake_units']})\n"
        f"💡 Model Odds: <b>@{prediction['fair_odds']:.2f}</b>\n\n"
        f"📊 <b>Win Chances:</b>\n"
        f"  • {home}: <b>{prediction['home_win_prob']*100:.0f}%</b>\n"
        f"  • Draw: <b>{prediction['draw_prob']*100:.0f}%</b>\n"
        f"  • {away}: <b>{prediction['away_win_prob']*100:.0f}%</b>\n\n"
        f"⚽ Over 2.5 Goals: <b>{prediction['over_2_5_prob']*100:.0f}%</b>\n"
        f"🥅 Both Teams To Score: <b>{btts} ({prediction['btts_yes_prob']*100:.0f}%)</b>\n"
        f"🔮 Expected Score: <b>{hg} - {ag}</b>"
    )


def format_accuracy_summary(stats: dict) -> str:
    if stats.get("total", 0) == 0:
        return ""
    return (
        f"📊 <b>RYUU PREDICTION PERFORMANCE</b>\n\n"
        f"✅ Total Resolved: <b>{stats['total']}</b>\n"
        f"🎯 Overall Hit Rate: <b>{stats['hit_rate']:.1f}%</b> ({stats['hits']}/{stats['total']})\n"
        f"🔥 High Confidence: <b>{stats['high_hit_rate']:.1f}%</b> ({stats['high_hits']}/{stats['high_total']})\n"
        f"⚡ Moderate Confidence: <b>{stats['med_hit_rate']:.1f}%</b> ({stats['med_hits']}/{stats['med_total']})\n"
        f"🔮 Exact Score Hits: <b>{stats['exact_hits']}</b>"
    )


# ----------------------------------------------------------------------
# Season Countdown Hype Announcements
# ----------------------------------------------------------------------
def generate_season_hype_message(comp_name: str, days_left: int, first_match_date_str: str) -> str:
    header      = random.choice(HYPE_HEADER_PREFIXES)
    raw_slogan  = random.choice(HYPE_ACTION_SLOGANS)
    slogan      = raw_slogan.replace("{comp}", comp_name)
    time_phrase = HYPE_COUNTDOWN_PHRASES.get(days_left, f"Starts in {days_left} days!")
    return (
        f"<b>{header}</b>\n\n"
        f"⚽ <b>{comp_name}</b>\n"
        f"📅 Opening Match: <b>{first_match_date_str}</b>\n"
        f"⏳ Countdown: <b>{time_phrase}</b>\n\n"
        f"<i>{slogan}</i>"
    )


def check_and_send_season_announcements(fixtures: list, sent_data: dict) -> None:
    """
    For each competition in the upcoming fixtures, if the first match is
    1–4 days away, send a randomised hype announcement (once per day per competition).

    NOTE: We skip day 0 from the announcement here — on matchday itself the
    prediction message is the announcement. We also skip fixtures happening in
    the next 24 hours (they are regular weekly predictions, not season openers).
    Announcement keys are stored separately from fixture IDs to avoid type clashes.
    """
    if not fixtures:
        return

    by_comp: dict[str, list] = {}
    for f in fixtures:
        by_comp.setdefault(f.get("competition_code", "GENERIC"), []).append(f)

    now        = datetime.now(timezone.utc)
    today_date = now.date()

    for code, comp_fixtures in by_comp.items():
        comp_fixtures.sort(key=lambda x: x["utc_date"])
        earliest = comp_fixtures[0]
        try:
            match_dt   = datetime.fromisoformat(earliest["utc_date"].replace("Z", "+00:00"))
            match_date = match_dt.date()
            days_left  = (match_date - today_date).days
        except (ValueError, TypeError):
            continue

        # Only announce 1–4 days before the first fixture of a competition.
        # Skip 0 (matchday — predictions serve as the announcement)
        # Skip if the match is already today or tomorrow within the normal
        # prediction window (it will appear as a regular prediction card).
        if 1 <= days_left <= 4:
            ann_key = f"announce_{code}_{match_date.isoformat()}_d{days_left}"
            if not is_announcement_sent(ann_key, sent_data):
                date_formatted = match_dt.strftime("%a %d %b, %H:%M UTC")
                hype_msg = generate_season_hype_message(earliest["competition"], days_left, date_formatted)

                print("=" * 40)
                print(f"[Season Announcement] {earliest['competition']} — {days_left} day(s) to kickoff")
                print(hype_msg)
                print("=" * 40)

                if send_telegram_message(hype_msg):
                    mark_announcement_sent(ann_key, sent_data)
                time.sleep(1)


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def run_once(days_ahead: int = None) -> None:
    if days_ahead is None:
        days_ahead = DAYS_AHEAD

    if not FOOTBALL_DATA_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        print(
            "Missing configuration. Please set FOOTBALL_DATA_API_KEY, "
            "TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_IDS in your .env file."
        )
        return

    sent_data = load_sent_fixtures()
    history   = load_prediction_history()

    # Step 1 — Resolve finished matches and update hit-rate stats
    print("Checking and resolving past predictions...")
    acc_stats = resolve_past_predictions(history)
    save_prediction_history(history)
    if acc_stats.get("resolved_new", 0) > 0:
        print(f"Resolved {acc_stats['resolved_new']} newly finished match(es).")
        summary_msg = format_accuracy_summary(acc_stats)
        if summary_msg:
            send_telegram_message(summary_msg)

    # Step 2 — Fetch fixtures for PREDICTIONS (7-day window)
    print(f"Fetching upcoming fixtures (next {days_ahead} days)...")
    fixtures = get_upcoming_fixtures(days_ahead=days_ahead)

    # Always persist state here regardless of what happens next
    save_sent_fixtures(sent_data)

    # Step 3 — Season countdown announcements using a WIDER 30-day window
    # This lets us announce "PL starts in 4 days" even before that week's
    # fixtures appear in the 7-day prediction window.
    print("Checking for upcoming season start announcements (next 30 days)...")
    announcement_fixtures = get_upcoming_fixtures(days_ahead=30)
    check_and_send_season_announcements(announcement_fixtures, sent_data)
    save_sent_fixtures(sent_data)  # persist announcement keys immediately

    if not fixtures:
        print(f"No upcoming fixtures found in next {days_ahead} days.")
        return

    # Step 4 — Filter to only new fixtures
    new_fixtures = [f for f in fixtures if not is_fixture_sent(f["id"], sent_data)]
    skipped      = len(fixtures) - len(new_fixtures)
    if skipped:
        print(f"Skipping {skipped} fixture(s) already sent in the past 7 days.")

    if not new_fixtures:
        print("All fixtures already sent. Nothing new to send.")
        return

    print(f"Found {len(new_fixtures)} new fixture(s). Building predictions...")

    stats_cache: dict[int, dict] = {}
    for fixture in new_fixtures:
        for team_id in (fixture["home_id"], fixture["away_id"]):
            if team_id not in stats_cache:
                matches = get_team_recent_matches(team_id)
                stats_cache[team_id] = team_goal_rates(team_id, matches)

        home_stats = stats_cache[fixture["home_id"]]
        away_stats = stats_cache[fixture["away_id"]]
        prediction = predict_match(home_stats, away_stats, fixture["competition_code"], fixture)

        record_prediction(fixture, prediction, history)

        message    = format_prediction_message(fixture, prediction)
        card_image = create_match_card_image(fixture, prediction)  # guarded — returns None on failure

        print("-" * 40)
        print(message)

        if send_telegram_photo(card_image, message):
            mark_fixture_sent(fixture["id"], sent_data)
        time.sleep(1)

    # Final persist
    save_sent_fixtures(sent_data)
    save_prediction_history(history)
    print("Done. Predictions & history log updated.")


def loop_forever(interval_hours: float = 24, days_ahead: int = None) -> None:
    while True:
        try:
            run_once(days_ahead=days_ahead)
        except Exception as e:
            print(f"[error] run failed: {e}", file=sys.stderr)
        print(f"Sleeping {interval_hours}h until next run...")
        time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Prophet Predictor Bot")
    parser.add_argument("--loop",     action="store_true", help="run continuously")
    parser.add_argument("--interval", type=float, default=24, help="hours between runs (loop mode)")
    parser.add_argument("--days",     type=int,   default=None, help="fixture lookahead window in days")
    args = parser.parse_args()

    if args.loop:
        loop_forever(args.interval, days_ahead=args.days)
    else:
        run_once(days_ahead=args.days)
