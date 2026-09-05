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

DAYS_AHEAD       = int(os.environ.get("DAYS_AHEAD", 2))  # configurable via GitHub Actions Variable (default: 2 days)
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
# Team Name Normalization
# ----------------------------------------------------------------------

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
    "Stade Rennais FC 1901":             "Rennes",
    "Olympique de Marseille":            "Marseille",
    "Olympique Lyonnais":                "Lyon",
    "AS Monaco FC":                      "Monaco",
    "Lille OSC":                         "Lille",
    "OGC Nice":                          "Nice",
    "RC Lens":                           "Lens",
    "FC Bayern München":                 "Bayern Munich",
    "Borussia Dortmund":                 "Dortmund",
    "Bayer 04 Leverkusen":               "Leverkusen",
    "RB Leipzig":                        "Leipzig",
    "VfB Stuttgart":                     "Stuttgart",
    "Eintracht Frankfurt":               "Frankfurt",
    "FC Internazionale Milano":          "Inter Milan",
    "AC Milan":                          "AC Milan",
    "Juventus FC":                       "Juventus",
    "SSC Napoli":                        "Napoli",
    "AS Roma":                           "Roma",
    "SS Lazio":                          "Lazio",
    "Atalanta BC":                       "Atalanta",
    "Real Madrid CF":                    "Real Madrid",
    "FC Barcelona":                      "Barcelona",
    "Athletic Club":                     "Athletic Bilbao",
    "Real Betis Balompié":               "Real Betis",
    "Sevilla FC":                        "Sevilla",
    "Arsenal FC":                        "Arsenal",
    "Chelsea FC":                        "Chelsea",
    "Liverpool FC":                      "Liverpool",
    "Aston Villa FC":                    "Aston Villa",
    "Newcastle United FC":               "Newcastle",
    "West Ham United FC":                "West Ham",
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
    for token in [" FC", " AFC", " CF", " SD", " UD", " CD", " SC", " SV", " AC", " SSV", " AS", " OSG", " FK", " SK", " OSC"]:
        if name.endswith(token):
            name = name[:-len(token)]
    import re
    name = re.sub(r"\s+(?:18\d{2}|19\d{2}|20\d{2}|04)$", "", name)
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
                wait = 12 * (attempt + 1)
                print(f"[warn] Rate limited (attempt {attempt + 1}/3) — waiting {wait}s before retry...", file=sys.stderr)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"[warn] Unexpected HTTP {resp.status_code} for {url}: {resp.text[:200]}", file=sys.stderr)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                print(f"[error] API request permanently failed after 3 attempts for {url}: {e}", file=sys.stderr)
                raise
            print(f"[warn] API request failed (attempt {attempt + 1}/3): {e}", file=sys.stderr)
            time.sleep(3)
    return {}


def get_upcoming_fixtures(days_ahead: int = None) -> list:
    """Fetch SCHEDULED fixtures for all competitions within the next `days_ahead` days."""
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
            comp_matches = data.get("matches", [])
            print(f"[info] {name}: {len(comp_matches)} scheduled fixture(s) in window.", file=sys.stderr)
            for m in comp_matches:
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
            print(f"[error] could not fetch {name} fixtures: {e}", file=sys.stderr)
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
# ----------------------------------------------------------------------
# Pillow Image Card Generator -- Clean White Minimal (1200x630)
# ----------------------------------------------------------------------


def get_card_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Finds bundled or system TrueType font with graceful fallback to default font."""
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts", "Inter-SemiBold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_match_card_image(fixture: dict, prediction: dict) -> str | None:
    """Generates a clean, minimal white match preview card PNG (1200x630)."""
    try:
        os.makedirs(MATCH_CARDS_DIR, exist_ok=True)
        file_path = os.path.join(MATCH_CARDS_DIR, f"card_{fixture['id']}.png")

        W, H = 1200, 480

        # Palette
        BG       = "#ffffff"
        CARD_BG  = "#f9fafb"
        BORDER   = "#e5e7eb"
        INK      = "#111827"
        MUTED    = "#6b7280"
        LIGHT    = "#9ca3af"
        GREEN    = "#16a34a"
        RED      = "#dc2626"
        GREY_BAR = "#d1d5db"
        DIVIDER  = "#e5e7eb"

        img  = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, W - 1, H - 1], outline=BORDER, width=1)

        # -- 1. HEADER ------------------------------------------------
        hdr_h = 60
        draw.rectangle([0, 0, W, hdr_h], fill=CARD_BG)
        draw.line([(0, hdr_h), (W, hdr_h)], fill=DIVIDER, width=1)

        font_comp  = get_card_font(13, bold=True)
        font_label = get_card_font(12)
        font_date  = get_card_font(13)

        comp_text = clean_text_for_image(fixture.get("competition", "COMPETITION")).upper()
        draw.text((36, hdr_h // 2 - 8), comp_text, fill=INK, font=font_comp)

        brand_txt = "RYUU PREDICTION AI"
        brand_bb  = draw.textbbox((0, 0), brand_txt, font=font_label)
        draw.text((W // 2 - (brand_bb[2] - brand_bb[0]) // 2, hdr_h // 2 - 8), brand_txt, fill=MUTED, font=font_label)

        dt       = datetime.fromisoformat(fixture["utc_date"].replace("Z", "+00:00"))
        date_str = dt.strftime("%a %d %b  -  %H:%M UTC")
        dt_bb    = draw.textbbox((0, 0), date_str, font=font_date)
        draw.text((W - 36 - (dt_bb[2] - dt_bb[0]), hdr_h // 2 - 8), date_str, fill=MUTED, font=font_date)

        # -- 2. TEAMS -------------------------------------------------
        home_clean = clean_team_name(fixture["home_name"])
        away_clean = clean_team_name(fixture["away_name"])

        font_team = get_card_font(44, bold=True)
        font_role = get_card_font(12)
        for s in [44, 38, 32, 26]:
            font_team = get_card_font(s, bold=True)
            h_bb = draw.textbbox((0, 0), home_clean, font=font_team)
            a_bb = draw.textbbox((0, 0), away_clean, font=font_team)
            if (h_bb[2] - h_bb[0]) < 440 and (a_bb[2] - a_bb[0]) < 440:
                break

        team_y = 88
        draw.text((36, team_y), home_clean, fill=INK, font=font_team)
        draw.text((36, team_y + 52), "HOME", fill=LIGHT, font=font_role)

        a_bb2  = draw.textbbox((0, 0), away_clean, font=font_team)
        away_w = a_bb2[2] - a_bb2[0]
        draw.text((W - 36 - away_w, team_y), away_clean, fill=INK, font=font_team)
        role_bb = draw.textbbox((0, 0), "AWAY", font=font_role)
        draw.text((W - 36 - (role_bb[2] - role_bb[0]), team_y + 52), "AWAY", fill=LIGHT, font=font_role)

        vs_cx, vs_cy = W // 2, team_y + 28
        draw.ellipse([vs_cx - 22, vs_cy - 22, vs_cx + 22, vs_cy + 22], outline=BORDER, width=2, fill=BG)
        font_vs = get_card_font(13, bold=True)
        vs_bb   = draw.textbbox((0, 0), "VS", font=font_vs)
        draw.text((vs_cx - (vs_bb[2] - vs_bb[0]) // 2, vs_cy - (vs_bb[3] - vs_bb[1]) // 2), "VS", fill=MUTED, font=font_vs)

        draw.line([(36, 175), (W - 36, 175)], fill=DIVIDER, width=1)

        # -- 3. PICK + SCORE ------------------------------------------
        pick_y = 192

        draw.text((36, pick_y), "OUR PICK", fill=LIGHT, font=get_card_font(11, bold=True))

        pick_text = clean_text_for_image(prediction["pick_name"]).upper()
        font_pick_val = get_card_font(36, bold=True)
        for ps in [36, 30, 24]:
            font_pick_val = get_card_font(ps, bold=True)
            pk_bb = draw.textbbox((0, 0), pick_text, font=font_pick_val)
            if pk_bb[2] - pk_bb[0] < 550:
                break
        draw.text((36, pick_y + 20), pick_text, fill=INK, font=font_pick_val)

        font_meta = get_card_font(13)
        stake_txt = f"Stake: {prediction['stake_units']}   -   Odds: @{prediction['fair_odds']:.2f}"
        draw.text((36, pick_y + 72), stake_txt, fill=MUTED, font=font_meta)

        draw.line([(W // 2 + 40, pick_y), (W // 2 + 40, pick_y + 90)], fill=DIVIDER, width=1)

        hg, ag     = prediction["score"]
        exp_txt    = f"{hg}  -  {ag}"
        font_sc_lbl = get_card_font(11, bold=True)
        font_sc_val = get_card_font(48, bold=True)
        lbl_sc     = "EXPECTED SCORE"
        lbl_sc_bb  = draw.textbbox((0, 0), lbl_sc, font=font_sc_lbl)
        lbl_sc_x   = W // 2 + 40 + ((W // 2 - 76) - (lbl_sc_bb[2] - lbl_sc_bb[0])) // 2
        sc_bb      = draw.textbbox((0, 0), exp_txt, font=font_sc_val)
        sc_x       = W // 2 + 40 + ((W // 2 - 76) - (sc_bb[2] - sc_bb[0])) // 2
        draw.text((lbl_sc_x, pick_y), lbl_sc, fill=LIGHT, font=font_sc_lbl)
        draw.text((sc_x, pick_y + 16), exp_txt, fill=INK, font=font_sc_val)

        div2_y = pick_y + 102
        draw.line([(36, div2_y), (W - 36, div2_y)], fill=DIVIDER, width=1)

        # -- 4. PROBABILITY BAR ----------------------------------------
        bar_y = div2_y + 22
        draw.text((36, bar_y - 16), "WIN PROBABILITY", fill=LIGHT, font=get_card_font(11, bold=True))

        hp = prediction["home_win_prob"]
        dp = prediction["draw_prob"]
        ap = prediction["away_win_prob"]

        bx0, by0 = 36, bar_y
        bx1, by1 = W - 36, bar_y + 40
        bw = bx1 - bx0
        bh = by1 - by0
        hw = int(bw * hp)
        dw = int(bw * dp)
        aw = bw - hw - dw

        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=GREY_BAR)
        if hw > 0:
            draw.rounded_rectangle([bx0, by0, bx0 + hw, by1], radius=8, fill=GREEN)
            if hw > 20:
                draw.rectangle([bx0 + hw - 8, by0, bx0 + hw, by1], fill=GREEN)
        if dw > 0:
            draw.rectangle([bx0 + hw, by0, bx0 + hw + dw, by1], fill="#9ca3af")
        if aw > 0:
            draw.rounded_rectangle([bx0 + hw + dw, by0, bx1, by1], radius=8, fill=RED)
            if aw > 20:
                draw.rectangle([bx0 + hw + dw, by0, bx0 + hw + dw + 8, by1], fill=RED)

        font_bar_pct = get_card_font(13, bold=True)
        font_bar_lbl = get_card_font(12, bold=True)
        if hw > 60:
            h_in = f"{hp*100:.0f}%"
            in_bb = draw.textbbox((0, 0), h_in, font=font_bar_pct)
            draw.text((bx0 + hw // 2 - (in_bb[2] - in_bb[0]) // 2, by0 + (bh - (in_bb[3] - in_bb[1])) // 2 - 1), h_in, fill="#ffffff", font=font_bar_pct)
        if dw > 50:
            d_in = f"{dp*100:.0f}%"
            in_bb = draw.textbbox((0, 0), d_in, font=font_bar_pct)
            draw.text((bx0 + hw + dw // 2 - (in_bb[2] - in_bb[0]) // 2, by0 + (bh - (in_bb[3] - in_bb[1])) // 2 - 1), d_in, fill="#ffffff", font=font_bar_pct)
        if aw > 60:
            a_in = f"{ap*100:.0f}%"
            in_bb = draw.textbbox((0, 0), a_in, font=font_bar_pct)
            draw.text((bx0 + hw + dw + aw // 2 - (in_bb[2] - in_bb[0]) // 2, by0 + (bh - (in_bb[3] - in_bb[1])) // 2 - 1), a_in, fill="#ffffff", font=font_bar_pct)

        leg_y = by1 + 10
        dot_r = 4
        draw.ellipse([36, leg_y + 3, 36 + dot_r * 2, leg_y + 3 + dot_r * 2], fill=GREEN)
        draw.text((36 + dot_r * 2 + 6, leg_y), f"{home_clean}  {hp*100:.0f}%", fill=MUTED, font=font_bar_lbl)

        dl_txt = f"Draw  {dp*100:.0f}%"
        dl_bb  = draw.textbbox((0, 0), dl_txt, font=font_bar_lbl)
        dl_x   = W // 2 - (dl_bb[2] - dl_bb[0]) // 2
        draw.ellipse([dl_x - 14, leg_y + 3, dl_x - 14 + dot_r * 2, leg_y + 3 + dot_r * 2], fill="#9ca3af")
        draw.text((dl_x, leg_y), dl_txt, fill=MUTED, font=font_bar_lbl)

        al_txt = f"{away_clean}  {ap*100:.0f}%"
        al_bb  = draw.textbbox((0, 0), al_txt, font=font_bar_lbl)
        al_x   = W - 36 - (al_bb[2] - al_bb[0])
        draw.ellipse([al_x - 14, leg_y + 3, al_x - 14 + dot_r * 2, leg_y + 3 + dot_r * 2], fill=RED)
        draw.text((al_x, leg_y), al_txt, fill=MUTED, font=font_bar_lbl)

        # -- 5. BOTTOM STAT + FOOTER -----------------------------------
        bot_y = leg_y + 30
        draw.line([(36, bot_y), (W - 36, bot_y)], fill=DIVIDER, width=1)

        font_stat_lbl = get_card_font(11, bold=True)
        font_stat_val = get_card_font(22, bold=True)
        over_pct = prediction["over_2_5_prob"] * 100

        draw.text((36, bot_y + 14), "GOALS OVER 2.5", fill=LIGHT, font=font_stat_lbl)
        draw.text((36, bot_y + 32), f"{over_pct:.0f}%", fill=INK, font=font_stat_val)

        font_foot = get_card_font(11)
        foot_txt  = "RYUU PREDICTION AI"
        foot_bb   = draw.textbbox((0, 0), foot_txt, font=font_foot)
        draw.text((W - 36 - (foot_bb[2] - foot_bb[0]), bot_y + 14), foot_txt, fill=LIGHT, font=font_foot)
        sub_txt = "Statistical Model"
        sub_bb  = draw.textbbox((0, 0), sub_txt, font=font_foot)
        draw.text((W - 36 - (sub_bb[2] - sub_bb[0]), bot_y + 32), sub_txt, fill=LIGHT, font=font_foot)

        img.save(file_path, quality=95)
        return file_path

    except Exception as e:
        print(f"[warn] Card image generation failed: {e} -- falling back to text.", file=sys.stderr)
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

    # Step 2 — Fetch upcoming fixtures (focused lookahead window)
    print(f"Fetching upcoming fixtures (next {days_ahead} days)...")
    fixtures = get_upcoming_fixtures(days_ahead=days_ahead)

    # Always persist state here regardless of what happens next
    save_sent_fixtures(sent_data)

    if not fixtures:
        print(f"No upcoming fixtures found in next {days_ahead} days.")
        return

    # Step 3 — Filter to only new fixtures
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
