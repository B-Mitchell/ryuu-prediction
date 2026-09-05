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
# Pillow Image Card Generator — Glassmorphism Dark (1200×675)
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


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple:
    """Convert a hex color string to an RGBA tuple."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


def _draw_glass_panel(
    img: Image.Image,
    x0: int, y0: int, x1: int, y1: int,
    fill_rgba: tuple = (255, 255, 255, 20),
    border_rgba: tuple = (0, 230, 200, 60),
    radius: int = 18,
    border_width: int = 1,
) -> None:
    """Draws a frosted-glass panel with transparency onto img (RGBA mode)."""
    panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)
    pdraw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill_rgba, outline=border_rgba, width=border_width)
    img.alpha_composite(panel)


def _draw_glow_line(img: Image.Image, x0: int, y: int, x1: int, color_rgba: tuple, thickness: int = 2) -> None:
    """Draws a soft horizontal glow line by layering semi-transparent strokes."""
    for spread, alpha_mult in [(4, 0.15), (2, 0.25), (0, 1.0)]:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(layer)
        r, g, b, _ = color_rgba
        a = int(color_rgba[3] * alpha_mult)
        for off in range(-spread, spread + 1):
            ldraw.line([(x0, y + off), (x1, y + off)], fill=(r, g, b, max(0, a - abs(off) * 10)), width=1)
        img.alpha_composite(layer)


def create_match_card_image(fixture: dict, prediction: dict) -> str | None:
    """Generates a premium glassmorphism dark-mode match preview card PNG (1200x675)."""
    try:
        os.makedirs(MATCH_CARDS_DIR, exist_ok=True)
        file_path = os.path.join(MATCH_CARDS_DIR, f"card_{fixture['id']}.png")

        W, H = 1200, 675

        # ── Palette ──────────────────────────────────────────────────────
        BG_DARK       = (10, 15, 28)        # near-black navy
        BG_MID        = (14, 22, 42)        # deep navy for cards
        TEAL          = (0, 212, 180)       # primary teal/aqua accent
        TEAL_DIM      = (0, 160, 136)       # darker teal
        TEAL_GLOW     = (0, 212, 180, 120)  # teal with alpha for glows
        PANEL_FILL    = (255, 255, 255, 14) # near-transparent white glass
        PANEL_BORDER  = (0, 212, 180, 55)   # teal border
        TEXT_WHITE    = "#f0f6ff"
        TEXT_MUTED    = "#7a93b8"
        TEXT_DIM      = "#445c7a"
        TEAL_HEX      = "#00d4b4"
        TEAL_DIM_HEX  = "#00a088"
        AMBER_HEX     = "#f59e0b"
        RED_HEX       = "#ff4d6d"
        SLATE_HEX     = "#64748b"
        WHITE_10      = (255, 255, 255, 10)
        WHITE_20      = (255, 255, 255, 20)
        WHITE_25      = (255, 255, 255, 25)

        # ── 0. BACKGROUND — deep dark with subtle radial-style gradient ──
        img = Image.new("RGBA", (W, H), BG_DARK)
        draw = ImageDraw.Draw(img)

        # Subtle dark blue-purple sweep across the top half
        for y_row in range(H // 2):
            alpha = int(30 * (1 - y_row / (H / 2)))
            layer_row = Image.new("RGBA", (W, 1), (20, 30, 80, alpha))
            img.alpha_composite(layer_row, dest=(0, y_row))

        # Soft teal orb glow top-left and bottom-right corners
        for cx, cy, rad, base_alpha in [(200, 100, 320, 25), (1020, 580, 260, 20)]:
            for r_step in range(rad, 0, -6):
                a = int(base_alpha * (1 - r_step / rad))
                x0g, y0g = cx - r_step, cy - r_step
                x1g, y1g = cx + r_step, cy + r_step
                glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                gd = ImageDraw.Draw(glow_layer)
                gd.ellipse([x0g, y0g, x1g, y1g], fill=(0, 212, 180, a))
                img.alpha_composite(glow_layer)

        # Outer card frame — subtle teal border
        _draw_glass_panel(img, 12, 12, W - 12, H - 12,
                          fill_rgba=(0, 0, 0, 0), border_rgba=(0, 212, 180, 40), radius=28, border_width=1)

        # ── 1. HEADER AREA ────────────────────────────────────────────
        # Top divider glow line
        _draw_glow_line(img, 40, 80, W - 40, TEAL_GLOW, thickness=1)

        comp_text = clean_text_for_image(fixture.get("competition", "COMPETITION")).upper()
        font_comp  = get_card_font(13, bold=True)
        font_eng   = get_card_font(12)
        font_date  = get_card_font(13)

        # Competition pill
        draw2 = ImageDraw.Draw(img)
        comp_bb = draw2.textbbox((0, 0), comp_text, font=font_comp)
        comp_w  = comp_bb[2] - comp_bb[0] + 32
        comp_h  = 28
        _draw_glass_panel(img, 42, 26, 42 + comp_w, 26 + comp_h,
                          fill_rgba=(0, 212, 180, 20), border_rgba=(0, 212, 180, 80), radius=8, border_width=1)
        draw2 = ImageDraw.Draw(img)
        # dot
        draw2.ellipse([52, 26 + comp_h // 2 - 4, 60, 26 + comp_h // 2 + 4], fill=TEAL)
        draw2.text((64, 26 + (comp_h - (comp_bb[3] - comp_bb[1])) // 2), comp_text, fill=TEAL_HEX, font=font_comp)

        # Centre engine pill
        eng_txt = "RYUU PREDICTION AI  •  STATISTICAL MODEL"
        eng_bb  = draw2.textbbox((0, 0), eng_txt, font=font_eng)
        eng_w   = eng_bb[2] - eng_bb[0] + 28
        eng_x   = W // 2 - eng_w // 2
        _draw_glass_panel(img, eng_x, 26, eng_x + eng_w, 26 + comp_h,
                          fill_rgba=WHITE_10, border_rgba=(0, 212, 180, 35), radius=8, border_width=1)
        draw2 = ImageDraw.Draw(img)
        draw2.text((eng_x + 14, 26 + (comp_h - (eng_bb[3] - eng_bb[1])) // 2), eng_txt, fill=TEXT_MUTED, font=font_eng)

        # Date pill (right)
        dt = datetime.fromisoformat(fixture["utc_date"].replace("Z", "+00:00"))
        date_str = dt.strftime("%a %d %b  •  %H:%M UTC")
        dt_bb   = draw2.textbbox((0, 0), date_str, font=font_date)
        dt_w    = dt_bb[2] - dt_bb[0] + 28
        dt_x    = W - 42 - dt_w
        _draw_glass_panel(img, dt_x, 26, dt_x + dt_w, 26 + comp_h,
                          fill_rgba=WHITE_10, border_rgba=(100, 116, 139, 50), radius=8, border_width=1)
        draw2 = ImageDraw.Draw(img)
        draw2.text((dt_x + 14, 26 + (comp_h - (dt_bb[3] - dt_bb[1])) // 2), date_str, fill=TEXT_MUTED, font=font_date)

        # ── 2. TEAMS SECTION ──────────────────────────────────────────
        home_clean = clean_team_name(fixture["home_name"])
        away_clean = clean_team_name(fixture["away_name"])

        font_team = get_card_font(38, bold=True)
        font_role = get_card_font(11, bold=True)

        for s in [38, 33, 28, 23]:
            font_team = get_card_font(s, bold=True)
            h_bb2 = draw2.textbbox((0, 0), home_clean, font=font_team)
            a_bb2 = draw2.textbbox((0, 0), away_clean, font=font_team)
            if (h_bb2[2] - h_bb2[0]) < 430 and (a_bb2[2] - a_bb2[0]) < 430:
                break

        draw2.text((48, 97), home_clean, fill=TEXT_WHITE, font=font_team)
        draw2.text((50, 148), "HOME", fill=TEXT_DIM, font=font_role)

        a_bb3 = draw2.textbbox((0, 0), away_clean, font=font_team)
        away_w = a_bb3[2] - a_bb3[0]
        draw2.text((W - 48 - away_w, 97), away_clean, fill=TEXT_WHITE, font=font_team)
        role_bb = draw2.textbbox((0, 0), "AWAY", font=font_role)
        draw2.text((W - 48 - (role_bb[2] - role_bb[0]), 148), "AWAY", fill=TEXT_DIM, font=font_role)

        # VS badge — glowing teal circle
        vs_cx, vs_cy = W // 2, 124
        for r_ring, alpha_ring in [(34, 15), (28, 30)]:
            gl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            gld = ImageDraw.Draw(gl)
            gld.ellipse([vs_cx - r_ring, vs_cy - r_ring, vs_cx + r_ring, vs_cy + r_ring],
                        fill=(0, 212, 180, alpha_ring))
            img.alpha_composite(gl)
        _draw_glass_panel(img, vs_cx - 26, vs_cy - 26, vs_cx + 26, vs_cy + 26,
                          fill_rgba=(0, 212, 180, 25), border_rgba=(0, 212, 180, 140), radius=26, border_width=1)
        draw2 = ImageDraw.Draw(img)
        font_vs = get_card_font(14, bold=True)
        vs_bb   = draw2.textbbox((0, 0), "VS", font=font_vs)
        draw2.text((vs_cx - (vs_bb[2] - vs_bb[0]) // 2, vs_cy - (vs_bb[3] - vs_bb[1]) // 2), "VS", fill=TEAL_HEX, font=font_vs)

        # Thin separator
        _draw_glow_line(img, 40, 170, W - 40, (0, 212, 180, 50), thickness=1)

        # ── 3. HERO PICK PANEL ────────────────────────────────────────
        pick_y0, pick_y1 = 182, 332
        _draw_glass_panel(img, 42, pick_y0, W - 42, pick_y1,
                          fill_rgba=(255, 255, 255, 12), border_rgba=(0, 212, 180, 90), radius=18, border_width=1)

        # Top accent line on the panel
        for lx in range(42, W - 42):
            prog = (lx - 42) / (W - 84)
            a_line = int(180 * (1 - abs(prog - 0.5) * 2.5))
            a_line = max(0, min(180, a_line))
            acc_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(acc_layer).point((lx, pick_y0 + 1), fill=(0, 212, 180, a_line))
            img.alpha_composite(acc_layer)

        draw2 = ImageDraw.Draw(img)

        # "MODEL RECOMMENDATION" micro-tag
        font_tag = get_card_font(11, bold=True)
        tag_txt  = "  MODEL RECOMMENDATION"
        tag_bb   = draw2.textbbox((0, 0), tag_txt, font=font_tag)
        tag_w    = tag_bb[2] - tag_bb[0] + 10
        _draw_glass_panel(img, 62, pick_y0 + 14, 62 + tag_w, pick_y0 + 14 + 22,
                          fill_rgba=(0, 212, 180, 18), border_rgba=(0, 212, 180, 60), radius=6, border_width=1)
        draw2 = ImageDraw.Draw(img)
        draw2.ellipse([68, pick_y0 + 21, 76, pick_y0 + 29], fill=TEAL)
        draw2.text((78, pick_y0 + 16), "MODEL RECOMMENDATION", fill=TEAL_HEX, font=font_tag)

        # Pick outcome — BIG
        pick_text = clean_text_for_image(prediction["pick_name"]).upper()
        font_pick = get_card_font(34, bold=True)
        for ps in [34, 29, 24]:
            font_pick = get_card_font(ps, bold=True)
            pk_bb = draw2.textbbox((0, 0), pick_text, font=font_pick)
            if pk_bb[2] - pk_bb[0] < 640:
                break
        draw2.text((66, pick_y0 + 46), pick_text, fill=TEXT_WHITE, font=font_pick)

        # Confidence & stake badges row
        conf_level = prediction["confidence_level"]
        conf_color = TEAL_HEX if "High" in conf_level else (AMBER_HEX if "Med" in conf_level else SLATE_HEX)
        font_badge = get_card_font(13)
        bx_cur     = 66
        badge_y    = pick_y0 + 104

        for badge_txt, b_text_col, b_border_alpha in [
            (f"Confidence: {conf_level}", conf_color, 80),
            (f"Stake: {prediction['stake_units']}", TEXT_MUTED, 50),
            (f"Odds: @{prediction['fair_odds']:.2f}", TEAL_HEX, 70),
        ]:
            bb_b = draw2.textbbox((0, 0), badge_txt, font=font_badge)
            bw_b = bb_b[2] - bb_b[0] + 26
            bh_b = 26
            _draw_glass_panel(img, bx_cur, badge_y, bx_cur + bw_b, badge_y + bh_b,
                              fill_rgba=WHITE_10, border_rgba=(0, 212, 180, b_border_alpha), radius=7, border_width=1)
            draw2 = ImageDraw.Draw(img)
            draw2.text((bx_cur + 13, badge_y + (bh_b - (bb_b[3] - bb_b[1])) // 2), badge_txt, fill=b_text_col, font=font_badge)
            bx_cur += bw_b + 10

        # Expected Score box — right side
        hg, ag  = prediction["score"]
        exp_txt = f"{hg} - {ag}"
        sc_box_w = 190
        sc_x0    = W - 42 - sc_box_w - 16
        _draw_glass_panel(img, sc_x0, pick_y0 + 14, sc_x0 + sc_box_w, pick_y1 - 14,
                          fill_rgba=(0, 212, 180, 10), border_rgba=(0, 212, 180, 80), radius=14, border_width=1)
        draw2 = ImageDraw.Draw(img)
        font_sc_lbl = get_card_font(11, bold=True)
        font_sc_val = get_card_font(40, bold=True)
        sc_lbl_bb   = draw2.textbbox((0, 0), "EXPECTED SCORE", font=font_sc_lbl)
        draw2.text((sc_x0 + (sc_box_w - (sc_lbl_bb[2] - sc_lbl_bb[0])) // 2, pick_y0 + 30),
                   "EXPECTED SCORE", fill=TEXT_DIM, font=font_sc_lbl)
        sc_val_bb = draw2.textbbox((0, 0), exp_txt, font=font_sc_val)
        draw2.text((sc_x0 + (sc_box_w - (sc_val_bb[2] - sc_val_bb[0])) // 2, pick_y0 + 56),
                   exp_txt, fill=TEAL_HEX, font=font_sc_val)

        # ── 4. PROBABILITY BAR ───────────────────────────────────────
        bar_y_top = 348
        draw2.text((46, bar_y_top), "WIN PROBABILITY DISTRIBUTION", fill=TEXT_DIM, font=get_card_font(11, bold=True))

        bx0, by0b, bx1, by1b = 46, bar_y_top + 20, W - 46, bar_y_top + 54
        bw = bx1 - bx0
        bh = by1b - by0b

        hp = prediction["home_win_prob"]
        dp = prediction["draw_prob"]
        ap = prediction["away_win_prob"]
        hw = int(bw * hp)
        dw = int(bw * dp)
        aw = bw - hw - dw

        # Background track
        _draw_glass_panel(img, bx0, by0b, bx1, by1b,
                          fill_rgba=(255, 255, 255, 8), border_rgba=(255, 255, 255, 20), radius=10, border_width=1)

        # Home segment — teal
        if hw > 0:
            seg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sd  = ImageDraw.Draw(seg)
            sd.rounded_rectangle([bx0, by0b, bx0 + hw, by1b], radius=10, fill=(0, 212, 180, 220))
            if hw > 20:
                sd.rectangle([bx0 + max(0, hw - 12), by0b, bx0 + hw, by1b], fill=(0, 212, 180, 220))
            img.alpha_composite(seg)

        # Draw segment — muted slate
        if dw > 0:
            seg2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sd2  = ImageDraw.Draw(seg2)
            sd2.rectangle([bx0 + hw, by0b, bx0 + hw + dw, by1b], fill=(100, 116, 139, 160))
            img.alpha_composite(seg2)

        # Away segment — coral red
        if aw > 0:
            seg3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sd3  = ImageDraw.Draw(seg3)
            sd3.rounded_rectangle([bx0 + hw + dw, by0b, bx1, by1b], radius=10, fill=(255, 77, 109, 200))
            if aw > 20:
                sd3.rectangle([bx0 + hw + dw, by0b, bx0 + hw + dw + 12, by1b], fill=(255, 77, 109, 200))
            img.alpha_composite(seg3)

        # In-bar labels
        draw2 = ImageDraw.Draw(img)
        font_in_bar = get_card_font(14, bold=True)
        if hw > 70:
            h_in = f"{hp*100:.0f}%"
            in_bb = draw2.textbbox((0, 0), h_in, font=font_in_bar)
            draw2.text((bx0 + hw // 2 - (in_bb[2] - in_bb[0]) // 2,
                        by0b + (bh - (in_bb[3] - in_bb[1])) // 2 - 1), h_in, fill="#001a14", font=font_in_bar)
        if dw > 60:
            d_in = f"{dp*100:.0f}%"
            in_bb = draw2.textbbox((0, 0), d_in, font=font_in_bar)
            draw2.text((bx0 + hw + dw // 2 - (in_bb[2] - in_bb[0]) // 2,
                        by0b + (bh - (in_bb[3] - in_bb[1])) // 2 - 1), d_in, fill="#f0f6ff", font=font_in_bar)
        if aw > 70:
            a_in = f"{ap*100:.0f}%"
            in_bb = draw2.textbbox((0, 0), a_in, font=font_in_bar)
            draw2.text((bx0 + hw + dw + aw // 2 - (in_bb[2] - in_bb[0]) // 2,
                        by0b + (bh - (in_bb[3] - in_bb[1])) // 2 - 1), a_in, fill="#f0f6ff", font=font_in_bar)

        # Legend row
        font_leg = get_card_font(13, bold=True)
        leg_y    = by1b + 10
        dot_r    = 5
        draw2.ellipse([46, leg_y + 4, 46 + dot_r * 2, leg_y + 4 + dot_r * 2], fill=TEAL)
        draw2.text((58, leg_y), f"{home_clean} ({hp*100:.0f}%)", fill=TEAL_HEX, font=font_leg)

        dl_txt = f"Draw  ({dp*100:.0f}%)"
        dl_bb  = draw2.textbbox((0, 0), dl_txt, font=font_leg)
        dl_x   = W // 2 - (dl_bb[2] - dl_bb[0]) // 2
        draw2.ellipse([dl_x - 16, leg_y + 4, dl_x - 16 + dot_r * 2, leg_y + 4 + dot_r * 2], fill=(100, 116, 139))
        draw2.text((dl_x, leg_y), dl_txt, fill=TEXT_MUTED, font=font_leg)

        al_txt = f"{away_clean} ({ap*100:.0f}%)"
        al_bb  = draw2.textbbox((0, 0), al_txt, font=font_leg)
        al_x   = W - 46 - (al_bb[2] - al_bb[0])
        draw2.ellipse([al_x - 16, leg_y + 4, al_x - 16 + dot_r * 2, leg_y + 4 + dot_r * 2], fill=(255, 77, 109))
        draw2.text((al_x, leg_y), al_txt, fill=RED_HEX, font=font_leg)

        # ── 5. STAT CARDS GRID ───────────────────────────────────────
        grid_y0 = 462
        gw = (W - 90 - 32) // 3
        gh = 120

        btts_label = "YES" if prediction["btts_yes_prob"] >= 0.5 else "NO"
        btts_pct   = prediction["btts_yes_prob"] * 100 if btts_label == "YES" else (1 - prediction["btts_yes_prob"]) * 100
        btts_col   = TEAL_HEX if btts_label == "YES" else AMBER_HEX

        cards_data = [
            ("GOALS OVER 2.5",       f"{prediction['over_2_5_prob']*100:.0f}%",   "Market Probability",   TEAL_HEX),
            ("BOTH TEAMS TO SCORE",  f"{btts_label} ({btts_pct:.0f}%)",           "Expected Outcome",     btts_col),
            ("MOST LIKELY SCORE",    f"{hg} - {ag}",                              "Dixon-Coles Mode",     TEAL_HEX),
        ]

        font_c_title = get_card_font(11, bold=True)
        font_c_val   = get_card_font(28, bold=True)
        font_c_sub   = get_card_font(10)

        for i, (title, val, sub, val_col) in enumerate(cards_data):
            cx0 = 45 + i * (gw + 16)
            cx1 = cx0 + gw
            _draw_glass_panel(img, cx0, grid_y0, cx1, grid_y0 + gh,
                              fill_rgba=(255, 255, 255, 10), border_rgba=(0, 212, 180, 45), radius=14, border_width=1)
            draw2 = ImageDraw.Draw(img)
            draw2.text((cx0 + 20, grid_y0 + 14), title, fill=TEXT_DIM, font=font_c_title)
            draw2.text((cx0 + 20, grid_y0 + 38), val, fill=val_col, font=font_c_val)
            draw2.text((cx0 + 20, grid_y0 + 88), sub, fill=TEXT_DIM, font=font_c_sub)

        # ── 6. FOOTER ────────────────────────────────────────────────
        _draw_glow_line(img, 40, H - 48, W - 40, (0, 212, 180, 40), thickness=1)
        draw2 = ImageDraw.Draw(img)
        font_foot = get_card_font(10)
        draw2.text((46, H - 36), "Dixon-Coles Bivariate Poisson Distribution  •  Data-driven algorithmic selections",
                   fill=TEXT_DIM, font=font_foot)
        foot_txt = "RYUU PREDICTION AI"
        foot_bb  = draw2.textbbox((0, 0), foot_txt, font=font_foot)
        draw2.text((W - 46 - (foot_bb[2] - foot_bb[0]), H - 36), foot_txt, fill=TEAL_HEX, font=font_foot)

        # Convert RGBA → RGB
        bg_rgb = Image.new("RGB", (W, H), (10, 15, 28))
        bg_rgb.paste(img.convert("RGB"), mask=img.split()[3])
        final_img = bg_rgb
        final_img.save(file_path, quality=95)
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
