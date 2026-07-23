#!/usr/bin/env python3
"""
Football Prophet — Test Suite
==============================
Runs offline unit tests (no API key needed) and optional live integration
tests (requires real API credentials in .env).

NOTE: All dates printed during unit tests are SYNTHETIC TEST DATA.
      Fixture dates like "26 Jul" are computed as today + N days purely
      to exercise countdown logic — they are NOT real fixtures and nothing
      is ever sent to Telegram during unit tests (send functions are mocked).

Usage:
  python test_predictor.py              # unit tests only (offline)
  python test_predictor.py --live       # unit + live API integration tests
  python test_predictor.py --live --telegram  # also fires a test Telegram message
"""

import os
import sys
import json
import argparse
import tempfile
import unittest
import unittest.mock
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import StringIO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Import the module under test ──────────────────────────────────────────────
import predictor


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures used across tests
# ─────────────────────────────────────────────────────────────────────────────
FIXTURE_ARSENAL_CHELSEA = {
    "id":               99001,
    "competition":      "Premier League",
    "competition_code": "PL",
    "utc_date":         "2026-08-23T14:00:00Z",
    "home_id":          57,
    "home_name":        "Arsenal FC",
    "away_id":          61,
    "away_name":        "Chelsea FC",
}

FIXTURE_REAL_BARCA = {
    "id":               99002,
    "competition":      "La Liga",
    "competition_code": "PD",
    "utc_date":         "2026-08-24T19:00:00Z",
    "home_id":          86,
    "home_name":        "Real Madrid CF",
    "away_id":          81,
    "away_name":        "FC Barcelona",
}

STRONG_HOME_STATS  = {"attack_home": 2.4, "defense_home": 0.6, "attack_away": 1.8, "defense_away": 0.8}
WEAK_AWAY_STATS    = {"attack_home": 0.8, "defense_home": 1.6, "attack_away": 0.7, "defense_away": 1.8}
BALANCED_STATS     = {"attack_home": 1.4, "defense_home": 1.2, "attack_away": 1.2, "defense_away": 1.4}
FALLBACK_STATS     = {"attack_home": 1.3, "defense_home": 1.1, "attack_away": 1.1, "defense_away": 1.3}

SAMPLE_FINISHED_MATCH = {
    "homeTeam": {"id": 57},
    "awayTeam": {"id": 61},
    "utcDate":  "2026-05-10T14:00:00Z",
    "score":    {"fullTime": {"home": 2, "away": 1}},
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Utility Helpers
# ═════════════════════════════════════════════════════════════════════════════
class TestHelpers(unittest.TestCase):

    # ── clean_team_name ───────────────────────────────────────────────────────
    def test_clean_fc_suffix(self):
        self.assertEqual(predictor.clean_team_name("Arsenal FC"), "Arsenal")

    def test_clean_afc_suffix(self):
        self.assertEqual(predictor.clean_team_name("AFC Bournemouth"), "Bournemouth")

    def test_clean_override_applied(self):
        self.assertEqual(predictor.clean_team_name("Manchester City FC"), "Man City")
        self.assertEqual(predictor.clean_team_name("Paris Saint-Germain FC"), "PSG")
        self.assertEqual(predictor.clean_team_name("Rayo Vallecano de Madrid"), "Rayo Vallecano")

    def test_clean_name_no_suffix_unchanged(self):
        self.assertEqual(predictor.clean_team_name("Barcelona"), "Barcelona")

    def test_clean_name_empty(self):
        self.assertEqual(predictor.clean_team_name(""), "")

    def test_clean_name_none_safe(self):
        self.assertEqual(predictor.clean_team_name(None), "")

    # ── clean_text_for_image ──────────────────────────────────────────────────
    def test_emoji_stripped(self):
        result = predictor.clean_text_for_image("🔥 Premier League ⚽")
        self.assertNotIn("🔥", result)
        self.assertNotIn("⚽", result)
        self.assertIn("Premier League", result)

    def test_no_emoji_unchanged(self):
        self.assertEqual(predictor.clean_text_for_image("Arsenal vs Chelsea"), "Arsenal vs Chelsea")

    # ── truncate_caption ──────────────────────────────────────────────────────
    def test_short_caption_untouched(self):
        text = "Short message"
        self.assertEqual(predictor.truncate_caption(text), text)

    def test_long_caption_truncated(self):
        text = "A" * 1100
        result = predictor.truncate_caption(text)
        self.assertLessEqual(len(result), 1000)
        self.assertTrue(result.endswith("..."))

    def test_caption_exactly_at_limit_untouched(self):
        text = "B" * 1000
        self.assertEqual(predictor.truncate_caption(text), text)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Sent Fixtures State Management
# ═════════════════════════════════════════════════════════════════════════════
class TestSentFixtures(unittest.TestCase):

    def _empty_state(self):
        return {"fixtures": {}, "announcements": {}}

    def test_mark_and_check_fixture_today(self):
        state = self._empty_state()
        predictor.mark_fixture_sent(12345, state)
        self.assertTrue(predictor.is_fixture_sent(12345, state))

    def test_unsent_fixture_returns_false(self):
        state = self._empty_state()
        self.assertFalse(predictor.is_fixture_sent(99999, state))

    def test_cross_midnight_dedup_7_days(self):
        """Fixture sent yesterday must still be flagged as sent today."""
        state = self._empty_state()
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        state["fixtures"][yesterday] = [55555]
        self.assertTrue(predictor.is_fixture_sent(55555, state))

    def test_fixture_outside_7_days_not_blocked(self):
        """Fixture sent 8 days ago should not block a resend."""
        state = self._empty_state()
        old_date = (datetime.now(timezone.utc) - timedelta(days=8)).date().isoformat()
        state["fixtures"][old_date] = [77777]
        self.assertFalse(predictor.is_fixture_sent(77777, state))

    def test_announcement_keys_separate_from_fixtures(self):
        state = self._empty_state()
        ann_key = "announce_PL_2026-08-21_d4"
        self.assertFalse(predictor.is_announcement_sent(ann_key, state))
        predictor.mark_announcement_sent(ann_key, state)
        self.assertTrue(predictor.is_announcement_sent(ann_key, state))
        # Must NOT appear in fixtures dict
        for bucket in state["fixtures"].values():
            self.assertNotIn(ann_key, bucket)

    def test_save_and_reload_roundtrip(self):
        state = self._empty_state()
        predictor.mark_fixture_sent(42, state)
        predictor.mark_announcement_sent("announce_BL1_2026-08-20_d0", state)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = f.name
        try:
            with open(tmp, "w") as f:
                json.dump(state, f)
            with unittest.mock.patch("predictor.SENT_FIXTURES_FILE", tmp):
                loaded = predictor.load_sent_fixtures()
            self.assertTrue(predictor.is_fixture_sent(42, loaded))
            self.assertTrue(predictor.is_announcement_sent("announce_BL1_2026-08-20_d0", loaded))
        finally:
            os.unlink(tmp)

    def test_old_flat_format_migrates(self):
        """Old format (bare dict of dates) should be auto-migrated to new structure."""
        today = datetime.now(timezone.utc).date().isoformat()
        old_format = {today: [111, 222]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(old_format, f)
            tmp = f.name
        try:
            with unittest.mock.patch("predictor.SENT_FIXTURES_FILE", tmp):
                result = predictor.load_sent_fixtures()
            self.assertIn("fixtures", result)
            self.assertIn("announcements", result)
            self.assertIn(today, result["fixtures"])
        finally:
            os.unlink(tmp)

    def test_corrupt_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("NOT VALID JSON {{{{")
            tmp = f.name
        try:
            with unittest.mock.patch("predictor.SENT_FIXTURES_FILE", tmp):
                result = predictor.load_sent_fixtures()
            self.assertEqual(result, {"fixtures": {}, "announcements": {}})
        finally:
            os.unlink(tmp)

    def test_retention_pruning(self):
        """Dates older than SENT_FIXTURES_RETENTION_DAYS should be pruned."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
        state = {"fixtures": {old_date: [1, 2]}, "announcements": {}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(state, f)
            tmp = f.name
        try:
            with unittest.mock.patch("predictor.SENT_FIXTURES_FILE", tmp):
                result = predictor.load_sent_fixtures()
            self.assertNotIn(old_date, result["fixtures"])
        finally:
            os.unlink(tmp)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Prediction Model
# ═════════════════════════════════════════════════════════════════════════════
class TestPredictionModel(unittest.TestCase):

    # ── poisson_pmf ───────────────────────────────────────────────────────────
    def test_poisson_pmf_zero_goals(self):
        """P(0 | lambda=1) = e^-1 ≈ 0.368"""
        import math
        val = predictor.poisson_pmf(0, 1.0)
        self.assertAlmostEqual(val, math.exp(-1), places=6)

    def test_poisson_pmf_positive(self):
        for k in range(7):
            self.assertGreater(predictor.poisson_pmf(k, 1.5), 0)

    # ── dixon_coles_adjustment ────────────────────────────────────────────────
    def test_dc_adjustment_0_0_increases(self):
        """With rho=-0.1, 0-0 adjustment > 1 (more 0-0 than pure Poisson)."""
        adj = predictor.dixon_coles_adjustment(0, 0, 1.2, 1.0, rho=-0.1)
        self.assertGreater(adj, 1.0)

    def test_dc_adjustment_high_scores_is_one(self):
        """For scores above 1-1, adjustment is exactly 1."""
        self.assertEqual(predictor.dixon_coles_adjustment(2, 2, 1.2, 1.0), 1.0)
        self.assertEqual(predictor.dixon_coles_adjustment(3, 1, 1.2, 1.0), 1.0)

    # ── team_goal_rates ───────────────────────────────────────────────────────
    def test_fallback_on_no_matches(self):
        rates = predictor.team_goal_rates(57, [])
        self.assertEqual(rates, {"attack_home": 1.3, "defense_home": 1.1, "attack_away": 1.1, "defense_away": 1.3})

    def test_home_match_contribution(self):
        """A team that always wins 3-0 at home should have high home attack."""
        matches = []
        for i in range(5):
            matches.append({
                "homeTeam": {"id": 57},
                "awayTeam": {"id": 100},
                "utcDate":  f"2026-0{i+1}-10T14:00:00Z",
                "score":    {"fullTime": {"home": 3, "away": 0}},
            })
        rates = predictor.team_goal_rates(57, matches)
        self.assertGreater(rates["attack_home"], 2.0)
        self.assertLess(rates["defense_home"], 0.5)

    def test_away_match_contribution(self):
        """A team that always loses 0-2 away should have low away attack."""
        matches = []
        for i in range(5):
            matches.append({
                "homeTeam": {"id": 100},
                "awayTeam": {"id": 57},
                "utcDate":  f"2026-0{i+1}-10T14:00:00Z",
                "score":    {"fullTime": {"home": 2, "away": 0}},
            })
        rates = predictor.team_goal_rates(57, matches)
        self.assertAlmostEqual(rates["attack_away"], 0.0, places=4)

    def test_skips_matches_with_null_scores(self):
        """Matches with None scores must be silently skipped."""
        matches = [
            {"homeTeam": {"id": 57}, "awayTeam": {"id": 100},
             "utcDate": "2026-01-10T14:00:00Z",
             "score": {"fullTime": {"home": None, "away": None}}},
        ]
        rates = predictor.team_goal_rates(57, matches)
        # Should fall back to neutral rates
        self.assertEqual(rates, predictor.FALLBACK_STATS if hasattr(predictor, "FALLBACK_STATS")
                         else {"attack_home": 1.3, "defense_home": 1.1, "attack_away": 1.1, "defense_away": 1.3})

    # ── predict_match ─────────────────────────────────────────────────────────
    def _predict(self, home=BALANCED_STATS, away=BALANCED_STATS, code="PL", fixture=FIXTURE_ARSENAL_CHELSEA):
        return predictor.predict_match(home, away, code, fixture)

    def test_probabilities_sum_to_one(self):
        pred = self._predict()
        total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_over_2_5_in_range(self):
        pred = self._predict()
        self.assertGreaterEqual(pred["over_2_5_prob"], 0.0)
        self.assertLessEqual(pred["over_2_5_prob"], 1.0)

    def test_btts_in_range(self):
        pred = self._predict()
        self.assertGreaterEqual(pred["btts_yes_prob"], 0.0)
        self.assertLessEqual(pred["btts_yes_prob"], 1.0)

    def test_strong_home_team_picks_home_win(self):
        pred = predictor.predict_match(STRONG_HOME_STATS, WEAK_AWAY_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)
        self.assertEqual(pred["pick_outcome"], "HOME_WIN")
        self.assertGreater(pred["home_win_prob"], 0.60)

    def test_strong_away_team_picks_away_win(self):
        pred = predictor.predict_match(WEAK_AWAY_STATS, STRONG_HOME_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)
        self.assertEqual(pred["pick_outcome"], "AWAY_WIN")

    def test_fair_odds_reciprocal_of_probability(self):
        pred = self._predict()
        self.assertAlmostEqual(pred["fair_odds"], round(1.0 / pred["pick_prob"], 2), places=2)

    def test_high_confidence_3_units(self):
        """A strong favourite should yield High confidence & 3/3 Units."""
        pred = predictor.predict_match(STRONG_HOME_STATS, WEAK_AWAY_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)
        self.assertEqual(pred["confidence_level"], "High")
        self.assertEqual(pred["stake_units"], "3/3 Units")

    def test_low_confidence_1_unit(self):
        """Balanced teams should yield Low/Medium confidence."""
        pred = predictor.predict_match(BALANCED_STATS, BALANCED_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)
        self.assertIn(pred["confidence_level"], ["Low", "Medium"])

    def test_lambda_clamp_upper(self):
        """Extremely strong team should not exceed lambda=4."""
        extreme = {"attack_home": 10.0, "defense_home": 0.1, "attack_away": 10.0, "defense_away": 0.1}
        pred = predictor.predict_match(extreme, WEAK_AWAY_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)
        self.assertLessEqual(pred["lambda_home"], 4.0)

    def test_lambda_clamp_lower(self):
        """Extremely weak team should not go below lambda=0.3."""
        extreme = {"attack_home": 0.01, "defense_home": 5.0, "attack_away": 0.01, "defense_away": 5.0}
        pred = predictor.predict_match(extreme, STRONG_HOME_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)
        self.assertGreaterEqual(pred["lambda_home"], 0.3)

    def test_fixture_none_uses_default_names(self):
        """predict_match with fixture=None should not crash and use generic names."""
        pred = predictor.predict_match(BALANCED_STATS, BALANCED_STATS)
        self.assertIn("pick_name", pred)

    def test_all_required_keys_present(self):
        pred = self._predict()
        required = [
            "score", "score_prob", "home_win_prob", "draw_prob", "away_win_prob",
            "over_2_5_prob", "btts_yes_prob", "pick_outcome", "pick_name",
            "pick_prob", "confidence_level", "confidence_label", "confidence_stars",
            "stake_units", "fair_odds", "lambda_home", "lambda_away",
        ]
        for key in required:
            self.assertIn(key, pred, f"Missing key: {key}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Prediction History & Accuracy Resolution
# ═════════════════════════════════════════════════════════════════════════════
class TestPredictionHistory(unittest.TestCase):

    def _make_prediction(self, fixture=FIXTURE_ARSENAL_CHELSEA, home=BALANCED_STATS, away=BALANCED_STATS):
        return predictor.predict_match(home, away, fixture["competition_code"], fixture)

    def test_record_prediction_appends(self):
        history = []
        pred = self._make_prediction()
        predictor.record_prediction(FIXTURE_ARSENAL_CHELSEA, pred, history)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["fixture_id"], FIXTURE_ARSENAL_CHELSEA["id"])
        self.assertEqual(history[0]["status"], "PENDING")

    def test_record_prediction_no_duplicates(self):
        history = []
        pred = self._make_prediction()
        predictor.record_prediction(FIXTURE_ARSENAL_CHELSEA, pred, history)
        predictor.record_prediction(FIXTURE_ARSENAL_CHELSEA, pred, history)
        self.assertEqual(len(history), 1)

    def test_history_contains_all_fields(self):
        history = []
        pred = self._make_prediction()
        predictor.record_prediction(FIXTURE_ARSENAL_CHELSEA, pred, history)
        item = history[0]
        for field in ["fixture_id", "predicted_pick", "pick_prob", "confidence_level",
                      "over_2_5_prob", "btts_yes_prob", "predicted_score", "created_at"]:
            self.assertIn(field, item)

    def test_resolve_skips_future_matches(self):
        """Pending predictions for future matches should not be resolved."""
        history = [{
            "fixture_id":     88888,
            "utc_date":       (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status":         "PENDING",
            "predicted_pick": "HOME_WIN",
            "predicted_score": [2, 0],
        }]
        with unittest.mock.patch("predictor.api_get") as mock_api:
            stats = predictor.resolve_past_predictions(history)
        mock_api.assert_not_called()
        self.assertEqual(stats["resolved_new"], 0)
        self.assertEqual(history[0]["status"], "PENDING")

    def test_resolve_updates_correct_outcome(self):
        """A finished HOME_WIN match should mark is_outcome_correct=True for HOME_WIN pick."""
        past_date = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        history = [{
            "fixture_id":       77001,
            "utc_date":         past_date,
            "status":           "PENDING",
            "predicted_pick":   "HOME_WIN",
            "predicted_score":  [2, 0],
        }]
        mock_response = {
            "status": "FINISHED",
            "score": {"fullTime": {"home": 3, "away": 1}},  # home won
        }
        with unittest.mock.patch("predictor.api_get", return_value=mock_response):
            with unittest.mock.patch("time.sleep"):
                stats = predictor.resolve_past_predictions(history)

        self.assertEqual(stats["resolved_new"], 1)
        self.assertEqual(history[0]["status"], "RESOLVED")
        self.assertTrue(history[0]["is_outcome_correct"])
        self.assertFalse(history[0]["is_exact_score_correct"])  # 3-1 ≠ 2-0

    def test_resolve_marks_wrong_outcome(self):
        past_date = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        history = [{
            "fixture_id":       77002,
            "utc_date":         past_date,
            "status":           "PENDING",
            "predicted_pick":   "HOME_WIN",
            "predicted_score":  [2, 0],
        }]
        mock_response = {
            "status": "FINISHED",
            "score": {"fullTime": {"home": 0, "away": 2}},  # away won
        }
        with unittest.mock.patch("predictor.api_get", return_value=mock_response):
            with unittest.mock.patch("time.sleep"):
                predictor.resolve_past_predictions(history)

        self.assertFalse(history[0]["is_outcome_correct"])
        self.assertEqual(history[0]["actual_outcome"], "AWAY_WIN")

    def test_resolve_exact_score_hit(self):
        past_date = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        history = [{
            "fixture_id":       77003,
            "utc_date":         past_date,
            "status":           "PENDING",
            "predicted_pick":   "HOME_WIN",
            "predicted_score":  [2, 1],
        }]
        mock_response = {
            "status": "FINISHED",
            "score": {"fullTime": {"home": 2, "away": 1}},
        }
        with unittest.mock.patch("predictor.api_get", return_value=mock_response):
            with unittest.mock.patch("time.sleep"):
                predictor.resolve_past_predictions(history)

        self.assertTrue(history[0]["is_exact_score_correct"])

    def test_accuracy_stats_calculation(self):
        """Accuracy stats should correctly aggregate resolved predictions."""
        past = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        history = [
            {"fixture_id": 1, "utc_date": past, "status": "PENDING", "predicted_pick": "HOME_WIN", "predicted_score": [2, 0]},
            {"fixture_id": 2, "utc_date": past, "status": "PENDING", "predicted_pick": "AWAY_WIN", "predicted_score": [0, 2]},
            {"fixture_id": 3, "utc_date": past, "status": "PENDING", "predicted_pick": "DRAW",     "predicted_score": [1, 1]},
        ]
        side_effects = [
            {"status": "FINISHED", "score": {"fullTime": {"home": 2, "away": 0}}},  # correct + exact
            {"status": "FINISHED", "score": {"fullTime": {"home": 1, "away": 0}}},  # wrong
            {"status": "FINISHED", "score": {"fullTime": {"home": 1, "away": 1}}},  # correct + exact
        ]
        with unittest.mock.patch("predictor.api_get", side_effect=side_effects):
            with unittest.mock.patch("time.sleep"):
                stats = predictor.resolve_past_predictions(history)

        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["resolved_new"], 3)
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["exact_hits"], 2)
        self.assertAlmostEqual(stats["hit_rate"], 200 / 3, places=2)

    def test_history_save_load_roundtrip(self):
        history = []
        pred = predictor.predict_match(BALANCED_STATS, BALANCED_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)
        predictor.record_prediction(FIXTURE_ARSENAL_CHELSEA, pred, history)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = f.name
        try:
            with unittest.mock.patch("predictor.PREDICTION_HISTORY_FILE", tmp):
                predictor.save_prediction_history(history)
                loaded = predictor.load_prediction_history()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["fixture_id"], FIXTURE_ARSENAL_CHELSEA["id"])
        finally:
            os.unlink(tmp)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Hype Announcements
# ═════════════════════════════════════════════════════════════════════════════
class TestSeasonAnnouncements(unittest.TestCase):

    def _make_fixture(self, days_offset: int, code: str = "PL", comp: str = "Premier League") -> dict:
        """
        Creates a SYNTHETIC test fixture at today + days_offset.
        These dates are fake and used only to exercise countdown logic.
        They do NOT represent real fixtures from the API.
        """
        utc_date = (datetime.now(timezone.utc) + timedelta(days=days_offset)).strftime("%Y-%m-%dT19:00:00Z")
        return {
            "id": 10000 + days_offset,
            "competition": comp,
            "competition_code": code,
            "utc_date": utc_date,
            "home_name": "Test Home FC",   # clearly fake team names
            "away_name": "Test Away FC",
        }

    def test_multiple_leagues_each_get_own_announcement(self):
        fixtures = [self._make_fixture(2, "PL", "Premier League"), self._make_fixture(2, "PD", "La Liga")]
        state    = {"fixtures": {}, "announcements": {}}

        sent_msgs = []
        with unittest.mock.patch("predictor.send_telegram_message", side_effect=lambda msg: (sent_msgs.append(msg) or True)):
            predictor.check_and_send_season_announcements(fixtures, state)

        self.assertEqual(len(sent_msgs), 2)
        leagues = {msg for msg in sent_msgs}
        self.assertTrue(any("Premier League" in m for m in leagues))
        self.assertTrue(any("La Liga" in m for m in leagues))

    def test_hype_message_contains_comp_name(self):
        msg = predictor.generate_season_hype_message("La Liga", 3, "Fri 15 Aug, 19:00 UTC")
        self.assertIn("La Liga", msg)

    def test_no_hardcoded_premier_league_in_generic_slogans(self):
        for _ in range(20):
            msg = predictor.generate_season_hype_message("Bundesliga", 2, "Fri 15 Aug, 14:00 UTC")
            self.assertNotIn("Premier League", msg)

    def test_countdown_phrase_correct(self):
        for days_left, expected_fragment in [(4, "4 DAYS"), (1, "TOMORROW"), (0, "TODAY")]:
            msg = predictor.generate_season_hype_message("Serie A", days_left, "Sat 16 Aug, 14:00 UTC")
            self.assertIn(expected_fragment, msg)

    def test_announcement_sent_only_once(self):
        fixtures = [self._make_fixture(3)]
        state    = {"fixtures": {}, "announcements": {}}

        sent_msgs = []
        # NOTE: send_telegram_message is mocked — no real messages are sent.
        #       The dates printed below are synthetic test data, not real fixtures.
        with unittest.mock.patch("predictor.send_telegram_message", side_effect=lambda msg: (sent_msgs.append(msg) or True)):
            predictor.check_and_send_season_announcements(fixtures, state)
            predictor.check_and_send_season_announcements(fixtures, state)

        self.assertEqual(len(sent_msgs), 1, "Announcement should be sent only once")

    def test_no_announcement_beyond_4_days(self):
        fixtures = [self._make_fixture(5)]
        state    = {"fixtures": {}, "announcements": {}}

        with unittest.mock.patch("predictor.send_telegram_message") as mock_send:
            predictor.check_and_send_season_announcements(fixtures, state)
        mock_send.assert_not_called()

    def test_matchday_0_no_announcement(self):
        """On matchday itself predictions serve as the announcement — no hype message."""
        fixtures = [self._make_fixture(0)]
        state    = {"fixtures": {}, "announcements": {}}

        with unittest.mock.patch("predictor.send_telegram_message") as mock_send:
            predictor.check_and_send_season_announcements(fixtures, state)
        mock_send.assert_not_called()

    def test_announcement_fires_on_day_1(self):
        """Day 1 (tomorrow) is the earliest valid announcement day."""
        fixtures = [self._make_fixture(1)]
        state    = {"fixtures": {}, "announcements": {}}

        sent_msgs = []
        with unittest.mock.patch("predictor.send_telegram_message", side_effect=lambda msg: (sent_msgs.append(msg) or True)):
            predictor.check_and_send_season_announcements(fixtures, state)

        self.assertEqual(len(sent_msgs), 1)
        self.assertIn("TOMORROW", sent_msgs[0])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Message Formatting
# ═════════════════════════════════════════════════════════════════════════════
class TestMessageFormatting(unittest.TestCase):

    def _pred(self, fixture=FIXTURE_ARSENAL_CHELSEA):
        return predictor.predict_match(STRONG_HOME_STATS, WEAK_AWAY_STATS, "PL", fixture)

    def test_prediction_message_contains_competition(self):
        msg = predictor.format_prediction_message(FIXTURE_ARSENAL_CHELSEA, self._pred())
        self.assertIn("Premier League", msg)

    def test_prediction_message_contains_clean_team_names(self):
        msg = predictor.format_prediction_message(FIXTURE_ARSENAL_CHELSEA, self._pred())
        self.assertIn("Arsenal", msg)
        self.assertIn("Chelsea", msg)
        self.assertNotIn("Arsenal FC", msg)

    def test_prediction_message_contains_pick(self):
        msg = predictor.format_prediction_message(FIXTURE_ARSENAL_CHELSEA, self._pred())
        self.assertIn("PICK:", msg)

    def test_prediction_message_contains_odds(self):
        msg = predictor.format_prediction_message(FIXTURE_ARSENAL_CHELSEA, self._pred())
        self.assertIn("@", msg)

    def test_accuracy_summary_returns_empty_for_zero_total(self):
        self.assertEqual(predictor.format_accuracy_summary({"total": 0}), "")

    def test_accuracy_summary_contains_hit_rate(self):
        stats = {
            "total": 10, "hits": 7, "hit_rate": 70.0, "exact_hits": 2,
            "high_total": 5, "high_hits": 4, "high_hit_rate": 80.0,
            "med_total": 5, "med_hits": 3, "med_hit_rate": 60.0,
        }
        msg = predictor.format_accuracy_summary(stats)
        self.assertIn("70.0%", msg)
        self.assertIn("80.0%", msg)

    def test_message_within_telegram_limit(self):
        pred = self._pred()
        msg  = predictor.format_prediction_message(FIXTURE_ARSENAL_CHELSEA, pred)
        self.assertLessEqual(len(predictor.truncate_caption(msg)), 1000)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Image Card Generation
# ═════════════════════════════════════════════════════════════════════════════
class TestCardGeneration(unittest.TestCase):

    def _pred(self):
        return predictor.predict_match(STRONG_HOME_STATS, WEAK_AWAY_STATS, "PL", FIXTURE_ARSENAL_CHELSEA)

    def test_card_generates_valid_png(self):
        pred = self._pred()
        # Use ignore_cleanup_errors to handle Windows file-lock on PIL-generated PNGs
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            with unittest.mock.patch("predictor.MATCH_CARDS_DIR", tmpdir):
                path = predictor.create_match_card_image(FIXTURE_ARSENAL_CHELSEA, pred)
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))
            # Check it's a valid image; close file handle before temp dir cleanup
            from PIL import Image as PILImage
            img = PILImage.open(path)
            try:
                self.assertEqual(img.size, (800, 450))
                self.assertEqual(img.mode, "RGB")
            finally:
                img.close()

    def test_card_returns_none_on_failure(self):
        """If image generation crashes it should return None, not raise."""
        pred = self._pred()
        with unittest.mock.patch("predictor.Image.new", side_effect=OSError("disk full")):
            path = predictor.create_match_card_image(FIXTURE_ARSENAL_CHELSEA, pred)
        self.assertIsNone(path)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Live Integration Tests (--live flag required)
# ═════════════════════════════════════════════════════════════════════════════
class TestLiveAPI(unittest.TestCase):
    """
    These tests make real HTTP requests to football-data.org.
    Run with:  python test_predictor.py --live
    """

    def setUp(self):
        if not getattr(self, "_live", False):
            self.skipTest("Live tests skipped — run with --live flag")
        if not predictor.FOOTBALL_DATA_API_KEY:
            self.skipTest("FOOTBALL_DATA_API_KEY not set")

    def test_api_get_competitions_endpoint(self):
        data = predictor.api_get("/competitions/PL")
        self.assertIn("name", data)
        self.assertEqual(data.get("code"), "PL")
        print(f"\n  [live] PL competition name: {data['name']}")

    def test_get_upcoming_fixtures_returns_list(self):
        fixtures = predictor.get_upcoming_fixtures(days_ahead=7)
        self.assertIsInstance(fixtures, list)
        print(f"\n  [live] Found {len(fixtures)} upcoming fixtures in next 7 days")
        if fixtures:
            f = fixtures[0]
            self.assertIn("id", f)
            self.assertIn("home_name", f)
            self.assertIn("away_name", f)
            self.assertIn("competition", f)
            print(f"  [live] First fixture: {f['home_name']} vs {f['away_name']} ({f['competition']})")

    def test_get_team_recent_matches_arsenal(self):
        matches = predictor.get_team_recent_matches(57)  # Arsenal
        self.assertIsInstance(matches, list)
        print(f"\n  [live] Arsenal: {len(matches)} recent finished matches")
        if matches:
            for m in matches[:3]:
                h = m.get("score", {}).get("fullTime", {}).get("home", "?")
                a = m.get("score", {}).get("fullTime", {}).get("away", "?")
                print(f"    {m.get('utcDate', '')[:10]}  {m['homeTeam']['name']} {h}-{a} {m['awayTeam']['name']}")

    def test_full_prediction_pipeline_live(self):
        """Full end-to-end: fetch stats, compute prediction, generate card."""
        matches_home = predictor.get_team_recent_matches(57)   # Arsenal
        matches_away = predictor.get_team_recent_matches(65)   # Man City
        home_stats   = predictor.team_goal_rates(57, matches_home)
        away_stats   = predictor.team_goal_rates(65, matches_away)

        self.assertNotEqual(home_stats, {"attack_home": 1.3, "defense_home": 1.1, "attack_away": 1.1, "defense_away": 1.3},
                            "Arsenal should not fall back to default stats")

        fixture = {
            "id": 99999, "competition": "Premier League", "competition_code": "PL",
            "utc_date": "2026-08-23T14:00:00Z",
            "home_id": 57, "home_name": "Arsenal FC",
            "away_id": 65, "away_name": "Manchester City FC",
        }
        pred = predictor.predict_match(home_stats, away_stats, "PL", fixture)
        msg  = predictor.format_prediction_message(fixture, pred)

        print(f"\n  [live] Arsenal vs Man City prediction:")
        print(f"    Pick: {pred['pick_name']} ({pred['confidence_label']}, {pred['stake_units']})")
        print(f"    Probabilities: H {pred['home_win_prob']*100:.0f}% | D {pred['draw_prob']*100:.0f}% | A {pred['away_win_prob']*100:.0f}%")
        print(f"    Over 2.5: {pred['over_2_5_prob']*100:.1f}%  BTTS: {pred['btts_yes_prob']*100:.1f}%")
        print(f"    Expected Score: {pred['score'][0]}-{pred['score'][1]}")

        self.assertGreater(pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"], 0.99)

        with tempfile.TemporaryDirectory() as tmpdir:
            with unittest.mock.patch("predictor.MATCH_CARDS_DIR", tmpdir):
                card = predictor.create_match_card_image(fixture, pred)
            self.assertIsNotNone(card)
            self.assertTrue(os.path.exists(card))
            print(f"    Card: {card}")

    def test_telegram_message_live(self):
        """Send a real test message to Telegram. Pass --telegram to enable."""
        if not getattr(self, "_telegram", False):
            self.skipTest("Telegram live test skipped — run with --live --telegram")
        if not predictor.TELEGRAM_BOT_TOKEN or not predictor.TELEGRAM_CHAT_IDS:
            self.skipTest("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_IDS not set")

        test_msg = (
            "🧪 <b>Football Prophet — Test Message</b>\n\n"
            "This is an automated test from the test suite.\n"
            "If you see this, Telegram delivery is working! ✅"
        )
        ok = predictor.send_telegram_message(test_msg)
        self.assertTrue(ok, "Telegram send failed")
        print("\n  [live] Telegram test message sent successfully.")


# ═════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Football Prophet Test Suite")
    parser.add_argument("--live",     action="store_true", help="Run live API integration tests (requires .env)")
    parser.add_argument("--telegram", action="store_true", help="Send a real Telegram test message (requires --live)")
    args, remaining = parser.parse_known_args()

    if args.live:
        TestLiveAPI._live = True
    if args.telegram:
        TestLiveAPI._telegram = True

    print("=" * 60)
    print("  Football Prophet Test Suite")
    print(f"  Mode: {'LIVE + UNIT' if args.live else 'UNIT ONLY (offline)'}")
    if args.telegram:
        print("  Telegram: ENABLED")
    print("=" * 60)
    print()
    if not args.live:
        print("  ⚠️  NOTE: Any dates/fixtures printed during tests are SYNTHETIC")
        print("  ⚠️  test data only. No real API calls or Telegram messages are")
        print("  ⚠️  made in offline mode. Real fixtures come from football-data.org.")
        print()

    loader = unittest.TestLoader()
    suites = [
        loader.loadTestsFromTestCase(TestHelpers),
        loader.loadTestsFromTestCase(TestSentFixtures),
        loader.loadTestsFromTestCase(TestPredictionModel),
        loader.loadTestsFromTestCase(TestPredictionHistory),
        loader.loadTestsFromTestCase(TestSeasonAnnouncements),
        loader.loadTestsFromTestCase(TestMessageFormatting),
        loader.loadTestsFromTestCase(TestCardGeneration),
    ]
    if args.live:
        suites.append(loader.loadTestsFromTestCase(TestLiveAPI))

    suite  = unittest.TestSuite(suites)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print()
    print("=" * 60)
    if result.wasSuccessful():
        print(f"  ✅ ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  ❌ {len(result.failures)} FAILURE(S) / {len(result.errors)} ERROR(S) out of {result.testsRun} tests")
    print("=" * 60)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
