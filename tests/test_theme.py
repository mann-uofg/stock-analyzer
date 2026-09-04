"""Invariants for the design system.

The redesign removed a mesh of radial gradients and a blur filter on every
surface. Both are the kind of thing that creeps back one rule at a time, so
the absence is asserted rather than assumed.
"""

from __future__ import annotations

import pytest

from analyzer import store
from views.theme import PALETTES, css

MODES = list(PALETTES)


class TestNoOrnament:
    @pytest.mark.parametrize("mode", MODES)
    def test_no_backdrop_blur(self, mode):
        # Filtering the backdrop cost a repaint on every scroll and, over a
        # flat ground, bought nothing but muddier text.
        assert "backdrop-filter" not in css(mode)

    @pytest.mark.parametrize("mode", MODES)
    def test_no_decorative_gradients(self, mode):
        sheet = css(mode)
        assert "radial-gradient" not in sheet
        assert "linear-gradient" not in sheet

    @pytest.mark.parametrize("mode", MODES)
    def test_the_one_gradient_left_is_the_progress_dial(self, mode):
        # A conic sweep is the arc of a dial: it encodes a value rather than
        # decorating a surface.
        assert css(mode).count("conic-gradient") == 1

    @pytest.mark.parametrize("mode", MODES)
    def test_no_dangling_blur_token(self, mode):
        # The --blur variable is gone; a leftover reference would silently
        # invalidate whatever rule still used it.
        assert "var(--blur)" not in css(mode)


class TestPalettes:
    def test_dark_is_the_default(self):
        assert css() == css("dark")
        assert store.DEFAULT_SETTINGS["appearance"] == "dark"

    def test_both_modes_define_the_same_tokens(self):
        # A token present in one mode and missing in the other would raise a
        # KeyError only when someone switched appearance.
        assert set(PALETTES["dark"]) == set(PALETTES["light"])

    @pytest.mark.parametrize("mode", MODES)
    def test_every_token_is_interpolated(self, mode):
        # An unreplaced placeholder means a variable resolves to nothing and
        # the rule using it silently drops.
        sheet = css(mode)
        assert "{p[" not in sheet and "None" not in sheet.split("/*")[0]

    @pytest.mark.parametrize("mode", MODES)
    def test_surfaces_are_opaque(self, mode):
        # Translucent cards were what made identical panels read differently
        # depending on what happened to sit behind them.
        surface = PALETTES[mode]["glass"]
        assert not surface.startswith("rgba"), f"{mode} surface is translucent"

    def test_unknown_mode_falls_back_to_dark(self):
        assert css("chartreuse") == css("dark")


class TestThemeEpochMigration:
    """A preference from the old look is not a choice about the new one."""

    @pytest.fixture
    def session(self, monkeypatch):
        state: dict = {}
        monkeypatch.setattr(store, "is_shared_host", lambda: True)
        monkeypatch.setattr(store, "_session_state", lambda: state)
        return state

    def test_a_stale_light_preference_is_dropped(self, session):
        session["_store_settings"] = {"appearance": "light", "theme_epoch": 0}
        assert store.load_settings()["appearance"] == "dark"

    def test_a_current_choice_is_respected(self, session):
        session["_store_settings"] = {"appearance": "light",
                                      "theme_epoch": store.THEME_EPOCH}
        assert store.load_settings()["appearance"] == "light"

    def test_settings_with_no_epoch_at_all_get_the_new_default(self, session):
        session["_store_settings"] = {"appearance": "light"}
        assert store.load_settings()["appearance"] == "dark"

    def test_other_settings_survive_the_migration(self, session):
        session["_store_settings"] = {"appearance": "light", "risk_pct": 2.5}
        loaded = store.load_settings()
        assert loaded["risk_pct"] == 2.5
        assert loaded["appearance"] == "dark"
