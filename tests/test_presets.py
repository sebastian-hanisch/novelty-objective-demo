"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import nobj_constants as C
import nobj_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["level"] in C.LEVELS and C.GENS_MIN <= preset["gens"] <= C.GENS_MAX and P.snap_gens(preset["gens"]) == preset["gens"]
        assert preset["selection"] in C.LIVE_SELECTIONS and C.K_MIN <= preset["k"] <= C.K_MAX
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Falle: etwas Ziel dazu"]["selection"] == "w25" and C.PRESETS["Zu viel Ziel"]["selection"] == "w90"
    assert C.PRESETS["Pareto"]["selection"] == "pareto" and C.PRESETS["Schlangenlinie"]["level"] == 4


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Falle: etwas Ziel dazu"]
    assert (p["level"], p["gens"], p["seed"], p["selection"], p["k"]) == (C.DEFAULT_LEVEL, C.DEFAULT_GENS, C.DEFAULT_SEED, C.DEFAULT_SELECTION, C.K_DEFAULT)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gens_slider") == (10, 200) and P.bounds("level_select") == (0, 4) and P.bounds("k_slider") == (1, 30)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS) == 5


@pytest.mark.parametrize("value,expected", [(0, 10), (14, 10), (16, 20), (104, 100), (106, 110), (999, 200)])
def test_generations_snap_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_gens(value) == expected


@pytest.mark.parametrize("value", C.LIVE_SELECTIONS)
def test_selection_caster_accepts_every_live_selection(value):
    assert P.SETTING_SPECS["selection_select"].caster(value) == value


@pytest.mark.parametrize("value", ["fitness", "novelty", "zufall", "", 3, None, ["w25"]])
def test_selection_caster_refuses_everything_else(value):
    """Fitness und reine Novelty sind der Bezug beziehungsweise nur in der Studie, nicht im Live-Lauf wählbar."""
    with pytest.raises(ValueError):
        P.SETTING_SPECS["selection_select"].caster(value)


def test_formatters():
    assert C.fmt_int(40000) == "40 000" and C.fmt_med(None) == "–" and C.fmt_med(988.5) == "988.5" and C.fmt_med(1195.0) == "1 195" and C.fmt_med(1308.5) == "1 308.5"
