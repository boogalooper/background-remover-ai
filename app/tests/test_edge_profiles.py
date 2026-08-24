from app.gui.main_window import (
    EDGE_PROFILES,
    PROFILE_CLEAN,
    PROFILE_CUSTOM,
    PROFILE_NATURAL,
    PROFILE_NONE,
    PROFILE_RECOMMENDED,
)


def test_edge_profiles_include_model_recommended_and_custom_states():
    assert list(EDGE_PROFILES) == [
        PROFILE_RECOMMENDED,
        PROFILE_NATURAL,
        PROFILE_CLEAN,
        PROFILE_NONE,
        PROFILE_CUSTOM,
    ]
    assert EDGE_PROFILES[PROFILE_RECOMMENDED] is None
    assert EDGE_PROFILES[PROFILE_CUSTOM] is None


def test_static_edge_profiles_are_complete_quality_presets():
    required_mask = {
        "black_point", "white_point", "gamma", "expand_pixels", "feather_radius",
        "guided_refine", "guided_max_long_edge", "guided_radius", "guided_blend",
    }
    required_cutout = {"decontaminate", "decontam_strength"}
    for profile in (PROFILE_NATURAL, PROFILE_CLEAN, PROFILE_NONE):
        settings = EDGE_PROFILES[profile]
        assert set(settings["mask"]) == required_mask
        assert set(settings["cutout"]) == required_cutout

    assert EDGE_PROFILES[PROFILE_NONE]["cutout"]["decontaminate"] is False
