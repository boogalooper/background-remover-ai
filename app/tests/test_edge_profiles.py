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


def test_custom_profile_names_are_separate_from_standard_profiles():
    from app.gui.main_window import MainWindow, USER_PROFILE_PREFIX

    window = object.__new__(MainWindow)
    window.custom_edge_profiles = {
        "Волосы": {
            "mask": {
                "black_point": 0.02, "white_point": 0.98, "gamma": 1.0,
                "expand_pixels": 0, "feather_radius": 0.0, "guided_refine": False,
                "guided_max_long_edge": 4096, "guided_radius": 8, "guided_blend": 0.35,
            },
            "cutout": {"decontaminate": True, "decontam_strength": 0.5},
        }
    }
    values = MainWindow._edge_profile_values(window)
    assert values[:4] == (PROFILE_RECOMMENDED, PROFILE_NATURAL, PROFILE_CLEAN, PROFILE_NONE)
    assert f"{USER_PROFILE_PREFIX}Волосы" in values
    assert values[-1] == PROFILE_CUSTOM


def test_invalid_custom_presets_are_ignored():
    from app.gui.main_window import MainWindow

    valid = {
        "mask": {
            "black_point": 0.02, "white_point": 0.98, "gamma": 1.0,
            "expand_pixels": 0, "feather_radius": 0.0, "guided_refine": False,
            "guided_max_long_edge": 4096, "guided_radius": 8, "guided_blend": 0.35,
        },
        "cutout": {"decontaminate": True, "decontam_strength": 0.5},
    }
    result = MainWindow._validated_custom_profiles({"OK": valid, "Broken": {"mask": {}}})
    assert list(result) == ["OK"]
