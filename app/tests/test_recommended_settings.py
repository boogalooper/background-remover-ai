from app.gui.main_window import MainWindow
from app.models.catalog import model_recommended_overrides


def test_recommended_settings_equal_when_identical():
    recommended = model_recommended_overrides("birefnet_hr_matting")
    current = {
        "mask": dict(recommended["mask"]),
        "cutout": dict(recommended["cutout"]),
    }
    assert MainWindow._settings_equal(current, recommended)


def test_recommended_settings_detects_any_changed_quality_value():
    recommended = model_recommended_overrides("birefnet_hr_matting")
    current = {
        "mask": dict(recommended["mask"]),
        "cutout": dict(recommended["cutout"]),
    }
    current["cutout"]["decontam_strength"] += 0.05
    assert not MainWindow._settings_equal(current, recommended)
