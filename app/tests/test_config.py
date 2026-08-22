from app.core.config import merged_config


def test_merged_config_is_deep_and_non_mutating():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = merged_config(base, {"a": {"x": 9}})
    assert out == {"a": {"x": 9, "y": 2}, "b": 3}
    assert base["a"]["x"] == 1
