import pytest
from rl_compat import rsl5_config


def test_deprecated_fields_removed_without_changing_distribution_or_source():
    config = {
        "actor": {
            "stochastic": True,
            "init_noise_std": 1,
            "noise_std_type": "scalar",
            "state_dependent_std": False,
            "distribution_cfg": {"init_std": 0.35},
        },
        "critic": {"stochastic": False, "distribution_cfg": None},
    }
    converted = rsl5_config(config, "5.0.1")
    assert converted["actor"] == {"distribution_cfg": {"init_std": 0.35}}
    assert converted["critic"] == {"distribution_cfg": None}
    assert config["actor"]["stochastic"] is True


def test_unknown_runtime_does_not_silently_drop_fields():
    with pytest.raises(ValueError, match="Recheck"):
        rsl5_config({}, "6.0.0")


def test_stochastic_actor_required():
    with pytest.raises(ValueError, match="actor"):
        rsl5_config({"actor": {}, "critic": {}}, "5.0.1")
