"""Narrow adapter for Isaac Lab beta configuration and pinned RSL-RL 5.0.1."""

from copy import deepcopy


def rsl5_config(config, version):
    if version != "5.0.1":
        raise ValueError("Recheck the PPO configuration adapter before changing RSL-RL versions")
    result = deepcopy(config)
    # Isaac Lab keeps deprecated fields for older RSL-RL versions. The v5
    # MLPModel constructor accepts distribution_cfg instead of these fields.
    for name in ("actor", "critic"):
        model = result[name]
        for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            model.pop(key, None)
    if not result["actor"].get("distribution_cfg"):
        raise ValueError("PPO actor requires an explicit stochastic distribution")
    if result["critic"].get("distribution_cfg") is not None:
        raise ValueError("Value critic must remain deterministic")
    return result
