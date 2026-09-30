from itertools import pairwise

import numpy as np
from rewards import PPO_GAMMA, progress_reward


def reward(old, new, success=False, invalid=False):
    return float(progress_reward(np.array(old), np.array(new), np.array(success), np.array(invalid)))


def test_stationary_alignment_does_not_pay_a_hover_bonus():
    assert reward(2.0, 2.0) < 0
    assert reward(5.9, 5.9) < 0


def test_seating_is_better_than_stalling_or_invalid_contact():
    assert reward(5.9, 6.0, success=True) > reward(5.9, 5.9)
    assert reward(5.9, 5.9) > reward(5.9, 6.0, invalid=True)


def test_potential_shaping_telescopes():
    potentials = [1.0, 2.0, 3.0, 5.0]
    value = sum(PPO_GAMMA**i * (reward(a, b) + 0.01) for i, (a, b) in enumerate(pairwise(potentials)))
    assert np.isclose(value, -potentials[0] + PPO_GAMMA**3 * potentials[-1])
