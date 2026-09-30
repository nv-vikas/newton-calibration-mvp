"""Progress shaping: do not pay a policy repeatedly for hovering near the hole."""

PPO_GAMMA = 0.99


def progress_reward(previous_potential, potential, success, invalid):
    # Absorbing true terminal states have zero potential. Time-limit truncation
    # is not a true terminal and remains eligible for value bootstrapping.
    continuing = ~(success | invalid)
    shaped = PPO_GAMMA * potential * continuing - previous_potential
    return shaped + 20.0 * success - 10.0 * invalid - 0.01
