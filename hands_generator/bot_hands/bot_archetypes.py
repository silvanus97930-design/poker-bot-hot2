"""
Named sandbox bot archetypes for synthetic training data.

Each name maps to a BotProfile tuned for recognizable behavior patterns.
Timing labels (e.g. human_like_timing_bot) are behavioral proxies only — the
simulator does not model real wall-clock delay; downstream datasets can add
timing features separately.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from hands_generator.bot_hands.sandbox_poker_bot import BotProfile

# Stable public ordering for CLI / manifests
TRAINING_ARCHETYPE_NAMES: List[str] = [
    "random_bot",
    "tight_bot",
    "loose_bot",
    "aggressive_bot",
    "passive_bot",
    "fast_bot",
    "consistent_bot",
    "pattern_bot",
    "gto_bot",
    "noisy_smart_bot",
    "mixed_bot",
    "multi_table_bot",
    "human_like_timing_bot",
]


def _archetype_presets() -> Dict[str, BotProfile]:
    return {
        "random_bot": BotProfile(
            name="random_bot",
            random_mode=True,
            tightness=0.55,
            aggression=0.55,
            bluff_freq=0.08,
        ),
        "tight_bot": BotProfile(
            name="tight_bot",
            tightness=0.74,
            aggression=0.48,
            bluff_freq=0.02,
            preflop_defend_bias=-0.22,
            postflop_continue_bias=-0.18,
            trap_frequency=-0.14,
            max_risk_fraction_of_stack=0.14,
        ),
        "loose_bot": BotProfile(
            name="loose_bot",
            tightness=0.36,
            aggression=0.54,
            bluff_freq=0.09,
            preflop_defend_bias=0.12,
            postflop_continue_bias=0.08,
            trap_frequency=0.02,
            max_risk_fraction_of_stack=0.22,
        ),
        "aggressive_bot": BotProfile(
            name="aggressive_bot",
            tightness=0.52,
            aggression=0.88,
            bluff_freq=0.11,
            preflop_defend_bias=0.04,
            postflop_continue_bias=0.06,
            trap_frequency=-0.04,
            bet_pot_fraction_small=0.38,
            bet_pot_fraction_medium=0.62,
            bet_pot_fraction_large=0.92,
        ),
        "passive_bot": BotProfile(
            name="passive_bot",
            tightness=0.58,
            aggression=0.30,
            bluff_freq=0.03,
            preflop_defend_bias=-0.12,
            postflop_continue_bias=-0.14,
            trap_frequency=0.06,
        ),
        "fast_bot": BotProfile(
            name="fast_bot",
            tightness=0.62,
            aggression=0.46,
            bluff_freq=0.04,
            preflop_defend_bias=-0.28,
            postflop_continue_bias=-0.32,
            trap_frequency=-0.20,
            max_risk_fraction_of_stack=0.12,
        ),
        "consistent_bot": BotProfile(
            name="consistent_bot",
            tightness=0.56,
            aggression=0.54,
            bluff_freq=0.05,
            preflop_defend_bias=-0.06,
            postflop_continue_bias=-0.05,
            trap_frequency=-0.06,
            bet_pot_fraction_small=0.34,
            bet_pot_fraction_medium=0.54,
            bet_pot_fraction_large=0.78,
            tilt_factor=0.0,
            decision_noise=0.0,
            pattern_repeat_probability=0.0,
        ),
        "pattern_bot": BotProfile(
            name="pattern_bot",
            tightness=0.54,
            aggression=0.56,
            bluff_freq=0.06,
            pattern_repeat_probability=0.52,
            preflop_defend_bias=-0.04,
            postflop_continue_bias=-0.02,
        ),
        "gto_bot": BotProfile(
            name="gto_bot",
            tightness=0.52,
            aggression=0.50,
            bluff_freq=0.065,
            preflop_defend_bias=0.0,
            postflop_continue_bias=0.0,
            trap_frequency=-0.02,
            bet_pot_fraction_small=0.33,
            bet_pot_fraction_medium=0.50,
            bet_pot_fraction_large=0.72,
            max_risk_fraction_of_stack=0.19,
        ),
        "noisy_smart_bot": BotProfile(
            name="noisy_smart_bot",
            tightness=0.58,
            aggression=0.58,
            bluff_freq=0.06,
            preflop_defend_bias=-0.08,
            postflop_continue_bias=-0.06,
            decision_noise=0.11,
        ),
        "mixed_bot": BotProfile(
            name="mixed_bot",
            tightness=0.50,
            aggression=0.62,
            bluff_freq=0.085,
            preflop_defend_bias=0.02,
            postflop_continue_bias=-0.04,
            trap_frequency=-0.04,
            decision_noise=0.04,
        ),
        "multi_table_bot": BotProfile(
            name="multi_table_bot",
            tightness=0.60,
            aggression=0.44,
            bluff_freq=0.05,
            preflop_defend_bias=-0.26,
            postflop_continue_bias=-0.36,
            decision_noise=0.07,
            max_risk_fraction_of_stack=0.13,
        ),
        "human_like_timing_bot": BotProfile(
            name="human_like_timing_bot",
            tightness=0.55,
            aggression=0.52,
            bluff_freq=0.055,
            preflop_defend_bias=-0.08,
            postflop_continue_bias=-0.06,
            trap_frequency=0.04,
            decision_noise=0.035,
            bet_pot_fraction_small=0.31,
            bet_pot_fraction_medium=0.52,
            bet_pot_fraction_large=0.82,
        ),
    }


def archetype_profile(name: str) -> BotProfile:
    """Return a fresh profile for the given archetype name."""
    presets = _archetype_presets()
    if name not in presets:
        known = ", ".join(TRAINING_ARCHETYPE_NAMES)
        raise KeyError(f"Unknown archetype {name!r}. Expected one of: {known}")
    return replace(presets[name])


def all_training_archetype_profiles() -> List[BotProfile]:
    """All named training archetypes in stable order."""
    return [archetype_profile(n) for n in TRAINING_ARCHETYPE_NAMES]


# Legacy-style names still used in some scripts / tests
LEGACY_PROFILE_ALIASES: Dict[str, str] = {
    "balanced": "consistent_bot",
    "tight_aggressive": "aggressive_bot",
    "loose_aggressive": "mixed_bot",
    "tight_passive": "tight_bot",
    "loose_passive": "loose_bot",
}


def resolve_profile_name(name: str) -> BotProfile:
    """Map legacy profile labels to archetypes when needed."""
    key = LEGACY_PROFILE_ALIASES.get(name, name)
    return archetype_profile(key)
