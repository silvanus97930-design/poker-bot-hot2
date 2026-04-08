import numpy as np

_STREETS = ("preflop", "flop", "turn", "river")
_ACTION_TYPES = ("call", "check", "raise", "fold", "bet")


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def encode_hand(hand):
    """Encode one sanitized hand into a fixed-size feature vector."""
    metadata = hand.get("metadata", {}) if isinstance(hand, dict) else {}
    players = hand.get("players", []) if isinstance(hand, dict) else []
    actions = hand.get("actions", []) if isinstance(hand, dict) else []

    features = []

    # 1) Metadata basics.
    max_seats = _safe_int(metadata.get("max_seats"), default=0)
    sb = _safe_float(metadata.get("sb"), default=0.0)
    bb = _safe_float(metadata.get("bb"), default=0.0)
    ante = _safe_float(metadata.get("ante"), default=0.0)
    blind_ratio = sb / bb if bb > 0 else 0.0
    ante_bb = ante / bb if bb > 0 else 0.0
    features.extend([max_seats, sb, bb, ante, blind_ratio, ante_bb])

    # 2) Player count and stack stats.
    player_count = len(players)
    stacks = [_safe_float(p.get("starting_stack", 0.0), default=0.0) for p in players if isinstance(p, dict)]
    if stacks:
        stacks_arr = np.array(stacks, dtype=np.float32)
        stack_mean = float(stacks_arr.mean())
        stack_std = float(stacks_arr.std())
        stack_min = float(stacks_arr.min())
        stack_max = float(stacks_arr.max())
    else:
        stack_mean = stack_std = stack_min = stack_max = 0.0
    features.extend([player_count, stack_mean, stack_std, stack_min, stack_max])

    # 3) Action counts and normalized amount / pot aggregates.
    action_count = len(actions)
    action_type_counts = {name: 0 for name in _ACTION_TYPES}
    amount_values = []
    pot_before_values = []
    pot_after_values = []
    actor_seats = []
    street_counts = {name: 0 for name in _STREETS}

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type", "")).lower()
        street = str(action.get("street", "")).lower()
        actor_seat = _safe_int(action.get("actor_seat"), default=0)
        amount = _safe_float(action.get("normalized_amount_bb", action.get("amount", 0.0)), default=0.0)
        pot_before = _safe_float(action.get("pot_before", 0.0), default=0.0)
        pot_after = _safe_float(action.get("pot_after", 0.0), default=0.0)

        if action_type in action_type_counts:
            action_type_counts[action_type] += 1
        if street in street_counts:
            street_counts[street] += 1

        amount_values.append(amount)
        pot_before_values.append(pot_before)
        pot_after_values.append(pot_after)
        actor_seats.append(actor_seat)

    denom = float(max(1, action_count))
    features.extend([action_count] + [action_type_counts[name] / denom for name in _ACTION_TYPES])

    if amount_values:
        amount_arr = np.array(amount_values, dtype=np.float32)
        features.extend([float(amount_arr.mean()), float(amount_arr.std()), float(amount_arr.max())])
    else:
        features.extend([0.0, 0.0, 0.0])

    if pot_before_values:
        pot_before_arr = np.array(pot_before_values, dtype=np.float32)
        features.extend([float(pot_before_arr.mean()), float(pot_before_arr.max())])
    else:
        features.extend([0.0, 0.0])

    if pot_after_values:
        pot_after_arr = np.array(pot_after_values, dtype=np.float32)
        features.extend([float(pot_after_arr.mean()), float(pot_after_arr.max())])
    else:
        features.extend([0.0, 0.0])

    # 4) Actor seat distribution features.
    if actor_seats:
        seat_arr = np.array(actor_seats, dtype=np.float32)
        features.extend([float(seat_arr.mean()), float(seat_arr.std()), float(seat_arr.max())])
    else:
        features.extend([0.0, 0.0, 0.0])

    # 5) Street distribution.
    features.extend([street_counts[name] / denom for name in _STREETS])

    return np.array(features, dtype=np.float32)


def encode_chunk(chunk):
    # Backward-compatible wrapper. Each sequence timestep is one hand.
    return encode_hand(chunk)