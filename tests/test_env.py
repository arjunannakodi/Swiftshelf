import pytest
import gymnasium as gym
from env.environment import InventoryEnv


# ================================================================== #
# 1. Reset returns valid observation
# ================================================================== #
def test_reset_returns_valid_obs():
    env = InventoryEnv()
    obs, info = env.reset()

    required_keys = {
        "item_states", "pending_orders", "budget_remaining",
        "near_expiry_count", "steps_elapsed", "expired_count",
    }
    for key in required_keys:
        assert key in obs, f"Missing obs key: {key}"

    assert isinstance(obs["item_states"], list)
    assert len(obs["item_states"]) == 20, "Expected 20 items"
    assert obs["budget_remaining"] == 1000.0
    assert obs["steps_elapsed"] == 0
    assert isinstance(info, dict)


# ================================================================== #
# 2. Step returns 5-tuple
# ================================================================== #
def test_step_returns_5_tuple():
    env = InventoryEnv()
    env.reset()
    result = env.step(5)   # hold

    assert isinstance(result, tuple), "step() must return a tuple"
    assert len(result) == 5, "step() must return exactly 5 values"

    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ================================================================== #
# 3. All 6 actions execute without exception
# ================================================================== #
def test_valid_actions_0_to_5():
    env = InventoryEnv()
    for action in range(6):
        env.reset()
        try:
            env.step(action)
        except Exception as exc:
            pytest.fail(f"Action {action} raised an unexpected exception: {exc}")


# ================================================================== #
# 4. Invalid action raises ValueError
# ================================================================== #
def test_invalid_action_raises():
    env = InventoryEnv()
    env.reset()
    with pytest.raises(ValueError, match="out of range"):
        env.step(99)


# ================================================================== #
# 5. Episode terminates on expired dispatch
# ================================================================== #
def test_episode_terminates():
    env = InventoryEnv()
    env.reset()

    # Force all items to expired state
    for it in env.items:
        it["expiry_days"] = 0.0
        it["stock"] = 50  # ensure plenty of stock

    # Ensure a pending order exists pointing to item 0
    env.pending_orders = [{"item_id": 0, "quantity": 1, "deadline": 10}]

    terminated = False
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(3)  # dispatch
        assert reward == pytest.approx(-500.0) or terminated, \
            "Dispatching expired item must give -500 or terminate"
        if terminated:
            break

    assert terminated is True, "Environment must terminate on expired dispatch"


# ================================================================== #
# 6. Reward is not always zero
# ================================================================== #
def test_reward_not_always_zero():
    env = InventoryEnv()
    env.reset()
    rewards = []
    for _ in range(20):
        _, reward, terminated, truncated, _ = env.step(5)  # hold
        rewards.append(reward)
        if terminated or truncated:
            break
    assert any(r != 0.0 for r in rewards), \
        "At least one step must produce a non-zero reward"


# ================================================================== #
# 7. Reset clears state back to initial values
# ================================================================== #
def test_reset_clears_state():
    env = InventoryEnv()
    env.reset()

    # Mutate state
    env.budget = 42.0
    env.steps  = 99
    env.orders_completed = 7

    # Reset must restore
    obs, _ = env.reset()
    assert obs["budget_remaining"] == 1000.0, "Budget must reset to 1000"
    assert obs["steps_elapsed"] == 0,         "steps must reset to 0"
    assert env.orders_completed == 0,         "orders_completed must reset"


# ================================================================== #
# 8. gymnasium.Env compliance
# ================================================================== #
def test_gymnasium_env_compliance():
    """Check env has required gymnasium attributes."""
    from env.environment import InventoryEnv
    env = InventoryEnv()
    assert hasattr(env, 'action_space')
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')


# ================================================================== #
# 9. FEFO dispatch bonus is conditional, not always given
# ================================================================== #
def test_fefo_bonus_is_conditional():
    """FEFO bonus only triggers when oldest stock is picked."""
    from env.environment import InventoryEnv
    env = InventoryEnv()
    env.reset()

    # Case 1: Pick the FEFO item
    env.items = [
        {"id": 0, "stock": 10, "expiry_days": 2.0,  "price": 10.0},
        {"id": 1, "stock": 10, "expiry_days": 25.0, "price": 10.0},
    ]
    env.pending_orders = [{"item_id": 0, "quantity": 1, "deadline": 10}]
    _, r_fefo, _, _, _ = env.step(0) # pick item 0

    # Case 2: Pick a non-FEFO item
    env.reset()
    env.items = [
        {"id": 0, "stock": 10, "expiry_days": 2.0,  "price": 10.0},
        {"id": 1, "stock": 10, "expiry_days": 25.0, "price": 10.0},
    ]
    env.pending_orders = [{"item_id": 1, "quantity": 1, "deadline": 10}]
    _, r_non_fefo, _, _, _ = env.step(0) # pick item 1

    assert r_fefo > r_non_fefo, f"Expected FEFO reward {r_fefo} > non-FEFO {r_non_fefo}"


# ================================================================== #
# 10. Seed reproducibility — same seed → identical observations
# ================================================================== #
def test_seed_reproducibility():
    """Two envs reset with the same seed must produce identical observations."""
    env_a = InventoryEnv()
    env_b = InventoryEnv()

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)

    # Budget and step counters must match
    assert obs_a["budget_remaining"] == obs_b["budget_remaining"]
    assert obs_a["steps_elapsed"] == obs_b["steps_elapsed"]
    assert obs_a["near_expiry_count"] == obs_b["near_expiry_count"]
    assert obs_a["expired_count"] == obs_b["expired_count"]

    # All 20 items must be identical
    assert len(obs_a["item_states"]) == len(obs_b["item_states"]) == 20
    for item_a, item_b in zip(obs_a["item_states"], obs_b["item_states"]):
        assert item_a["id"] == item_b["id"]
        assert item_a["stock"] == item_b["stock"]
        assert item_a["expiry_days"] == pytest.approx(item_b["expiry_days"])
        assert item_a["price"] == pytest.approx(item_b["price"])

    # Pending orders must match
    assert len(obs_a["pending_orders"]) == len(obs_b["pending_orders"])
    for o_a, o_b in zip(obs_a["pending_orders"], obs_b["pending_orders"]):
        assert o_a["item_id"] == o_b["item_id"]
        assert o_a["quantity"] == o_b["quantity"]
        assert o_a["deadline"] == o_b["deadline"]


# ================================================================== #
# 11. Different seeds → different observations (randomness check)
# ================================================================== #
def test_seed_different_produces_different_obs():
    """Two different seeds must not produce the same item states."""
    env = InventoryEnv()

    obs_42, _ = env.reset(seed=42)
    obs_99, _ = env.reset(seed=99)

    # At least one item must differ (expiry_days or stock)
    diffs = [
        item_a["expiry_days"] != item_b["expiry_days"]
        or item_a["stock"] != item_b["stock"]
        for item_a, item_b in zip(obs_42["item_states"], obs_99["item_states"])
    ]
    assert any(diffs), "Different seeds must produce at least one different item state"
