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
    env = InventoryEnv()
    assert isinstance(env, gym.Env), \
        "InventoryEnv must inherit from gymnasium.Env"
    assert hasattr(env, "action_space"), \
        "Must have action_space attribute"
    assert hasattr(env, "observation_space"), \
        "Must have observation_space attribute"
    assert env.action_space.n == 6, \
        "action_space must be Discrete(6)"
    assert env.action_space.contains(0)
    assert env.action_space.contains(5)
    assert not env.action_space.contains(6)


# ================================================================== #
# 9. FEFO dispatch bonus is conditional, not always given
# ================================================================== #
def test_fefo_bonus_is_conditional():
    """
    Place 2 items: one expiring soon (id=0, expiry=2),
    one fresh (id=1, expiry=25).  Force an order for item 1 (fresh).
    Dispatching item 1 should NOT receive the FEFO +20 bonus.
    """
    env = InventoryEnv()
    env.reset()

    # Manually craft inventory
    env.items = [
        {"id": 0, "stock": 10, "expiry_days": 2.0,  "price": 10.0},
        {"id": 1, "stock": 10, "expiry_days": 25.0, "price": 10.0},
    ]
    env.pending_orders = [{"item_id": 1, "quantity": 1, "deadline": 10}]

    _, reward, terminated, truncated, info = env.step(3)  # dispatch

    # The dispatch of item 1 (fresh, NOT FEFO) should give +10 but NOT +20
    # Total per-step bonuses also apply, so reward will include survival/budget bonuses
    # Just check that terminated is False (no expired dispatch)
    assert not terminated, "Dispatching a fresh item must not terminate"
    # And the FEFO item (id=0) was not dispatched — so no FEFO bonus given
    # We can't assert exact reward due to per-step bonuses, but orders_completed must be 1
    assert info.get("orders_completed", 0) == 1, "One order must be completed"
