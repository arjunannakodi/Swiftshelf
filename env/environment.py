import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Any, Optional, Tuple


class InventoryEnv(gym.Env):
    """
    SwiftShelf++ Inventory Decision Environment
    OpenEnv-compliant, gymnasium.Env based.
    
    Observation Space:
        Dict with keys: item_states, pending_orders, budget_remaining,
        near_expiry_count, steps_elapsed, expired_count

    Action Space:
        Discrete(6):
            0: pick_item   — FEFO-aware item pick for first pending order
            1: restock     — Add stock (costs budget)
            2: apply_discount — Discount near-expiry items
            3: dispatch_order — Fulfill first pending order
            4: batch_pick  — Dispatch up to 3 orders at once
            5: hold        — Do nothing
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    # Class-level constants
    # ------------------------------------------------------------------ #
    NUM_ITEMS     = 20
    MAX_STEPS     = 200
    INITIAL_BUDGET = 1000.0
    RESTOCK_COST  = 50.0
    RESTOCK_STOCK = 20

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # ---- Action Space -------------------------------------------- #
        self.action_space = spaces.Discrete(6)

        # ---- Observation Space --------------------------------------- #
        # Flat representation for gymnasium compliance.
        # The actual step/reset return is a rich dict (for API / grader).
        self.observation_space = spaces.Dict({
            "budget_remaining": spaces.Box(
                low=0.0, high=float("inf"), shape=(1,), dtype=np.float32),
            "near_expiry_count": spaces.Discrete(self.NUM_ITEMS + 1),
            "steps_elapsed": spaces.Discrete(self.MAX_STEPS + 1),
            "expired_count": spaces.Discrete(self.NUM_ITEMS + 1),
            # Flattened item stats: [stock, expiry_days, price] × NUM_ITEMS
            "item_stats_flat": spaces.Box(
                low=0.0, high=1000.0,
                shape=(self.NUM_ITEMS * 3,), dtype=np.float32),
            # Flattened pending order stats: [item_id, quantity, deadline] × 30
            "order_stats_flat": spaces.Box(
                low=0.0, high=1000.0,
                shape=(30 * 3,), dtype=np.float32),
        })

        # Initialise (will be overwritten by reset())
        self.items: List[Dict] = []
        self.pending_orders: List[Dict] = []
        self.budget: float = self.INITIAL_BUDGET
        self.steps: int = 0
        self.orders_completed: int = 0
        self.expired_count: int = 0
        self.picked_item_id: Optional[int] = None

    # ================================================================== #
    # Gymnasium API
    # ================================================================== #

    @property
    def orders(self):
        return self.pending_orders

    @orders.setter
    def orders(self, value):
        self.pending_orders = value

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Reset and return (observation, info)."""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.items = [
            {
                "id": i,
                "stock": random.randint(5, 50),
                # Mix of near-expiry (1-5) and fresh (6-30) items
                "expiry_days": float(
                    random.randint(1, 5) if i < 4 else random.randint(6, 30)
                ),
                "price": round(random.uniform(10.0, 50.0), 2),
            }
            for i in range(self.NUM_ITEMS)
        ]
        self.pending_orders = []
        for _ in range(3):
            self._generate_order()

        self.budget = self.INITIAL_BUDGET
        self.steps = 0
        self.orders_completed = 0
        self.expired_count = 0
        self.picked_item_id = None

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Apply action and advance one step.

        Returns:
            obs        — rich dict observation
            reward     — float reward signal
            terminated — True on critical failure (dispatched expired item)
            truncated  — True when max_steps reached
            info       — dict with orders_completed, expired_count, step
        """
        if not (0 <= int(action) < 6):
            raise ValueError(
                f"Action {action} is out of range. Valid: 0-5."
            )
        action = int(action)

        reward = 0.0
        terminated = False
        truncated = False
        self.steps += 1

        # ---- 1. Environment Dynamics --------------------------------- #
        self._age_items()

        # ---- 2. Action Processing ------------------------------------ #
        action_reward, terminated = self._process_action(action)
        reward += action_reward

        # ---- 3. Episode Termination ---------------------------------- #
        # Critical failure: return immediately with final state and penalty
        if terminated:
            # Fix 4: Ensure exact return format for termination
            obs = self._get_observation()
            info = {
                "orders_completed": int(self.orders_completed),
                "expired_count": sum(1 for it in self.items if it["expiry_days"] <= 0),
                "step": int(self.steps),
            }
            return obs, reward, True, False, info

        # ---- 4. Regular Dynamics ------------------------------------- #
        reward += self._compute_per_step_reward()
        self._tick_deadlines(reward_ref := [0.0])
        reward += reward_ref[0]
        self._maybe_generate_orders()

        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.expired_count = sum(
            1 for it in self.items if it["expiry_days"] <= 0
        )
        reward = float(reward)

        info = {
            "orders_completed": int(self.orders_completed),
            "expired_count": int(self.expired_count),
            "step": int(self.steps),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            obs = self._get_obs()
            print(
                f"[Step {obs['steps_elapsed']:>3}] "
                f"Budget: {obs['budget_remaining']:>8.2f} | "
                f"Orders: {len(obs['pending_orders']):>2} | "
                f"Near-expiry: {obs['near_expiry_count']:>2} | "
                f"Expired: {obs['expired_count']:>2} | "
                f"Completed: {self.orders_completed:>3}"
            )

    def close(self):
        pass

    # ================================================================== #
    # Internal helpers
    # ================================================================== #

    def _generate_order(self):
        self.pending_orders.append({
            "item_id": random.randint(0, self.NUM_ITEMS - 1),
            "quantity": random.randint(1, 3),
            "deadline": random.randint(10, 20),
        })

    def _age_items(self):
        """Items lose 0.05 expiry_day every step.
        Expired items have their stock zeroed (they become waste)."""
        self._newly_expired = 0
        for it in self.items:
            if it["expiry_days"] > 0:
                it["expiry_days"] = max(0.0, it["expiry_days"] - 0.05)
                if it["expiry_days"] <= 0 and it["stock"] > 0:
                    # Item just expired — waste it
                    self._newly_expired += 1
                    it["stock"] = 0  # expired stock is gone

    def _compute_per_step_reward(self) -> float:
        """Survival and budget bonuses / expiry penalties (per step)."""
        reward = 0.0
        newly_expired = getattr(self, '_newly_expired', 0)

        # Check total expired items (Bug 3)
        current_expired_count = sum(1 for it in self.items if it["expiry_days"] <= 0)
        if current_expired_count == 0:
            reward += 2.0              # +2 survival bonus
        
        # Penalty for newly expired items this step (Bug 2)
        if newly_expired > 0:
            reward -= newly_expired * 5.0

        if self.budget > 500:
            reward += 1.0              # +1 budget comfort bonus
        return reward

    def _tick_deadlines(self, reward_ref: List[float]):
        """Decrement order deadlines and penalise missed ones."""
        missed_penalty = 0.0
        for order in self.pending_orders:
            order["deadline"] -= 1
            if order["deadline"] < 0:
                missed_penalty -= 10.0   # -10 missed deadline
        self.pending_orders = [
            o for o in self.pending_orders if o["deadline"] >= 0
        ]
        reward_ref[0] = missed_penalty

    def _maybe_generate_orders(self):
        """Generate 1-2 new orders every 10 steps."""
        if self.steps % 10 == 0:
            for _ in range(random.randint(1, 2)):
                self._generate_order()

    # ------------------------------------------------------------------ #
    # Action handlers
    # ------------------------------------------------------------------ #

    def _process_action(self, action: int) -> Tuple[float, bool]:
        """Dispatch to the correct action handler. Returns (reward, terminated)."""
        handlers = {
            0: self._act_pick_item,
            1: self._act_restock,
            2: self._act_apply_discount,
            3: self._act_dispatch_order,
            4: self._act_batch_pick,
            5: self._act_hold,
        }
        return handlers[action]()

    def _fefo_item(self) -> Optional[Dict]:
        """Return the in-stock item with the lowest (but >0) expiry_days."""
        candidates = [
            it for it in self.items
            if it["stock"] > 0 and it["expiry_days"] > 0
        ]
        return min(candidates, key=lambda x: x["expiry_days"]) if candidates else None

    def _act_pick_item(self) -> Tuple[float, bool]:
        """
        Pick item for the first pending order.
        +5  if the required item IS the FEFO item (oldest safe stock)
        0   if it isn't (no penalty — just not optimal)
        -50 if no stock available at all
        """
        if not self.pending_orders:
            return -5.0, False      # no orders to pick for

        fefo = self._fefo_item()
        if fefo is None:
            return -50.0, False     # out of stock entirely

        order = self.pending_orders[0]
        required_item = next(
            (it for it in self.items if it["id"] == order["item_id"]), None
        )
        self.picked_item_id = order["item_id"]

        if required_item and required_item["id"] == fefo["id"]:
            return 5.0, False       # FEFO-optimal pick
        return 0.0, False           # valid pick, not FEFO-optimal

    def _act_restock(self) -> Tuple[float, bool]:
        reward = 0.0
        if self.budget < 200:
            reward -= 30.0          # -30 unsafe restock (low budget)

        if self.budget >= self.RESTOCK_COST:
            self.budget -= self.RESTOCK_COST
            target = random.choice(self.items)
            target["stock"] += self.RESTOCK_STOCK
            target["expiry_days"] = float(random.randint(20, 40))
        else:
            reward -= 50.0          # -50 out of budget

        return reward, False

    def _act_apply_discount(self) -> Tuple[float, bool]:
        reward = 0.0
        discounted = False
        for it in self.items:
            if 0 < it["expiry_days"] <= 3:
                it["price"] = round(it["price"] * 0.9, 2)
                reward += 5.0       # +5 per near-expiry discount
                discounted = True
        if not discounted:
            reward -= 5.0           # -5 wasteful discount action
        return reward, False

    def _act_dispatch_order(self) -> Tuple[float, bool]:
        if not self.pending_orders:
            return -10.0, False

        order = self.pending_orders[0]
        it = next(
            (i for i in self.items if i["id"] == order["item_id"]), None
        )
        if it is None:
            return -50.0, False

        if it["stock"] < order["quantity"]:
            return -50.0, False     # -50 out of stock

        # --- Safety check ---
        if it["expiry_days"] <= 0:
            # Critical Failure: dispatching truly expired item
            reward = -500.0
            terminated = True
            return reward, terminated

        if 0 < it["expiry_days"] <= 1:
            # Unsafe dispatch (last day remaining)
            it["stock"] -= order["quantity"]
            self.pending_orders.pop(0)
            return -100.0, False    # -100 unsafe but not terminal

        # --- Valid dispatch ---
        it["stock"] -= order["quantity"]
        self.pending_orders.pop(0)
        self.orders_completed += 1
        reward = 10.0               # +10 valid dispatch

        # FEFO bonus: only if the dispatched item IS the globally oldest safe item
        fefo = self._fefo_item()
        # fefo check BEFORE we removed stock — so use item id
        if fefo is None or it["id"] == fefo["id"]:
            reward += 20.0          # +20 FEFO dispatch

        return reward, False

    def _act_batch_pick(self) -> Tuple[float, bool]:
        dispatched = 0
        i = 0
        while dispatched < 3 and i < len(self.pending_orders):
            order = self.pending_orders[i]
            it = next(
                (x for x in self.items if x["id"] == order["item_id"]), None
            )
            if it and it["stock"] >= order["quantity"] and it["expiry_days"] > 1:
                it["stock"] -= order["quantity"]
                self.orders_completed += 1
                self.pending_orders.pop(i)
                dispatched += 1
            else:
                i += 1              # skip this order, try next

        if dispatched == 0:
            return -50.0, False
        return dispatched * 10.0, False   # +10 per dispatched order

    def _act_hold(self) -> Tuple[float, bool]:
        return 0.0, False

    # ------------------------------------------------------------------ #
    # Observation builder
    # ------------------------------------------------------------------ #

    def observe(self) -> Dict[str, Any]:
        """Alias for _get_obs() for OpenEnv compliance."""
        return self._get_obs()

    def _get_observation(self) -> Dict[str, Any]:
        """User-requested alias for obs generation."""
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        """
        Returns the canonical rich dict observation.
        Keys are exactly what the grader and API expect.
        """
        near_expiry = sum(
            1 for it in self.items if 0 < it["expiry_days"] <= 3
        )
        expired = sum(1 for it in self.items if it["expiry_days"] <= 0)

        return {
            "item_states": [
                {
                    "id": int(it["id"]),
                    "stock": int(it["stock"]),
                    "expiry_days": float(it["expiry_days"]),
                    "price": float(it["price"]),
                }
                for it in self.items
            ],
            "pending_orders": [
                {
                    "item_id": int(o["item_id"]),
                    "quantity": int(o["quantity"]),
                    "deadline": int(o["deadline"]),
                }
                for o in self.pending_orders
            ],
            "budget_remaining": float(self.budget),
            "near_expiry_count": int(near_expiry),
            "steps_elapsed": int(self.steps),
            "expired_count": int(expired),
        }
