from typing import Dict, Any


# ------------------------------------------------------------------ #
# Task class definitions
# ------------------------------------------------------------------ #

class Task1:
    """Basic Fulfillment: Complete at least one order with zero expired items."""

    def evaluate(self, state: Dict, info: Dict, done: bool) -> float:
        success = (
            info.get("orders_completed", 0) >= 1
            and state.get("expired_count", 0) == 0
        )
        return 1.0 if success else 0.0


class Task2:
    """Waste Reduction: Keep expired items ≤ 2 while completing at least 1 order."""

    def evaluate(self, state: Dict, info: Dict, done: bool) -> float:
        success = (
            state.get("expired_count", 0) <= 2
            and info.get("orders_completed", 0) >= 1
        )
        return 1.0 if success else 0.0


class Task3:
    """
    Efficiency Score: Normalised composite metric.
    efficiency = (orders_completed × 20) - (expired_count × 15) - steps_elapsed
    Normalised to [0, 1] by dividing by 100.
    """

    def evaluate(self, state: Dict, info: Dict, done: bool) -> float:
        orders  = info.get("orders_completed", 0)
        expired = state.get("expired_count", 0)
        steps   = state.get("steps_elapsed", 0)
        raw = (orders * 20) - (expired * 15) - steps
        return max(0.0, min(1.0, float(raw) / 100.0))


# ------------------------------------------------------------------ #
# Task registry
# ------------------------------------------------------------------ #

TASKS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Basic Fulfillment",
        "description": "Complete >= 1 order with expired_count == 0.",
        "class": Task1,
    },
    2: {
        "name": "Waste Reduction",
        "description": "Keep expired_count <= 2 while completing >= 1 order.",
        "class": Task2,
    },
    3: {
        "name": "Efficiency Score",
        "description": (
            "Normalised score: (orders×20 - expired×15 - steps) / 100, "
            "clamped to [0, 1]."
        ),
        "class": Task3,
    },
}
