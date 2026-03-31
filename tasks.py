from typing import Dict, Any

class Task1:
    """Basic Fulfillment: Complete orders without waste."""
    def __init__(self):
        pass

    def evaluate(self, state, info, done):
        # SUCCESS if orders_completed >= 1 and expired_count == 0
        return info["orders_completed"] >= 1 and info["expired_count"] == 0

class Task2:
    """Waste Reduction & Performance: Expired <= 2 and at least 1 order."""
    def __init__(self):
        pass

    def evaluate(self, state, info, done):
        # SUCCESS if expired_count <= 2 and orders_completed >= 1
        return (
            info["expired_count"] <= 2 and
            info["orders_completed"] >= 1
        )

class Task3:
    """Efficiency: Maximize performance across metrics."""
    def __init__(self):
        pass

    def evaluate(self, state, info, done):
        orders = info["orders_completed"]
        expired = info["expired_count"]
        steps = state["steps_elapsed"]
        
        efficiency_score = (orders * 20) - (expired * 15) - steps
        return efficiency_score > 0

TASKS = {
    1: {"name": "Basic Fulfillment: Orders >= 1, Expired == 0", "class": Task1},
    2: {"name": "Waste Reduction: Expired <= 2, Orders >= 1", "class": Task2},
    3: {"name": "Efficiency: Score (Orders, Expired, Steps) > 0", "class": Task3},
}
