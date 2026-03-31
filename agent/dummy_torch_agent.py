import torch
import random
from typing import Dict, Any

class DummyTorchAgent:
    """A minimal PyTorch-aligned agent for demonstration."""
    def __init__(self, action_space: int = 6):
        self.action_space = action_space
        # A simple linear layer (no training required for dummy)
        self.policy_head = torch.nn.Linear(10, self.action_space)

    def select_action(self, observation: Dict[str, Any]) -> int:
        """Select action using dummy policy logic."""
        # Convert some observation data to tensor for alignment
        # (Using steps_elapsed and current stock as a dummy feature vector)
        stock_sum = sum(item["stock"] for item in observation.get("item_states", []))
        features = torch.tensor([
            float(observation.get("steps_elapsed", 0)),
            float(observation.get("budget_remaining", 0)),
            float(stock_sum),
            float(observation.get("expired_count", 0)),
            float(observation.get("near_expiry_count", 0)),
            float(observation.get("system_online", 1)),
            0.0, 0.0, 0.0, 0.0 # Padding
        ], dtype=torch.float32)

        # Non-trained linear pass
        with torch.no_grad():
            logits = self.policy_head(features)
            action = torch.argmax(logits).item()

        # For this demonstration, we just return a random valid action
        # but the above shows the PyTorch infrastructure is ready.
        return random.randint(0, self.action_space - 1)

if __name__ == "__main__":
    agent = DummyTorchAgent()
    print("Dummy Torch Agent Initialized.")
    # Mock observation
    mock_obs = {"steps_elapsed": 10, "budget_remaining": 500, "expired_count": 2}
    print(f"Action selected: {agent.select_action(mock_obs)}")
