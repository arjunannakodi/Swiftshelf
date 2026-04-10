"""
SwiftShelf++ OpenEnv Client
Connects agents to the SwiftShelf++ environment server.
"""

import requests
from typing import Optional
from models import InventoryObservation, InventoryAction, StepResult, InventoryState


class SwiftShelfClient:
    """
    HTTP client for the SwiftShelf++ environment.
    Compatible with OpenEnv EnvClient interface.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._episode_reward = 0.0
        self._step = 0

    def reset(self, seed: Optional[int] = None) -> InventoryObservation:
        """Reset environment and return initial observation."""
        payload = {"seed": seed} if seed is not None else {}
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        self._episode_reward = 0.0
        self._step = 0
        return InventoryObservation(**data["observation"])

    def step(self, action: InventoryAction) -> StepResult:
        """Execute action and return result."""
        r = requests.post(
            f"{self.base_url}/step",
            json={"action": action.action},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        self._episode_reward += data["reward"]
        self._step += 1
        return StepResult(**data)

    def state(self) -> InventoryState:
        """Return current full episode state."""
        r = requests.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        data = r.json()
        return InventoryState(
            observation=InventoryObservation(**data["observation"]),
            total_reward=data.get("total_reward", self._episode_reward),
            step=data.get("step", self._step),
            done=data.get("done", False),
        )

    def close(self):
        """Cleanup (no-op for HTTP client)."""
        pass

    @classmethod
    def from_url(cls, url: str) -> "SwiftShelfClient":
        return cls(base_url=url)
