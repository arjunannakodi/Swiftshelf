from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ItemState(BaseModel):
    id: int
    stock: int
    expiry_days: float
    price: float


class PendingOrder(BaseModel):
    item_id: int
    quantity: int
    deadline: int


class InventoryObservation(BaseModel):
    item_states: List[Dict[str, Any]]
    pending_orders: List[Dict[str, Any]]
    budget_remaining: float
    near_expiry_count: int
    expired_count: int
    steps_elapsed: int


class InventoryAction(BaseModel):
    action: int  # 0-5


class InventoryState(BaseModel):
    """Full episode state for state() endpoint."""
    observation: InventoryObservation
    total_reward: float
    step: int
    done: bool
    episode_id: Optional[str] = None


class StepResult(BaseModel):
    observation: InventoryObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
