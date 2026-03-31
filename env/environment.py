import numpy as np
import random
from typing import List, Dict, Any

class InventoryEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # Initialize all variables safely
        self.items = [
            {"id": i, "stock": 50, "expiry_days": random.randint(10, 30), "price": random.uniform(10.0, 50.0), "discounted": False}
            for i in range(10)
        ]
        self.pending_orders = [
            {"id": i, "item_id": random.randint(0, 9), "quantity": random.randint(1, 5), "deadline": random.randint(5, 15)}
            for i in range(5)
        ]
        self.budget = 1000.0
        self.steps = 0
        self.expired_count = 0
        self.orders_completed = 0
        self.near_expiry_count = 0
        self.system_online = 1
        return self.state()

    def state(self) -> Dict[str, Any]:
        # Calculate near_expiry_count before returning
        self.near_expiry_count = sum(1 for item in self.items if 0 < item["expiry_days"] <= 3)
        return {
            "item_states": [
                {
                    "id": str(item["id"]),
                    "stock": int(item["stock"]),
                    "expiry_days": int(item["expiry_days"]),
                    "price": float(item["price"]),
                    "discounted": bool(item.get("discounted", False))
                }
                for item in self.items
            ],
            "pending_orders": self.pending_orders,
            "budget_remaining": float(self.budget),
            "steps_elapsed": int(self.steps),
            "expired_count": int(self.expired_count),
            "near_expiry_count": int(self.near_expiry_count),
            "system_online": int(self.system_online)
        }

    def step(self, action: int):
        try:
            # --- SAFE DEFAULTS ---
            reward = 0.0
            done = False
            self.steps += 1
            
            # --- PROCESS DYNAMICS (Legacy Core) ---
            self.system_online = 1 if random.random() >= 0.02 else 0

            for item in self.items:
                if item["expiry_days"] > 0:
                    item["expiry_days"] -= 1
                    if item["expiry_days"] == 0:
                        self.expired_count += 1
                        reward -= 10.0

            for order in self.pending_orders:
                order["deadline"] -= 1
                if order["deadline"] < 0:
                    reward -= 20.0
            self.pending_orders = [o for o in self.pending_orders if o["deadline"] >= 0]

            # --- PROCESS ACTION ---
            action = int(action)
            if action == 0: # pick_item
                avail = [it for it in self.items if it["stock"] > 0 and it["expiry_days"] > 0]
                if avail:
                    fefo = min(avail, key=lambda x: x["expiry_days"])
                    if self.pending_orders:
                        if fefo["id"] == self.pending_orders[0]["item_id"]:
                            reward += 5.0
                    else: reward -= 5.0
                else: reward -= 50.0

            elif action == 1: # restock
                if self.budget >= 200.0:
                    self.budget -= 200.0
                    target = min(self.items, key=lambda x: x["stock"])
                    target["stock"] += 50
                    target["expiry_days"] = random.randint(8, 15)
                    reward += 10.0
                else: reward -= 50.0

            elif action == 2: # apply_discount
                nears = [it for it in self.items if 0 < it["expiry_days"] <= 3 and not it["discounted"]]
                if nears:
                    nears[0]["price"] *= 0.8
                    nears[0]["discounted"] = True
                    reward += 10.0
                else: reward -= 50.0

            elif action == 3: # dispatch_order
                if self.pending_orders:
                    ord = self.pending_orders.pop(0)
                    it = next((i for i in self.items if i["id"] == ord["item_id"]), None)
                    if it and it["stock"] >= ord["quantity"]:
                        if it["expiry_days"] <= 0:
                            reward -= 500.0
                            done = True
                        else:
                            it["stock"] -= ord["quantity"]
                            reward += 10.0
                            self.orders_completed += 1
                            if ord["deadline"] > 5: reward += 5.0
                    else: reward -= 100.0
                else: reward -= 50.0

            elif action == 4: # batch_pick
                dispatched = 0
                to_remove = []
                for i, ord in enumerate(self.pending_orders):
                    it = next((it for it in self.items if it["id"] == ord["item_id"]), None)
                    if it and it["stock"] >= ord["quantity"] and it["expiry_days"] > 0:
                        it["stock"] -= ord["quantity"]
                        to_remove.append(i)
                        dispatched += 1
                        self.orders_completed += 1
                    if dispatched >= 3: break
                if dispatched > 0:
                    for idx in reversed(to_remove): self.pending_orders.pop(idx)
                    reward += (dispatched * 8.0)
                else: reward -= 50.0

            # Step bonuses
            if all(it["expiry_days"] > 0 or it["stock"] == 0 for it in self.items): reward += 1.0
            if self.budget > 200: reward += 1.0

            # Periodic orders
            if self.steps % 3 == 0:
                self.pending_orders.append({
                    "id": len(self.pending_orders) + self.steps,
                    "item_id": random.randint(0, 9),
                    "quantity": random.randint(1, 5),
                    "deadline": random.randint(5, 15)
                })

            # --- UPDATE STATE ---
            state = self.state()

            # --- BUILD INFO (MANDATORY KEYS) ---
            info = {
                "orders_completed": int(getattr(self, "orders_completed", 0)),
                "expired_count": int(getattr(self, "expired_count", 0))
            }

            # --- TYPE SAFETY ---
            reward = float(reward)
            done = bool(done)

            return state, reward, done, info

        except Exception as e:
            # FAIL-SAFE RETURN (NEVER CRASH)
            return self.state(), -50.0, True, {
                "orders_completed": 0,
                "expired_count": 0,
                "error": str(e)
            }
