from env.environment import InventoryEnv
import random


def heuristic_agent(state: dict) -> int:
    """
    Priority-based heuristic:
    1. Batch dispatch if 2+ viable orders exist
    2. Dispatch single order if viable
    3. Apply discount to near-expiry items
    4. Restock if any item is critically low and budget is healthy
    5. Hold
    """
    items = {it["id"]: it for it in state["item_states"]}

    # Count how many pending orders can be fulfilled safely
    fulfillable = [
        o for o in state["pending_orders"]
        if (item := items.get(o["item_id"])) is not None
        and item["stock"] >= o["quantity"]
        and item["expiry_days"] > 1
    ]

    if len(fulfillable) >= 2:
        return 4  # batch_pick

    if fulfillable:
        return 3  # dispatch_order (single)

    # Apply discount if near-expiry items exist
    if state["near_expiry_count"] > 0:
        return 2  # apply_discount

    # Restock if critically low and budget is safe
    if state["budget_remaining"] >= 300:
        if any(it["stock"] < 5 for it in state["item_states"]):
            return 1  # restock

    # Pick item for pending order (exercises action 0)
    if state["pending_orders"]:
        return 0  # pick_item

    return 5  # hold


def run_episodes(num_episodes: int = 5, max_steps: int = 200) -> float:
    env = InventoryEnv()
    episode_results = []

    sep = "=" * 70
    print(f"\n{sep}")
    print(
        f"{'Episode':<10} | {'Reward':<12} | "
        f"{'Expired':<10} | {'Orders':<10} | {'Steps':<10}"
    )
    print("-" * 70)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        info: dict = {}

        while not done and step_count < max_steps:
            action = heuristic_agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        episode_results.append({
            "reward": total_reward,
            "expired": obs["expired_count"],
            "orders": info.get("orders_completed", 0),
            "steps": step_count,
        })
        print(
            f"{ep + 1:<10} | {total_reward:<12.2f} | "
            f"{obs['expired_count']:<10} | "
            f"{info.get('orders_completed', 0):<10} | "
            f"{step_count:<10}"
        )

    # Summary
    avg_reward  = sum(r["reward"]  for r in episode_results) / num_episodes
    avg_expired = sum(r["expired"] for r in episode_results) / num_episodes
    avg_orders  = sum(r["orders"]  for r in episode_results) / num_episodes
    avg_steps   = sum(r["steps"]   for r in episode_results) / num_episodes

    print("-" * 70)
    print(
        f"{'AVERAGE':<10} | {avg_reward:<12.2f} | "
        f"{avg_expired:<10.2f} | {avg_orders:<10.2f} | {avg_steps:<10.2f}"
    )
    print(f"{sep}\n")

    # PASS / FAIL thresholds
    reward_threshold = -500.0
    waste_limit = 15.0

    if avg_reward >= reward_threshold and avg_expired < waste_limit:
        print("RESULT: PASS")
        print(
            f"Confidence Level: High "
            f"(Avg Reward {avg_reward:.2f} >= {reward_threshold}, "
            f"Avg Waste {avg_expired:.2f} < {waste_limit})"
        )
    else:
        print("RESULT: FAIL")
        print(
            f"Reason: avg_reward={avg_reward:.2f} "
            f"(threshold {reward_threshold}), "
            f"avg_expired={avg_expired:.2f} (limit {waste_limit})"
        )

    return avg_reward


if __name__ == "__main__":
    run_episodes()
