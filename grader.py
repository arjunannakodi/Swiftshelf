from env.environment import InventoryEnv
import random

def heuristic_agent(state):
    # Action logic:
    # 1. Dispatch if possible (Highest priority)
    if state["pending_orders"]:
        order = state["pending_orders"][0]
        # Match item by casting id to string (since state["item_states"] uses strings)
        item = next((i for i in state["item_states"] if str(i["id"]) == str(order["item_id"])), None)
        if item and item["stock"] >= order["quantity"] and item["expiry_days"] > 0:
            return 3 # dispatch_order
            
    # 2. Check for restock if budget is healthy and stock is low for any item
    if state["budget_remaining"] >= 200:
        if any(item["stock"] < 5 for item in state["item_states"]):
            return 1 # restock

    # 3. Apply discount to near-expiry items
    if state["near_expiry_count"] > 0:
        return 2 # apply_discount

    # 4. Default to hold
    return 5 

def run_episodes(num_episodes=5, max_steps=50):
    env = InventoryEnv()
    episode_results = []
    
    print("\n" + "=" * 65)
    print(f"{'Episode':<10} | {'Reward':<10} | {'Expired':<10} | {'Orders':<10} | {'Steps':<10}")
    print("-" * 65)
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        while not done and step_count < max_steps:
            action = heuristic_agent(state)
            state, reward, done, info = env.step(action)
            # LIVE DATA FEED
            print(f"DEBUG | Step: {step_count+1:<3} | Info: {info}")
            total_reward += reward
            step_count += 1
            
        episode_results.append({
            "reward": total_reward,
            "expired": info["expired_count"],
            "orders": info["orders_completed"],
            "steps": step_count
        })
        print(f"{ep+1:<10} | {total_reward:<10.2f} | {info['expired_count']:<10} | {info['orders_completed']:<10} | {step_count:<10}")

    # Summary Metrics
    avg_reward = sum(r["reward"] for r in episode_results) / num_episodes
    avg_expired = sum(r["expired"] for r in episode_results) / num_episodes
    avg_orders = sum(r["orders"] for r in episode_results) / num_episodes
    avg_steps = sum(r["steps"] for r in episode_results) / num_episodes

    print("-" * 65)
    print(f"{'AVERAGE':<10} | {avg_reward:<10.2f} | {avg_expired:<10.2f} | {avg_orders:<10.2f} | {avg_steps:<10.2f}")
    print("=" * 65 + "\n")
    
    # PASS Condition: reward threshold and waste limit
    reward_threshold = -500.0
    waste_limit = 15.0
    
    if avg_reward >= reward_threshold and avg_expired < waste_limit:
        print("RESULT: PASS ✅")
        print(f"Confidence Level: High (Avg Reward {avg_reward:.2f} > {reward_threshold}, Avg Waste {avg_expired:.2f} < {waste_limit})")
    else:
        print("RESULT: FAIL ❌")
        print(f"Reason: Performance fell below thresholds.")
    
    return avg_reward

if __name__ == "__main__":
    run_episodes()
