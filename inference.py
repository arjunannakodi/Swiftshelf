import os
import json
import requests
import time
from typing import List, Optional
from openai import OpenAI

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1") # Default placeholder
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo") # Default placeholder
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = "http://localhost:8000"

# ------------------------------------------------------------------ #
# Logging Helpers
# ------------------------------------------------------------------ #
def log_start(task: str, env_name: str, model: str):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ------------------------------------------------------------------ #
# LLM Logic
# ------------------------------------------------------------------ #
def get_action_from_llm(client: OpenAI, obs: dict) -> int:
    """Uses LLM to select an action (0-5) based on observation."""
    prompt = f"""
Inventory Manager.
Near expiry items: {obs['near_expiry_count']}
Expired items: {obs['expired_count']}
Pending orders: {len(obs['pending_orders'])}
Budget: {obs['budget_remaining']:.2f}

Actions:
0: pick_item (FEFO pick)
1: restock
2: apply_discount
3: dispatch_order
4: batch_pick
5: hold

Pick the best action digit (0-5). Reply with ONLY the digit.
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        if content.isdigit() and 0 <= int(content) <= 5:
            return int(content)
    except Exception:
        pass
    
    # Fallback heuristic if LLM fails or gives bad output
    if len(obs['pending_orders']) >= 3: return 4
    if len(obs['pending_orders']) > 0: return 3
    if obs['near_expiry_count'] > 0: return 2
    return 5

# ------------------------------------------------------------------ #
# Main Loop
# ------------------------------------------------------------------ #
def run_task(task_id: int, client: OpenAI):
    # Fetch task metadata
    tasks_resp = requests.get(f"{ENV_URL}/tasks").json()
    task_meta = next((t for t in tasks_resp if t["id"] == task_id), None)
    if not task_meta:
        return

    log_start(task=task_meta["name"], env="SwiftShelf++", model=MODEL_NAME)

    # Reset
    resp = requests.post(f"{ENV_URL}/reset").json()
    obs = resp["observation"]
    
    rewards = []
    done = False
    step = 0
    max_steps = 50 # Limit for baseline speed

    while not done and step < max_steps:
        step += 1
        action = get_action_from_llm(client, obs)
        
        step_resp = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
        
        obs = step_resp["observation"]
        reward = step_resp["reward"]
        terminated = step_resp["terminated"]
        truncated = step_resp["truncated"]
        done = terminated or truncated
        
        rewards.append(reward)
        log_step(step=step, action=str(action), reward=reward, done=done)

    # Evaluate score and success based on the task
    # (Normalization for SwiftShelf is total_reward / (max_possible_reward))
    # Approximation: max_reward is around 20-30 per step if optimal
    raw_score = sum(rewards)
    normalized_score = max(0.0, min(1.0, raw_score / 1000.0)) # Clamped score in [0, 1]
    
    success = False
    if task_id == 1:
        success = obs["expired_count"] == 0 and sum(1 for r in rewards if r >= 10) >= 1
    elif task_id == 2:
        success = obs["expired_count"] <= 2 and sum(1 for r in rewards if r >= 10) >= 1
    elif task_id == 3:
        success = normalized_score > 0.1

    log_end(success=success, steps=step, score=normalized_score, rewards=rewards)

if __name__ == "__main__":
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # Run the 3 tasks
    for tid in [1, 2, 3]:
        try:
            run_task(tid, client)
        except Exception as e:
            print(f"Error running task {tid}: {e}")
