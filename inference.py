"""
inference.py — SwiftShelf++ Baseline Inference Script
OpenEnv-compliant stdout format: [START], [STEP], [END]
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ── Environment variables (mandatory) ───────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://arjunannakodi-swiftshelf.hf.space")

BENCHMARK    = "swiftshelf-plus-plus"
MAX_STEPS    = 50
TEMPERATURE  = 0.3
MAX_TOKENS   = 80

SYSTEM_PROMPT = """You are an AI agent managing a dark-store inventory.
Choose the single best action (0-5) based on item stock and expiry.
0=pick_item, 1=restock, 2=apply_discount, 3=dispatch_order, 4=batch_pick, 5=hold.
Respond with ONLY a single digit digit: 0, 1, 2, 3, 4, or 5.
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={r_str}",
        flush=True,
    )

def get_action(client: OpenAI, obs: dict, step: int) -> int:
    user_msg = f"Step {step}/{MAX_STEPS}. Budget: {obs.get('budget_remaining', 0):.0f}. Choose action (0-5):"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        for ch in text:
            if ch.isdigit() and int(ch) < 6:
                return int(ch)
    except:
        pass
    return 5 # Fallback hold

def run_task(client: OpenAI, task_id: int) -> None:
    # --- FIX 3: Robust tasks meta fetching ---
    try:
        tasks_resp = requests.get(f"{ENV_BASE_URL}/tasks").json()
        if isinstance(tasks_resp, dict):
            tasks_list = list(tasks_resp.values())
        else:
            tasks_list = tasks_resp
        task_meta = next((t for t in tasks_list if t.get("id") == task_id or t.get("id") == f"task{task_id}"), None)
    except:
        task_meta = None

    if not task_meta:
        task_meta = {
            "id": task_id,
            "name": f"Task {task_id}",
            "difficulty": ["easy","medium","hard"][task_id-1]
        }

    log_start(task=task_meta["name"], env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    success = False
    score = 0.0
    steps_taken = 0
    obs = {}

    try:
        r = requests.post(f"{ENV_BASE_URL}/reset")
        obs = r.json().get("observation", {})
        
        for step in range(1, MAX_STEPS + 1):
            action = get_action(client, obs, step)
            try:
                step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}).json()
                obs = step_resp.get("observation", obs)
                reward = float(step_resp.get("reward", 0.0))
                done = bool(step_resp.get("terminated") or step_resp.get("truncated"))
                info = step_resp.get("info", {})
            except:
                reward, done, info = 0.0, False, {}

            rewards.append(reward)
            steps_taken = step
            log_step(step, str(action), reward, done, None)
            
            if done: break

        # --- FIX 4: Correct Score Normalization ---
        orders_completed = info.get("orders_completed", 0)
        expired_count = obs.get("expired_count", 0)
        steps_elapsed = obs.get("steps_elapsed", steps_taken)

        if task_id == 1:
            score = 1.0 if (orders_completed >= 1 and expired_count == 0) else 0.0
            success = score >= 0.5
        elif task_id == 2:
            score = 1.0 if (expired_count <= 2 and orders_completed >= 1) else 0.0
            success = score >= 0.5
        elif task_id == 3:
            raw = (orders_completed * 20) - (expired_count * 15) - steps_elapsed
            score = max(0.0, min(1.0, float(raw) / 100.0))
            success = score >= 0.1

    except Exception as e:
        print(f"[ERROR] {e}")

    log_end(success, steps_taken, score, rewards)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    for tid in [1, 2, 3]:
        run_task(client, tid)

if __name__ == "__main__":
    main()
