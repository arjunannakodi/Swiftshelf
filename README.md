---
title: SwiftShelf++
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Dark-store inventory RL environment for OpenEnv Hackathon
---

# SwiftShelf++: Inventory Decision Environment

![Meta OpenEnv](https://img.shields.io/badge/Meta-OpenEnv-brightgreen)
![PyTorch Hackathon](https://img.shields.io/badge/PyTorch-Hackathon-EE4C2C)
![Python 3.11](https://img.shields.io/badge/Python-3.11-yellow)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.26%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**SwiftShelf++** is a complete, **OpenEnv-compliant** reinforcement learning environment simulating a real-world dark-store inventory management system. Submitted to the **Meta PyTorch OpenEnv Hackathon**.

---

##  Project Architecture

```
SwiftShelf++/
├── inference.py         # OpenEnv baseline inference script (root)
├── openenv.yaml         # OpenEnv metadata and spec (root)
├── env/
│   ├── __init__.py          # Package init
│   └── environment.py       # Core RL Environment (gymnasium.Env)
├── server/
│   ├── app.py               # FastAPI REST server (port 7860)
│   └── index.html           # Live dashboard (D3.js)
├── agent/
│   └── llm_agent.py         # PyTorch LLM Agent (OPT-125m)
├── tests/
│   └── test_env.py          # 9-test PyTest suite
├── tasks.py                 # OpenEnv Task Definitions (3 tasks)
├── grader.py                # Automated Performance Evaluator
├── Dockerfile               # Container build (python:3.11-slim)
├── requirements.txt         # Dependencies
└── README.md
```

---

## OpenEnv Compliance

This environment is fully compliant with the OpenEnv standard:

| Requirement | Status |
|---|---|
| `openenv.yaml` present | ✅ |
| `openenv validate` passes | ✅ |
| `inference.py` in root | ✅ |
| HF Space deployed | ✅ |
| Typed Pydantic models | ✅ |
| 3 tasks with graders (0.0–1.0) | ✅ |
| Structured stdout logs | ✅ |

### Run Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

Expected output: `[START]`, `[STEP]`, `[END]` traces for all 3 tasks.

---

##  Environment

### Core Properties

| Property | Value |
|---|---|
| Base class | `gymnasium.Env` |
| Action space | `Discrete(6)` |
| Episode length | 200 steps |
| Items | 20 unique SKUs |
| Initial budget | 1000.0 |

### Action Space

| ID | Action | Description |
|---|---|---|
| 0 | `pick_item` | Pick item for first pending order using FEFO logic |
| 1 | `restock` | Add 20 units of stock to a random item (costs 50 budget) |
| 2 | `apply_discount` | Apply 10% discount to near-expiry items (expiry ≤ 3) |
| 3 | `dispatch_order` | Fulfill the first pending order |
| 4 | `batch_pick` | Dispatch up to 3 orders in one step |
| 5 | `hold` | Strategic inaction |

### Observation Space

| Field | Type | Description |
|---|---|---|
| `item_states` | `list[dict]` | `id`, `stock`, `expiry_days`, `price` per item (20 items) |
| `pending_orders` | `list[dict]` | `item_id`, `quantity`, `deadline` per order |
| `budget_remaining` | `float` | Available budget (starts at 1000.0) |
| `near_expiry_count` | `int` | Items with `expiry_days` ≤ 3 |
| `expired_count` | `int` | Items with `expiry_days` ≤ 0 |
| `steps_elapsed` | `int` | Steps taken this episode |

### Reward Table

| Event | Reward |
|---|---|
| Dispatch with FEFO (oldest safe stock first) | **+20** |
| Valid dispatch (any safe fulfillment) | **+10** |
| Batch dispatch (per order) | **+10** |
| Discount near-expiry item | **+5** |
| FEFO-optimal pick | **+5** |
| Survival bonus (zero expired items per step) | **+2** |
| Budget comfort bonus (budget > 500) | **+1** |
| FEFO violation pick | 0 (missed bonus) |
| No orders to pick for | −5 |
| Missed order deadline | −10 |
| Restock when budget < 200 | −30 |
| Out of budget / out of stock | −50 |
| Unsafe dispatch (expiry_days ≤ 1) | −100 |
| Expired item dispatched (expiry_days ≤ 0) | **−500 + terminate** |

---

##  LLM Agent (Meta PyTorch Integration)

SwiftShelf++ ships with a **PyTorch-powered LLM agent** using Meta's `facebook/opt-125m` model. This is the core demonstration of the Meta PyTorch requirement.

### How It Works

1. The agent converts the current observation into a natural-language prompt.
2. The `OPT-125m` model generates a response via `torch.no_grad()` inference.
3. The agent parses the digit output to select an action (0–5).
4. Runs via the HTTP API (`/reset` → `/step` loop).

### Run the LLM Agent

> **Prerequisite**: start the API server first (see below).
> 
> Optional: set `HF_TOKEN` env var to avoid HuggingFace
> rate-limit warnings when loading OPT-125m:
> `export HF_TOKEN=your_token_here`

```bash
python agent/llm_agent.py
```

> Note: OPT-125m runs on CPU. Each episode uses 50 steps
> for demo speed (~2-3 minutes per episode).
> 
> LLM Agent output (actual):
> LLM Episode 1: 148.0
> LLM Episode 2: 328.0
> LLM Episode 3: 80.0
> LLM Agent Average: 185.3
> 
> Note: Episode rewards vary per run due to stochastic
> environment dynamics. Typical average range: 120–190.
>
> The LLM agent's role is to demonstrate PyTorch inference
> integration. The heuristic agent is the performance benchmark.

### Sample Prompt

```
Inventory manager. Near expiry: 2. Pending orders: 4. Budget: 950. Expired items: 0.
Actions: 0=pick 1=restock 2=discount 3=dispatch 4=batch 5=hold
Best action digit:
```

---

##  Agent Performance Comparison

| Agent | Avg Episode Reward | Strategy |
|---|---|---|
| Heuristic Agent | ~978 (range: 344–1352) | Rule-based FEFO + dispatch priority |
| LLM Agent (OPT-125m) | ~185 | PyTorch inference + observation-based fallback |

> Heuristic reward varies per run due to stochastic item
> generation and order timing. All runs produce RESULT: PASS.
>
> The heuristic agent is the performance benchmark.
> The LLM agent demonstrates Meta PyTorch integration.
> Both significantly outperform random baseline (expected ~−50 to −200).

---

##  Running SwiftShelf++

### Option 1 — Docker (Recommended)

```bash
docker build -t swiftshelf-env .
docker run -p 7860:7860 swiftshelf-env
```

### Option 2 — Local

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

---

##  API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check → `{"status":"ok","env":"SwiftShelf++ v1.0"}` |
| `POST` | `/reset` | Reset environment → `{"observation": {...}, "info": {}}` |
| `GET` | `/state` | Current observation as JSON |
| `POST` | `/step` | Body: `{"action": int}` → `{observation, reward, terminated, truncated, info}` |
| `POST` | `/grade` | Run 3 heuristic episodes → `{avg_reward, status: "PASS"/"FAIL"}` |
| `GET` | `/tasks` | List evaluation task metadata |

### Step Request / Response

**Request:**
```json
{"action": 3}
```

**Response:**
```json
{
  "observation": { "item_states": [...], "pending_orders": [...], "budget_remaining": 950.0, "near_expiry_count": 1, "steps_elapsed": 1, "expired_count": 0 },
  "reward": 33.0,
  "terminated": false,
  "truncated": false,
  "info": { "orders_completed": 1, "expired_count": 0, "step": 1 }
}
```

---

##  Testing and Validation

### Run Tests

```bash
python -m pytest tests/ -v
```

### Expected Output

```
tests/test_env.py::test_reset_returns_valid_obs        PASSED
tests/test_env.py::test_step_returns_5_tuple           PASSED
tests/test_env.py::test_valid_actions_0_to_5           PASSED
tests/test_env.py::test_invalid_action_raises          PASSED
tests/test_env.py::test_episode_terminates             PASSED
tests/test_env.py::test_reward_not_always_zero         PASSED
tests/test_env.py::test_reset_clears_state             PASSED
tests/test_env.py::test_gymnasium_env_compliance       PASSED
tests/test_env.py::test_fefo_bonus_is_conditional      PASSED

9 passed in 0.XX s
```

### Run Grader

```bash
python grader.py
```

### Sample Grader Output

```
======================================================================
Episode    | Reward       | Expired    | Orders     | Steps     
----------------------------------------------------------------------
1          | 643.00       | 8          | 23         | 200       
2          | 2358.00      | 7          | 29         | 200       
3          | 106.00       | 4          | 24         | 200       
4          | 1757.00      | 2          | 27         | 200       
5          | 1328.00      | 11         | 23         | 200       
----------------------------------------------------------------------
AVERAGE    | 1238.40      | 6.40       | 25.20      | 200.00    
======================================================================

RESULT: PASS
Confidence Level: High (Avg Reward 1238.40 >= -500.0, Avg Waste 6.40 < 15.0)
```

---

##  Evaluation Tasks

| Task | Name | Success Condition | Return Type |
|---|---|---|---|
| 1 | Basic Fulfillment | `orders_completed >= 1` AND `expired_count == 0` | `float (0.0 or 1.0)` |
| 2 | Waste Reduction | `expired_count <= 2` AND `orders_completed >= 1` | `float (0.0 or 1.0)` |
| 3 | Efficiency Score | `(orders×20 - expired×15 - steps) / 100` clamped `[0, 1]` | `float [0.0, 1.0]` |

---

## Evaluation Metrics

1. **Fulfillment Rate** — Number of orders dispatched per episode.
2. **Waste Management** — Items at `expiry_days <= 0` (penalised heavily).
3. **FEFO Compliance** — First-Expired-First-Out dispatch order rewarded with bonus.
4. **Efficiency Score** — Combined metric balancing orders, waste, and speed.

---

Built for the **Meta PyTorch OpenEnv Hackathon**   
Powered by `gymnasium`, `FastAPI`, and `facebook/opt-125m` via PyTorch.
