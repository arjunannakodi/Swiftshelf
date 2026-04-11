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
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**SwiftShelf++** is a complete, **OpenEnv-compliant** reinforcement learning environment simulating a real-world dark-store inventory management system. Submitted to the **Meta PyTorch OpenEnv Hackathon**.

> 🌐 **Live on Hugging Face Spaces:** [arjunannakodi/Swiftshelf](https://huggingface.co/spaces/arjunannakodi/Swiftshelf)

---

## Project Architecture

```
SwiftShelf++/
├── __init__.py          # Package exports (InventoryEnv, SwiftShelfClient, models)
├── models.py            # Pydantic models: InventoryAction, InventoryObservation, InventoryState, StepResult
├── client.py            # SwiftShelfClient — OpenEnv-compatible HTTP client
├── inference.py         # OpenEnv baseline inference script (stdout: [START]/[STEP]/[END])
├── tasks.py             # OpenEnv Task Definitions (Task1, Task2, Task3)
├── grader.py            # Automated heuristic performance evaluator
├── openenv.yaml         # OpenEnv spec (spec_version: 1)
├── pyproject.toml       # Package metadata + [project.scripts] server entry
├── Dockerfile           # Container build (python:3.11-slim)
├── README.md
├── env/
│   ├── __init__.py
│   └── environment.py   # Core RL Environment (gymnasium.Env, Discrete(6))
├── server/
│   ├── app.py           # FastAPI REST server (port 7860) + keepalive lifespan
│   └── index.html       # Live dashboard (D3.js)
├── agent/
│   └── llm_agent.py     # PyTorch LLM Agent (facebook/opt-125m)
├── tests/
│   └── test_env.py      # PyTest suite (9 tests)
└── tmp/
    ├── openenv_validation.py    # OpenEnv runtime & local validation library
    └── openenv_validate_cmd.py  # `openenv validate` CLI command implementation
```

---

## OpenEnv Compliance

| Requirement | Status |
|---|---|
| `openenv.yaml` (spec_version: 1) | ✅ |
| `models.py` (Pydantic typed models) | ✅ |
| `client.py` (SwiftShelfClient) | ✅ |
| `__init__.py` (package exports) | ✅ |
| `inference.py` in root | ✅ |
| Structured stdout `[START]`/`[STEP]`/`[END]` logs | ✅ |
| HF Space deployed (Docker SDK) | ✅ |
| 3 tasks scoring in range `(0.01, 0.99)` | ✅ |
| `/health`, `/metadata`, `/schema`, `/mcp` endpoints | ✅ |
| `[project.scripts]` server entry point | ✅ |

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

## Environment

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
| 2 | `apply_discount` | Apply 10% discount to near-expiry items (expiry ≤ 3 days) |
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

## LLM Agent (Meta PyTorch Integration)

SwiftShelf++ ships with a **PyTorch-powered LLM agent** using Meta's `facebook/opt-125m` model.

### How It Works

1. The agent converts the current observation into a natural-language prompt.
2. The `OPT-125m` model generates a response via `torch.no_grad()` inference.
3. The agent parses the digit output to select an action (0–5).
4. Falls back to a heuristic rule if the model output is invalid.
5. Runs via the HTTP API (`/reset` → `/step` loop).

### Run the LLM Agent

> **Prerequisite**: start the API server first (see below).

```bash
python agent/llm_agent.py
```

> Note: OPT-125m runs on CPU. Each episode uses up to 50 steps.
>
> LLM Agent sample output:
> ```
> LLM Episode 1: 148.0
> LLM Episode 2: 328.0
> LLM Episode 3: 80.0
> LLM Agent Average: 185.3
> ```
> Episode rewards vary per run due to stochastic environment dynamics.
> Typical average range: 120–190.

### Sample Prompt

```
Inventory manager. Near expiry: 2. Pending orders: 4. Budget: 950. Expired items: 0.
Actions: 0=pick 1=restock 2=discount 3=dispatch 4=batch 5=hold
Best action digit:
```

---

## Agent Performance Comparison

| Agent | Avg Episode Reward | Strategy |
|---|---|---|
| Heuristic Agent | ~978 (range: 344–1352) | Rule-based FEFO + dispatch priority |
| LLM Agent (OPT-125m) | ~185 (range: 80–330) | PyTorch inference + heuristic fallback |

> The heuristic agent is the performance benchmark.
> The LLM agent demonstrates Meta PyTorch integration.
> Both significantly outperform a random baseline (~−50 to −200).

---

## Running SwiftShelf++

### Option 1 — Docker (Recommended)

```bash
docker build -t swiftshelf-env .
docker run -p 7860:7860 swiftshelf-env
```

### Option 2 — Local (uv)

```bash
# Install dependencies with uv
pip install uv
uv sync

# Start FastAPI server
uv run server
# or equivalently:
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Option 3 — Local (pip)

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Live D3.js dashboard (HTML) |
| `GET` | `/health` | Health check → `{"status":"healthy","service":"swiftshelf-plus-plus"}` |
| `GET` | `/metadata` | Environment name, description, version |
| `GET` | `/schema` | JSON schemas for action, observation, state |
| `GET` | `/state` | Full current episode state |
| `GET` | `/tasks` | List of evaluation task metadata |
| `POST` | `/reset` | Reset environment → `{"observation": {...}, "info": {}}` |
| `POST` | `/step` | Execute action → `{observation, reward, terminated, truncated, info}` |
| `POST` | `/grade` | Run 3 heuristic episodes → `{"avg_reward": float, "status": "PASS"/"FAIL"}` |
| `POST` | `/mcp` | JSON-RPC 2.0 stub for validator compliance |

### Step Request / Response

**Request:**
```json
{"action": 3}
```

**Response:**
```json
{
  "observation": {
    "item_states": [...],
    "pending_orders": [...],
    "budget_remaining": 950.0,
    "near_expiry_count": 1,
    "expired_count": 0,
    "steps_elapsed": 1
  },
  "reward": 33.0,
  "terminated": false,
  "truncated": false,
  "info": {"orders_completed": 1, "expired_count": 0, "step": 1}
}
```

---

## SwiftShelfClient Usage

```python
from client import SwiftShelfClient
from models import InventoryAction

client = SwiftShelfClient("http://localhost:7860")

obs = client.reset()
print(obs.budget_remaining)   # 1000.0

result = client.step(InventoryAction(action=3))
print(result.reward)          # e.g. 33.0
print(result.terminated)      # False

state = client.state()
print(state.total_reward)     # cumulative reward
```

---

## Testing and Validation

### Install Test Dependencies

```bash
pip install gymnasium pytest
```

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

## Evaluation Tasks

| Task | Name | Success Condition | Score |
|---|---|---|---|
| 1 | Basic Fulfillment | `orders_completed >= 1` AND `expired_count == 0` | `0.99` (pass) / `0.01` (fail) |
| 2 | Waste Reduction | `expired_count <= 2` AND `orders_completed >= 1` | `0.99` (pass) / `0.01` (fail) |
| 3 | Efficiency Score | `(orders×20 − expired×15 − steps) / 100` | `float` clamped to `[0.01, 0.99]` |

---

## Evaluation Metrics

1. **Fulfillment Rate** — Number of orders dispatched per episode.
2. **Waste Management** — Items reaching `expiry_days <= 0` (penalised heavily).
3. **FEFO Compliance** — First-Expired-First-Out dispatch order rewarded with bonus.
4. **Efficiency Score** — Combined metric balancing orders completed, waste, and steps taken.

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | REST API server |
| `uvicorn` | ASGI server |
| `gymnasium` | RL environment base class |
| `numpy` | Numerical operations |
| `requests` | HTTP client (agent & client) |
| `pydantic` | Typed data models |
| `openai` | OpenAI-compatible LLM calls in inference.py |
| `httpx` | Async HTTP for keepalive ping |
| `openenv-core>=0.2.0` | OpenEnv validator compliance |

---

Built for the **Meta PyTorch OpenEnv Hackathon**  
Powered by `gymnasium`, `FastAPI`, and `facebook/opt-125m` via PyTorch.
