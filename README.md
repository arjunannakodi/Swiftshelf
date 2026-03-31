# SwiftShelf++: Inventory Decision Environment 📦🚀

![](https://img.shields.io/badge/Meta-OpenEnv-brightgreen)
![](https://img.shields.io/badge/PyTorch-Hackathon-blue)
![](https://img.shields.io/badge/Python-3.11-yellow)

**SwiftShelf++** is a complete, OpenEnv-compliant reinforcement learning environment simulating a real-world inventory management system. It's designed for the Meta PyTorch OpenEnv Hackathon with a focus on **correctness**, **stability**, and **scalability**.

## 🏗️ Project Architecture

```
project/
├── env/environment.py   # Core RL Environment (OpenEnv Style)
├── api/server.py        # FastAPI API Server (port 8000)
├── tasks.py             # OpenEnv Task Definitions
├── grader.py            # Automated Performance Evaluator
├── tests/test_env.py    # PyTest Suite
├── Dockerfile           # Optimized Container Build
├── requirements.txt     # Dependency Specification
└── README.md            # Comprehensive Documentation
```

## 🎮 Environment Features

*   **Real-world Logistics Models**: Item freshness (expiry), order fulfillment deadlines, and dynamic budget constraints.
*   **Action Space (Discrete)**:
    1. `pick_item`: Select item using FEFO logic.
    2. `restock`: Replenish inventory (cost involved).
    3. `apply_discount`: Reduce price of near-expiry stock.
    4. `dispatch_order`: Fulfill customer orders.
    5. `batch_pick`: Multi-order selection optimization.
    6. `hold`: Strategic inactivity.
*   **Dense Reward Function**: Balanced for positive (completed orders, efficiency) and negative (expiry, missed deadlines, unsafe dispatch) outcomes.

## 🚀 Running SwiftShelf++

### Using Docker (Recommended)

```bash
docker build -t swiftshelf-env .
docker run -p 8000:8000 swiftshelf-env
```

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## 📮 API Endpoints

*   `GET  /health` : System status check.
*   `POST /reset`  : Reinitialize environment.
*   `GET  /state`  : Retrieve current observation.
*   `POST /step`   : Execute action (input: `{"action": int}`).
*   `POST /grade`  : Run automated performance assessment.

## Evaluation Metrics

SwiftShelf++ uses three primary metrics to evaluate agent performance:

1.  **Fulfillment Rate**: Tracks the number of customer orders successfully dispatched. High fulfillment directly increases rewards.
2.  **Waste Management (Expired Items)**: Measures the number of items that reach expiry in the inventory. Agents must optimize stock rotation to minimize this.
3.  **Efficiency Score**: A combined metric calculated as:
    `efficiency_score = (orders_completed * 20) - (expired_count * 15) - steps_elapsed`
    This score penalizes both waste and time while rewarding quick fulfillment.

### 📊 Example Grader Output

```text
=================================================================
Episode    | Reward     | Expired    | Orders     | Steps     
-----------------------------------------------------------------
1          | -245.00    | 12         | 4          | 50        
2          | -312.40    | 14         | 2          | 50        
3          | -280.00    | 12         | 3          | 50        
-----------------------------------------------------------------
AVERAGE    | -279.13    | 12.67      | 3.00       | 50.00     
=================================================================

RESULT: PASS ✅
Confidence Level: High (Avg Reward -279.13 > -500.0, Avg Waste 12.67 < 15.0)
```

## 🧪 Testing and Validation

## 🏆 Hackathon Tasks

1.  **Order Fulfillment**: Successfully complete at least one shipment.
2.  **Expiry Minimization**: Optimize shelf-life management to reduce waste.
3.  **Maximum Efficiency**: Balance stock rotation with order throughput.

---

Built for **Meta PyTorch OpenEnv Hackathon**. Crafted with ❤️ for Senior RL Researchers.
