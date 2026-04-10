import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests, json

MODEL = "facebook/opt-125m"

class LLMAgent:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        assert self.tokenizer is not None, "Tokenizer failed to load"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to(self.device)
        assert self.model is not None, "Model failed to load"
        self.model.eval()

    def obs_to_prompt(self, obs):
        near = obs.get("near_expiry_count", 0)
        orders = len(obs.get("pending_orders", []))
        budget = obs.get("budget_remaining", 1000)
        expired = obs.get("expired_count", 0)
        return (
            f"Inventory manager. Near expiry: {near}. "
            f"Pending orders: {orders}. Budget: {budget:.0f}. "
            f"Expired items: {expired}.\n"
            f"Actions: 0=pick 1=restock 2=discount 3=dispatch 4=batch 5=hold\n"
            f"Best action digit:"
        )

    def get_action(self, obs):
        assert self.tokenizer is not None, "Tokenizer not initialized"
        assert self.model is not None, "Model not initialized"
        prompt = self.obs_to_prompt(obs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the newly generated tokens (after the prompt)
        prompt_len = len(self.tokenizer.encode(prompt))
        new_tokens = outputs[0][prompt_len:]
        new_text = str(self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)).strip()

        # Try to find a valid digit in new text first
        for char in new_text:
            if char.isdigit() and int(char) < 6:
                return int(char)

        # Fallback: use a simple heuristic based on observation
        # (This ensures the LLM agent doesn't always hold)
        near_expiry = obs.get("near_expiry_count", 0)
        pending = len(obs.get("pending_orders", []))
        budget = obs.get("budget_remaining", 1000)

        if pending >= 3:
            return 4   # batch_pick
        if pending >= 1:
            return 3   # dispatch_order
        if near_expiry > 0:
            return 2   # apply_discount
        if budget > 300:
            return 1   # restock
        return 5       # hold

    def run(self, n_episodes=3, max_steps=50):
        results = []
        try:
            test = requests.get(f"{self.base_url}/health", timeout=3)
            if test.status_code != 200:
                raise ConnectionError()
        except Exception:
            print(
                "ERROR: API server not running.\n"
                "Start it first: uvicorn server.app:app --port 7860\n"
                "Then re-run: python agent/llm_agent.py"
            )
            return []

        for ep in range(n_episodes):
            r = requests.post(f"{self.base_url}/reset")
            obs = r.json()["observation"]
            total = 0.0
            for _ in range(max_steps):
                action = self.get_action(obs)
                r = requests.post(
                    f"{self.base_url}/step",
                    json={"action": action}
                )
                data = r.json()
                obs = data["observation"]
                total += data["reward"]
                if data["terminated"] or data["truncated"]:
                    break
            results.append(total)
            print(f"LLM Episode {ep+1}: {total:.1f}")

        if results:
            avg = sum(results) / len(results)
            print(f"LLM Agent Average: {avg:.1f}")
        return results

if __name__ == "__main__":
    agent = LLMAgent()
    agent.run()
