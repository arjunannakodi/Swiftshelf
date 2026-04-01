import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests, json

MODEL = "facebook/opt-125m"

class LLMAgent:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to(self.device)
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
        prompt = self.obs_to_prompt(obs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract the digit after the prompt
        for ch in reversed(text):
            if ch.isdigit() and int(ch) < 6:
                return int(ch)
        return 5

    def run(self, n_episodes=3, max_steps=200):
        results = []
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
        avg = sum(results) / len(results)
        print(f"LLM Agent Average: {avg:.1f}")
        return results

if __name__ == "__main__":
    agent = LLMAgent()
    agent.run()
