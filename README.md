# Parkour Experiments with Action Acceleration Reward

This repository extends **Project Instinct** by introducing a new reward term:

> **Action Acceleration Penalty (`action_acc`)**

This term penalizes rapid changes in actions between timesteps, encouraging smoother and more physically realistic motions during humanoid parkour tasks.

---

## 🎯 What’s New

We added a reward component that minimizes action jerk:

* Encourages **smooth control signals**
* Reduces **high-frequency oscillations**
* Improves **stability during dynamic parkour movements**

Conceptually:

```
action_acc = ||a_t - a_{t-1}||
```

This term is incorporated into the overall reward as a penalty.

---

## 🎥 Example Parkour Videos

Below are three example rollouts demonstrating the effect of the new reward term.

### 1. Baseline (No Action Acc Penalty)

* Jerky movements
* Higher energy usage
* Less stable landings

👉 [Watch Video](./videos/parkour_baseline.mp4)

---

### 2. Moderate Action Acc Penalty

* Smoother transitions
* Improved balance
* More controlled jumps

👉 [Watch Video](./videos/parkour_action_acc_medium.mp4)

---

### 3. Strong Action Acc Penalty

* Very smooth motions
* Conservative but stable behavior
* Slightly reduced agility

👉 [Watch Video](./videos/parkour_action_acc_strong.mp4)

---

## 📊 Observations

| Setting          | Smoothness  | Stability   | Agility   |
| ---------------- | ----------- | ----------- | --------- |
| Baseline         | ❌ Low       | ⚠️ Medium   | ✅ High    |
| Moderate Penalty | ✅ Good      | ✅ High      | ✅ Good    |
| Strong Penalty   | ✅ Very High | ✅ Very High | ⚠️ Medium |

Key takeaway:

> There is a trade-off between **smoothness** and **agility**. A moderate penalty provides the best balance.

---

## ⚙️ How to Enable

Add the following reward implementation:

```python
def action_acc_l2(
    env: ManagerBasedRLEnv,
    action_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Second-order action smoothness (action acceleration penalty).
    
    Penalizes: -∑ⱼ(aⱼ,ₜ − 2aⱼ,ₜ₋₁ + aⱼ,ₜ₋₂)²
    """
    action = env.action_manager.action
    if action_ids is not None:
        action = action[:, action_ids]
    # Compute the second-order finite difference (acceleration)
    if not hasattr(env, "_last_action"):
        env._last_action = torch.zeros_like(action)
        env._last_last_action = torch.zeros_like(action)
    action_acc = action - 2 * env._last_action + env._last_last_action
    # Update the last actions
    env._last_last_action[:] = env._last_action
    env._last_action[:] = action
    # Compute the L2 penalty
    action_acc_l2 = torch.sum(torch.square(action_acc), dim=-1)  # (batch_size,)
    return action_acc_l2
```

Then register it in your reward config:

````python
reward_terms = {
    "action_acc": {
        "weight": -0.01,  # tune this value
    },
}
```python
reward_terms = {
    "action_acc": {
        "weight": -0.01,  # tune this value
    },
}
````

Make sure to compute the difference between consecutive actions inside your reward function.

---

## 🧪 Reproducing Results

Train using:

```bash
python scripts/instinct_rl/train.py \
  --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 \
  --headless
```

Adjust the `action_acc` weight in your config to reproduce different behaviors.

---

## 📌 Notes

* Too high penalty → overly conservative motions
* Too low penalty → unstable, jerky behaviors
* Works especially well for **parkour-style dynamic tasks**

---

## 📎 Related

* [Project Instinct](https://project-instinct.github.io/)
* [instinct_rl](https://github.com/project-instinct/instinct_rl)
* [instinctlab](https://github.com/project-instinct/instinctlab)

---

## License

This project follows the same license as Project Instinct (CC BY-NC 4.0).
