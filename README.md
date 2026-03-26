# Parkour Experiments with Action Acceleration Reward

This repository extends **Project Instinct** by introducing a new reward term:

> **Action Acceleration Penalty (`action_acc`)**

This term penalizes rapid changes in actions between timesteps, encouraging smoother and more physically realistic motions during humanoid parkour tasks.

---

## 🎯 What’s New

We distinguish between two types of action smoothness:

### 1️⃣ First-Order Smoothness (Velocity Penalty)

Penalizes rapid changes between consecutive actions:

```
- ∑ⱼ (aⱼ,t − aⱼ,t−1)²
```

* Reduces **jerky control signals**
* Encourages **temporal consistency**

---

### 2️⃣ Second-Order Smoothness (Acceleration Penalty)

Penalizes changes in action differences (i.e., action acceleration):

```
- ∑ⱼ (aⱼ,t − 2aⱼ,t−1 + aⱼ,t−2)²
```

* Encourages **even smoother trajectories**
* Reduces **high-frequency oscillations further**
* Leads to more **physically realistic motion**

---

In this project, we focus on **second-order smoothness (`action_acc`)**, which provides stronger regularization for dynamic tasks like parkour.

---

## 🎥 Example Parkour Videos

We showcase three representative parkour scenarios using the **action acceleration reward**:

### 1. Descending Stairs

* Stable foot placement while going down
* Reduced jitter during contact transitions
* Improved balance over multiple steps

👉 [Watch Video](./parkour_down_stairs.mp4)

---

### 2. Gap Jump (Across Empty Space)

* Controlled takeoff and landing
* Smooth coordination during flight phase
* Reduced impact instability

👉 [Watch Video](./parkour_gap_jump.mp4)

---

### 3. Climbing a Large Step

* Strong but stable push-off
* Smooth weight transfer
* Reliable recovery after ascent

👉 [Watch Video](./parkour_big_step_up.mp4)

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
