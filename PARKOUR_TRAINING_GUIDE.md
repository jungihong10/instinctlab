# Parkour Training Guide - Domain Randomization Setup

Based on GitHub issue feedback, the released parkour config has **removed critical domain randomizations** that you need to add back for successful sim-to-real transfer.

## ✅ What I've Added

### 1. **Camera Offset Randomization** (MOST CRITICAL)
Location: `EventCfg` in parkour_env_cfg.py

```python
randomize_camera_offsets = EventTerm(
    func=instinct_mdp.randomize_camera_offsets,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("camera"),
        "offset_pose_ranges": {
            "x": (-0.01, 0.01),      # ±1cm position error
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
            "roll": (-0.05, 0.05),   # ±2.86° orientation error
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        },
        "distribution": "uniform",
    },
)
```

**Why critical**: Real robot cameras are never perfectly calibrated. This randomizes the camera pose each reset to match real-world sensor errors.

### 2. **Advanced Depth Noise Models**
Location: `camera` sensor in SceneCfg

Added to noise_pipeline:
- **RangeBasedGaussianNoiseCfg**: Distance-dependent noise (farther = noisier)
- **DepthArtifactNoiseCfg**: Random depth artifacts (holes, spikes)

```python
"range_based_noise": RangeBasedGaussianNoiseCfg(
    ranges=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    stds=[0.005, 0.01, 0.02, 0.04, 0.08, 0.12],
),
"depth_artifact": DepthArtifactNoiseCfg(
    artifacts_prob=0.0001,
    artifacts_height_mean_std=[2, 0.5],
    artifacts_width_mean_std=[2, 0.5],
    noise_value=0.0,
),
```

### 3. **Actuator Gain Randomization**
```python
randomize_actuator_gains = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (0.8, 1.2),  # ±20%
        "damping_distribution_params": (0.9, 1.1),    # ±10%
    },
)
```

**Why important**: Real motors don't match specs exactly. This accounts for PD gain variations.

### 4. **Mass Randomization**
```python
randomize_rigid_body_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=["torso_link", "left_ankle.*", "right_ankle.*"],
        ),
        "mass_distribution_params": (0.85, 1.15),  # ±15%
    },
)
```

**Why important**: Accounts for payload variations and manufacturing tolerances.

## 🎚️ Tuning Recommendations

### Start Conservative, Then Increase
1. **Initial training** (first 10k iterations): Use the values I've set (moderate randomization)
2. **If sim-to-real gap remains**: Gradually increase ranges
3. **If training becomes unstable**: Reduce ranges

### Key Parameters to Tune

#### Camera Offset (Most Sensitive)
- **Position**: Start with ±1cm, adjust based on your actual calibration accuracy
- **Orientation**: Start with ±0.05 rad (±2.86°), critical for depth perception

```python
# If your camera calibration is very good:
"x": (-0.005, 0.005),  # ±5mm
"roll": (-0.03, 0.03),  # ±1.7°

# If your camera calibration is poor:
"x": (-0.02, 0.02),     # ±2cm
"roll": (-0.08, 0.08),  # ±4.6°
```

#### Depth Noise Stds
Increase for noisier real depth camera:
```python
stds=[0.01, 0.02, 0.04, 0.08, 0.16, 0.24],  # More aggressive
```

#### Actuator Gains
```python
# Conservative (good hardware):
"stiffness_distribution_params": (0.9, 1.1),  # ±10%

# Aggressive (variable hardware):
"stiffness_distribution_params": (0.7, 1.3),  # ±30%
```

## 🐛 Troubleshooting Training Issues

### Issue 1: Robot Won't Stand Still (takes small steps at velocity=0)

**Likely causes:**
1. `stand_still` reward weight too low (currently -0.3)
2. `dont_wait` penalty too aggressive (currently -0.5)

**Try increasing stand_still penalty:**
```python
stand_still = RewTerm(
    func=mdp.stand_still, 
    weight=-1.0,  # Increase from -0.3 to -1.0
    params={"command_name": "base_velocity", "offset": 4.0}
)
```

### Issue 2: Policy Fails at Stairs

**Likely causes:**
1. Insufficient depth noise (policy relies on perfect depth)
2. Camera randomization too weak
3. Need more training iterations (30k might not be enough)

**Solutions:**
1. Increase depth noise as shown above
2. Train for 50k+ iterations
3. Check if `volume_points_penetration` weight is appropriate (-4.0)

### Issue 3: Training Becomes Unstable

**Signs**: Reward drops suddenly, policy oscillates
**Causes**: Randomization too aggressive

**Solutions:**
1. Reduce camera offset ranges by 50%
2. Reduce actuator gain ranges to ±10%
3. Lower learning rate from 1e-3 to 5e-4

## 📊 Expected Training Results

With proper randomization:
- **Iterations 0-5k**: Policy learns basic standing and walking
- **Iterations 5k-15k**: Learns to navigate terrain, reward improves steadily
- **Iterations 15k-30k**: Refines stair climbing, handles obstacles
- **Iterations 30k+**: Fine-tuning, diminishing returns

**Minimum for deployment**: 30k iterations
**Recommended for robust performance**: 50k iterations

## 🚀 Training Command

```bash
python scripts/instinct_rl/train.py \
    --headless \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --num_envs=4096
```

With wandb logging (already configured):
```bash
wandb login  # First time only
python scripts/instinct_rl/train.py \
    --headless \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --num_envs=4096
```

## 📝 Notes from Maintainers

From GitHub issue #XX:
> "We removed some noise models and domain randomization in the training config. Due to hardware differences in real robots, please determine these parameters yourself."

> "The critical one is `randomize_camera_offsets`."

> "For perception depth, please refer to our paper."

## 🔧 Hardware-Specific Tuning

Since you have the same G1 29DOF as in the paper, these values should work well. However:

1. **Measure your camera calibration error** (if possible)
   - Use a checkerboard calibration
   - Set camera offset ranges based on actual error

2. **Test depth camera noise characteristics**
   - Capture depth images of stairs
   - Adjust noise models to match real sensor behavior

3. **Verify actuator response**
   - Log actual vs commanded joint positions
   - Adjust gain randomization accordingly

## 📚 Related Files Modified

1. `/home/jungi-hong/instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
   - Added camera offset randomization to EventCfg
   - Added depth noise to camera sensor
   - Added actuator and mass randomization

2. `/home/jungi-hong/instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
   - Configured wandb logging

3. `/home/jungi-hong/instinctlab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/rl_cfg.py`
   - Uncommented wandb configuration options

Good luck with training! 🎯
