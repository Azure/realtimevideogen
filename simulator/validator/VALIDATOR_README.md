# Simulator Validator

The **validator** provides ground-truth test data for verifying external simulators that mimic the StreamWise multi-modal workflow system. It uses the current simulator implementation as the authoritative source to produce expected outputs (cost, TTFF, TBF, total time) for a comprehensive set of GPU allocation scenarios.

## Purpose

If you are building a simulator (e.g., an LLM-generated simulator, a simplified analytical model, or a custom scheduling simulator) that takes GPU allocations as input and produces time/cost metrics, this validator lets you verify your simulator's correctness against the real StreamWise simulator.

## Key Concepts

### What the Simulator Does

The StreamWise simulator models a **multi-modal video generation workflow** (StreamCast) with these model components:

| Component | Key (`model` field) | Description |
|-----------|-------------------|-------------|
| Gemma (LLM) | `gemma` | Generates scene descriptions and narration |
| Flux (Image Gen) | `flux` | Creates keyframe images from descriptions |
| HunyuanFramePack (Video Gen) | `hf` | Generates video frames from keyframes |
| HunyuanFramePack VAE | `hf_vae` | Decodes latent video representations |
| FantasyTalking (Lip-Sync) | `ft` | Synchronizes lip movements with audio |
| FantasyTalking VAE | `ft_vae` | **Not used** (disaggregation disabled for FT) |
| Upscaler | `upscaler` | Enhances final video resolution |
| Others (YOLO + Kokoro) | `others` | Object detection + audio generation |

### What the Validator Tests

Given a GPU allocation (how many GPUs of what type are assigned to each component, with how many devices per instance and how many replicas), the simulator produces:

| Metric | Description |
|--------|-------------|
| `cost` | Total GPU cost in USD (based on spot pricing × GPU-hours) |
| `ttff_s` | Time To First Frame in seconds |
| `tbf_s` | Time Between Frames in seconds (after the first frame) |
| `total_time_s` | Total end-to-end execution time in seconds |

## How to Use

### 1. Understanding the Ground Truth File

The file `simulator/validator_ground_truth.json` contains 44 pre-computed scenarios. Each scenario has:

```json
{
  "8xA100_minimal": {
    "allocations": [
      {"gpu_type": "A100", "model": "gemma", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "flux", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "hf", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "hf_vae", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "ft", "devices": 1, "replicas": 2},
      {"gpu_type": "A100", "model": "upscaler", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "others", "devices": 1, "replicas": 1}
    ],
    "num_gpus": {"A100": 8},
    "expected": {
      "cost": 37.9343,
      "ttff_s": 15353.6813,
      "tbf_s": 1.1505,
      "total_time_s": 15953.6813
    }
  }
}
```

### 2. Implementing Your Simulator

Your simulator must accept the same allocation specification and produce the same 4 metrics. Here's what each allocation field means:

- **`gpu_type`**: The GPU hardware type — one of `"A100"`, `"H100"`, or `"H200"`
- **`model`**: Which workflow component this allocation is for
- **`devices`**: Number of GPUs used per instance (tensor/sequence parallelism)
- **`replicas`**: Number of independent instances of this model running in parallel

**Key constraints your simulator should know:**
- `hf_vae` always has `devices=1` (no tensor parallelism for VAE)
- `ft_vae` is NOT used (FantasyTalking VAE disaggregation is disabled)
- `others` always has `devices=1`
- Valid device counts vary per model (see `DEVICE_OPTIONS` in `simulator/constants.py`)
- Total GPUs used must not exceed `num_gpus` for each GPU type

**Multi-allocation (heterogeneous GPU) support:**

A single model can have allocations on **multiple GPU types** simultaneously. For example, `hf` may run 4 replicas on A100 and 3 replicas on H200 in the same scenario:

```json
{
  "hetero_24H200_40A100_hf_split": {
    "allocations": [
      {"gpu_type": "A100", "model": "gemma", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "flux", "devices": 2, "replicas": 1},
      {"gpu_type": "A100", "model": "others", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "hf", "devices": 4, "replicas": 4},
      {"gpu_type": "A100", "model": "hf_vae", "devices": 1, "replicas": 1},
      {"gpu_type": "A100", "model": "ft", "devices": 2, "replicas": 6},
      {"gpu_type": "A100", "model": "upscaler", "devices": 1, "replicas": 4},
      {"gpu_type": "H200", "model": "hf", "devices": 4, "replicas": 3},
      {"gpu_type": "H200", "model": "hf_vae", "devices": 1, "replicas": 1},
      {"gpu_type": "H200", "model": "ft", "devices": 2, "replicas": 2},
      {"gpu_type": "H200", "model": "upscaler", "devices": 1, "replicas": 2}
    ],
    "num_gpus": {"A100": 40, "H200": 24},
    "expected": { ... }
  }
}
```

Key rules for multi-allocation scenarios:
- `SINGLE_INSTANCE_MODELS` (`gemma`, `flux`, `others`) must be assigned to exactly one GPU type
- `hf`, `hf_vae`, `ft`, and `upscaler` may appear on multiple GPU types
- Each GPU type's budget (`num_gpus`) is tracked independently

### 3. Running Validation

#### Option A: Programmatic (recommended for automated testing)

```python
import sys
sys.path.insert(0, "simulator")
sys.path.insert(0, "streamwise")

from validator import load_ground_truth, validate, validate_all_passed

# Load the ground truth
ground_truth = load_ground_truth()

# Run your simulator on each scenario and collect results
my_results = {}
for scenario_name, scenario_data in ground_truth.items():
    allocations = scenario_data["allocations"]
    num_gpus = scenario_data["num_gpus"]

    # YOUR SIMULATOR CALL HERE:
    result = my_simulator(allocations, num_gpus)

    my_results[scenario_name] = {
        "cost": result.cost,
        "ttff_s": result.ttff_s,
        "tbf_s": result.tbf_s,
        "total_time_s": result.total_time_s,
    }

# Validate against ground truth (1% relative tolerance)
results = validate(my_results, ground_truth)
for r in results:
    if not r.passed:
        print(f"FAIL: {r.scenario_name}")
        for err in r.errors:
            print(f"  {err}")

# Or simply:
assert validate_all_passed(my_results, ground_truth)
```

#### Option B: CLI with JSON file

1. Save your simulator outputs as a JSON file with the same structure:

```json
{
  "8xA100_minimal": {"cost": 37.93, "ttff_s": 15353.68, "tbf_s": 1.15, "total_time_s": 15953.68},
  "16xH200_balanced": {"cost": 66.67, "ttff_s": 3462.70, "tbf_s": 0.29, "total_time_s": 4062.70},
  ...
}
```

2. Run validation:

```bash
cd <repo_root>
python -c "
import sys; sys.path.insert(0, 'simulator'); sys.path.insert(0, 'streamwise')
from validator import validate
import json
with open('my_results.json') as f:
    results = validate(json.load(f))
for r in results:
    status = 'PASS' if r.passed else 'FAIL'
    print(f'{status}: {r.scenario_name}')
    for err in r.errors:
        print(f'  {err}')
"
```

### 4. Regenerating Ground Truth

If the simulator code changes (e.g., updated latency profiles), regenerate:

```bash
cd <repo_root>
python -c "
import sys; sys.path.insert(0, 'simulator'); sys.path.insert(0, 'streamwise')
from validator import generate_ground_truth
generate_ground_truth()
"
```

This overwrites `simulator/validator_ground_truth.json`.

## Scenario Coverage

The 44 scenarios cover diverse configurations:

| Category | Count | GPU Counts |
|----------|-------|------------|
| Single server (8 GPUs) | 3 | A100, H100, H200 |
| Two servers (16 GPUs) | 6 | All types, balanced/FT-heavy/HF-heavy/high-TP |
| Three servers (24 GPUs) | 3 | H100, A100, H200 |
| Four servers (32 GPUs) | 3 | H200, A100, H100 |
| Five servers (40 GPUs) | 2 | H200, A100 |
| Six servers (48 GPUs) | 2 | H100, H200 |
| Eight servers (64 GPUs) | 2 | H200, A100 |
| Mixed GPU types | 6 | A100+H200, A100+H100, H100+H200, triple |
| High tensor parallelism | 3 | TP=8, TP=16 configurations |
| High replica count | 3 | Up to 16 replicas |
| Upscaler-heavy | 2 | Focused on upscaler replicas |
| LLM-focused (Gemma/Flux TP) | 2 | High TP for LLM/image models |
| Heterogeneous multi-allocation | 7 | Same model on multiple GPU types |

## Tolerance

The validator uses **1% relative error** for all metrics. This means:
- If expected `cost = $37.93`, any value between `$37.55` and `$38.31` passes.
- If expected `ttff_s = 0.0`, actual must also be exactly `0.0`.

## Workflow Configuration

All scenarios use the **default workflow configuration** (`DEFAULT_WORKFLOW_CONFIG`):
- 10-minute video (600 seconds)
- 43 scenes, 172 subscenes
- 30 FPS for HunyuanFramePack, 23 FPS for FantasyTalking
- 20K input tokens for Gemma
- HF disaggregation enabled (separate HF_VAE)
- FT disaggregation disabled (FT_VAE not used)
- Upscaler enabled
- Spot GPU pricing

## File Structure

```
simulator/
├── validator.py                    # Validator implementation
├── validator_ground_truth.json     # Pre-computed ground truth (37 scenarios)
└── VALIDATOR_README.md             # This file

tests/simulator/
└── test_validator.py               # Tests for the validator itself
```
