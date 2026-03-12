# StreamWise Formulation

## Overview

This document describes the Mixed-Integer Optimization formulation used to optimally allocate GPU resources and compute time for parallel model execution in a streaming video generation pipeline.
This problem has multiple variations:
* Mixed Integer Linear Programming (MILP): optimization for raw cost, time, and energy.
* Mixed Integer Quadratic Constrained Programming (MIQCP): when enforcing bilinear equalities and optimizing for quadratic targets (e.g., TTFF $\times$ Cost).

## Problem Statement

Given a set of multi-modal AI models that must generate a video workflow, determine:
- Which GPU types and device counts to allocate to each model
- How to partition work across multiple instances (for parallelization)
- The optimal allocation that minimizes cost, latency, energy, or other objectives

---

## Sets

| Symbol | Description |
|--------|-------------|
| $G$ | Set of GPU types (e.g., H200, GB200, A100) |
| $M$ | Set of models (GEMMA, FLUX, HF, VAE, FT, UPSCALER, OTHERS) |
| $I$ | Set of instance IDs (0 to MAX_INSTANCES-1) |
| $D_m$ | Set of valid device counts for model $m$ |

---

## Decision Variables

### Primary Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| $x_{g,m,i,d}$ | Binary | Whether instance $i$ of model $m$ uses exactly $d$ devices of GPU type $g$ |
| $w_{g,m,i,d}$ | Non-negative Integer | Amount of work for instance $i$ using $d$ devices of type $g$ |
| $w_{g,m,i}$ | Non-negative Integer | Total work assigned to instance $i$ of model $m$ on GPU type $g$ |
| $n_{g,m,i}$ | Non-negative Integer | Number of GPUs allocated to instance $i$ of model $m$ on type $g$ |
| $a_{g,m,i}$ | Binary | Whether instance $i$ of model $m$ on type $g$ is active |
| $s_{g,m,i}$ | Binary | Whether instance $i$ is the min-TTFF instance for model $m$ on type $g$ |

### Time Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| $t_{g,m,i}$ | Non-negative Real | Execution time for instance $i$ of model $m$ on GPU type $g$ |
| $t_f_{g,m,i}$ | Non-negative Real | Time to first frame (TTFF) for instance $i$ of model $m$ |
| $T^{max}_m$ | Non-negative Real | Maximum time for model $m$ (across all instances) |
| $T$ | Non-negative Real | Total makespan (sum of max times per model) |

### Objective Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| $TTFF_{user}$ | Non-negative Real | User-perceived TTFF (sum of min TTFF across models) |
| $TTFF^{min}_m$ | Non-negative Real | Minimum TTFF for model $m$ |
| $C$ | Non-negative Real | Total cost (hourly GPU cost) |
| $E$ | Non-negative Real | Total energy consumption (Joules) |
| $S_g$ | Non-negative Integer | Number of servers needed for GPU type $g$ |

---

## Parameters

### Data Parameters

| Parameter | Description |
|-----------|-------------|
| $L_{g,m,d}$ | Latency per unit of work for model $m$ with $d$ devices of type $g$ |
| $P_{g,m,d}$ | Power consumption for model $m$ with $d$ devices of type $g$ (Watts) |
| $P^{idle}_g$ | Idle power for GPU type $g$ |
| $\text{cost}_g$ | Hourly cost per GPU of type $g$ |

### Workflow Parameters

| Parameter | Description |
|-----------|-------------|
| $W_m$ | Total work units required for model $m$ |
| $N_g$ | Total number of available GPUs of type $g$ |
| $S^{gpus}_g$ | Number of GPUs per server for type $g$ (from NUM_GPUS_PER_SERVER) |

---

## Constraints

### 1. Device Choice Constraints

#### Single device selection per instance
$$\sum_{d \in D_m \cup \{0\}} x_{g,m,i,d} = 1 \quad \forall g, m, i$$

Each instance must select exactly one device count (including 0 for inactive).

#### GPU count linkage
$$n_{g,m,i} = \sum_{d \in D_m \cup \{0\}} d \cdot x_{g,m,i,d} \quad \forall g, m, i$$

The number of GPUs equals the selected device count.

---

### 2. Work Allocation Constraints

#### Work must be selected device
$$w_{g,m,i} = \sum_{d \in D_m} w_{g,m,i,d} \quad \forall g, m, i$$

Work is allocated only through non-zero device choices.

#### Work bounded by device choice
$$w_{g,m,i,d} \leq W_m \cdot x_{g,m,i,d} \quad \forall g, m, i, d$$

Work for a device choice is bounded by the total work and the device selection.

#### If device is selected, work must be positive
$$w_{g,m,i} \geq \sum_{d \in D_m} x_{g,m,i,d} \quad \forall g, m, i$$

If any non-zero device is selected, at least 1 unit of work must be assigned.

---

### 3. Instance Activation Constraints

#### Active if work is assigned
$$a_{g,m,i} \leq w_{g,m,i} \quad \forall g, m, i$$

An instance can only be active if it has work.

#### GPUs require active instance
$$n_{g,m,i} \leq N_g \cdot a_{g,m,i} \quad \forall g, m, i$$

GPUs can only be allocated to active instances.

#### Active instance must have GPUs
$$n_{g,m,i} \geq a_{g,m,i} \quad \forall g, m, i$$

If instance is active, it must allocate at least 1 GPU.

#### No GPUs without active instance (for device 0)
$$w_{g,m,i} \leq W_m \cdot (1 - x_{g,m,i,0}) \quad \forall g, m, i$$

If 0 devices are selected, no work can be assigned.

---

### 4. Total Work Constraints

$$\sum_{g \in G, i \in I} w_{g,m,i} = W_m \quad \forall m \in M$$

All work for each model must be completed across all GPU types and instances.

---

### 5. GPU Resource Constraints

#### Total GPUs per type capacity
$$\sum_{m \in M, i \in I} n_{g,m,i} \leq N_g \quad \forall g \in G$$

Cannot exceed available GPUs of each type.

#### Server-based GPU allocation
$$\sum_{m \in M, i \in I} n_{g,m,i} = S_g \cdot S^{gpus}_g \quad \forall g \in G$$

Total GPUs must be a multiple of GPUs per server (for realistic infrastructure).

---

### 6. Time Constraints

#### Model-specific time linkage
$$t_{g,m,i} \leq T^{max}_m \quad \forall g, m, i$$

Each instance's time contributes to the model's maximum time.

#### Makespan definition (sequential execution)
$$T = \sum_{m \in M} T^{max}_m$$

Total makespan is the sum of maximum times per model (models run sequentially).

#### Model-specific time calculation
For each model, time is calculated based on model-specific latency:
$$t_{g,m,i} = \sum_{d \in D_m} L_{g,m,d} \cdot w_{g,m,i,d}$$

---

### 7. Time to first frame (TTFF) Constraints

#### TTFF per instance
$$t_{f_{g,m,i}} = \sum_{d \in D_m} x_{g,m,i,d} \cdot L_{g,m,d}^{first}$$

TTFF depends on the selected device count for the first unit of work.

#### TTFF selection constraint
$$TTFF^{min}_m \geq t_{f_{g,m,i}} - T^{max} \cdot (1 - s_{g,m,i}) \quad \forall g, m, i$$

The model's minimum TTFF is determined by one selected instance.

#### Single TTFF instance per model
$$\sum_{g \in G, i \in I} s_{g,m,i} = 1 \quad \forall m$$

Exactly one instance per model is selected as the TTFF representative.

#### Min TTFF instance must be active
$$s_{g,m,i} \leq a_{g,m,i} \quad \forall g, m, i$$

Only active instances can be TTFF instances.

---

### 8. User TTFF Definition

$$TTFF_{user} \geq \sum_{m \in M} TTFF^{min}_m$$

User-perceived TTFF is the sum of minimum TTFFs across all models.

$$TTFF_{user} \geq T - \text{total\_video\_seconds}$$

User TTFF accounts for parallel processing within the workflow.

---

### 9. Symmetry Breaking

$$n_{g,m,i} \geq n_{g,m,i+1} \quad \forall g, m, i < \text{MAX\_INSTANCES}-1$$

Instances are filled in order (lexicographic ordering) to reduce search space.

---

### 10. Cost Constraint (Optional)

$$C \leq C_{max}$$

Maximum budget constraint (if specified).

---

## Objective Functions

The model supports multiple objective functions via the allocator policy:

### TIME: Minimize Makespan
$$\min T$$

### TTFF: Minimize Time To First Frame
$$\min TTFF_{user}$$

### COST: Minimize Cost
$$\min C$$

### ENERGY: Minimize Energy
$$\min E$$

### TTFF_COST: Minimize TTFF × Cost (Bilinear)
$$\min TTFF_{user} \cdot C$$

Requires Gurobi solver.

### TIME_ENERGY: Minimize Makespan × Energy (Bilinear)
$$\min T \cdot E$$

Requires Gurobi solver.

### ENERGY_COST: Minimize Energy × Cost (Bilinear)
$$\min E \cdot C$$

Requires Gurobi solver.

---

## Cost Calculation

### Running Cost Mode (Per-Model Execution)
$$C = \sum_{g,m,i,d} L_{g,m,d} \cdot d \cdot w_{g,m,i,d} \cdot \frac{\text{cost}_g}{3600}$$

Cost is based on actual GPU execution time per model.

### Total Makespan Cost Mode
$$C = T \cdot \sum_{g} \text{cost}_g \cdot \sum_{m,i} n_{g,m,i} / 3600$$

Cost is allocated to the entire makespan duration.

---

## Energy Calculation

$$E = \sum_{g,m,i,d} L_{g,m,d} \cdot d \cdot w_{g,m,i,d} \cdot (P_{g,m,d} - P^{idle}_g) + \sum_g P^{idle}_g \cdot N_g \cdot T$$

Energy consists of:
1. **Active energy**: Model-specific power × execution time
2. **Idle energy**: Idle power × total GPUs × makespan

---

## Model Statistics

### Typical Model Size
- **Decision Variables**: ~2,000-3,000
  - Binary variables: ~1,000-1,500 (device choices, activation)
  - Integer variables: ~500-1,000 (work, GPU counts)
  - Continuous variables: ~200-300 (time, TTFF, objectives)

- **Constraints**: ~2,000-3,000
  - Device selection: O(|G| × |M| × |I| × |D_m|)
  - Activation: O(|G| × |M| × |I|)
  - Time: O(|G| × |M| × |I|)
  - Work: O(|G| × |M| × |I|)
  - Global: O(|G| + |M|)

### Typical Solve Times
- **HiGHS solver**: 10-50 seconds (with 50s timeout)
- **Gurobi solver**: 5-20 seconds (for quadratic objectives)

---

## Linearization Techniques

### Avoiding Bilinear Terms
The formulation avoids direct products of device choice and work by introducing:
- **work_device** variable: Linearizes the product of work and device selection
- **device_choice** variable: Binary indicator for each device option

This converts bilinear constraints to:
$$\text{bilinear term} = \text{linear constraint} + \text{linearization variable}$$

---

## Extensions and Notes

### Model-Specific Latencies
Each model has unique latency characteristics:
- **GEMMA**: First-scene + per-scene latencies
- **FLUX**: Per-step latency (num_steps_flux iterations)
- **HF/FT**: Frame-based latencies with disaggregation support
- **VAE**: Per-frame latency for disaggregated execution
- **UPSCALER**: Per-frame latency (optional)
- **OTHERS**: Minimal processing (Kokoro, YOLO)

### Dynamic Latency Ratio
When upscaler is disabled, latency ratio is adjusted:
$$\text{latency\_ratio} = \frac{\text{num\_pixels\_high}}{\text{num\_pixels\_medium}}$$

This accounts for higher resolution processing without upscaler.

### Server-Based Infrastructure
The constraint ensures realistic infrastructure where GPUs come in server units:
$$\text{GPUs used} = S_g \times S^{gpus}_g$$

For example, with 8 GPUs per server, only allocations of 0, 8, 16, 24, ... are valid.

---

## Implementation Details

### Solver Support
- **HiGHS**: Open-source MILP solver (free, good for linear objectives)
- **Gurobi**: Commercial solver (better for bilinear/quadratic objectives)

---

## Example Workflow

A typical 10-minute podcast video generation with 30 scenes:

| Model | Work Units | Typical Time (H100) | Max Parallel Instances |
|-------|-----------|-------------------|----------------------|
| GEMMA | 1 scene | 8-15s | 1 |
| FLUX | 1 scene | 120-180s | 4 (parallelized) |
| HF | 172 subscenes | 60-90s | 4 |
| FT | 30 subscenes | 600-800s | 10 |
| Others | 1 | 10-20s | 1 |
| **Total** | — | **~1000-1200s** | — |

With optimal parallelization and resource allocation, the MILP reduces makespan by 20-40% compared to greedy heuristics.
