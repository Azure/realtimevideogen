# Multimodal Workflow Simulator

This directory contains code for running large-scale simulations of multimodal workflows to estimate costs, energy consumption, and resource allocation strategies.
The simulator uses **StreamCast** as a reference example, i.e., a video generation workflow composed of multiple generative model components including LLMs, audio generation, image generation, video generation, and lip-sync.

## Overview

The simulator helps answer critical questions for deploying multimodal AI workflows:
- **Cost Estimation**: What is the total cost to run X requests using Y GPU configurations?
- **Energy Estimation**: How much energy will the workflow consume across different hardware?
- **Resource Allocation**: How should GPUs be distributed across different model components to minimize latency and cost?

The simulator operates on profiled data (latency and power measurements) from different GPU types (A100, H100, H200) and uses optimization algorithms to determine optimal resource allocation.

## Directory Structure

```
simulator/
├── README.md                          # This file
├── algorithms.py                      # Core resource allocation and scheduling algorithms
├── algorithms_baseline.py             # Baseline scheduling strategies for comparison
├── algorithms_power.py                # Power-related scheduling algorithms
├── constants.py                       # Workflow configuration constants
├── data_loading.py                    # Data loading utilities for profiles
├── utils.py                           # Helper utilities
├── cost_estimator.ipynb               # Notebook for cost analysis and estimation
├── cost_estimator_baselines.ipynb     # Notebook comparing baseline strategies
├── energy_estimator.ipynb             # Notebook for energy consumption analysis
├── data/                              # Latency and power profile data
│   ├── latency_*_*.csv                # Latency profiles per component and GPU type
│   ├── power_*_*.csv                  # Power consumption profiles
│   └── *.csv                          # Other benchmark data
└── results/                           # Simulation output results will be located here
```

## Key Components

### 1. Notebooks

The simulator provides three main notebooks for analysis:

#### `cost_estimator.ipynb`
- **Purpose**: Estimate total cost of running workloads on different GPU configurations
- **Process**:
  1. Load latency profiles for each workflow component
  2. Configure workflow parameters (number of scenes, frames, etc.)
  3. Run resource allocation algorithm to determine optimal GPU distribution
  4. Calculate total cost based on GPU pricing and execution time
- **Use Case**: Budgeting for production deployments, comparing hardware options

#### `cost_estimator_baselines.ipynb`
- **Purpose**: Compare different baseline scheduling strategies
- **Process**:
  1. Load latency profiles
  2. Run multiple baseline algorithms (equal allocation, greedy, FCFS, etc.)
  3. Compare results against optimized strategy
- **Use Case**: Validating algorithm improvements, understanding trade-offs

#### `energy_estimator.ipynb`
- **Purpose**: Estimate energy consumption and carbon footprint
- **Process**:
  1. Load both latency and power profiles
  2. Run power-aware scheduling algorithms
  3. Calculate total energy consumption across workflow stages
- **Use Case**: Energy consumption analysis, comparing energy efficiency

### 2. Algorithms (`algorithms.py`)

Core optimization algorithms for resource allocation and scheduling across multimodal components:

- **Resource Allocation**: Determines how to distribute given heterogeneous GPUs across different model components (LLM, image gen, video gen, lip-sync, etc.)
- **Scheduling**: Decides prioritization to minimize overall latency
- **Optimization Strategy**: Iteratively assigns GPUs to stages that yield largest return to reduce end-to-end time

### 3. Data Files (`data/`)

Contains empirically measured latency and power profiles:

- **Latency Profiles**: `latency_<component>_<gpu_type>.csv`
  - Measured execution time for each component on different GPU counts
  - Components: gemma (LLM), flux (image gen), hf (HunyuanFramePack video gen), ft (FantasyTalking lip-sync), upscaler

- **Power Profiles**: `power_<component>_<gpu_type>.csv`
  - Measured power consumption during execution
  - Includes both standard and high-power configurations

- **GPU Types**: A100, H100, H200

### 4. Supporting Files

- **`constants.py`**: Workflow configuration parameters (number of scenes, frames per scene, model hyperparameters)
- **`data_loading.py`**: Utilities to load and parse model (component) profile CSV files
- **`algorithms_baseline.py`**: Baseline implementations for comparison
- **`algorithms_power.py`**: Algorithm variants with energy estimation

## How to Use

### Running Cost Estimation

1. Open `cost_estimator.ipynb` in Jupyter
2. Configure workflow parameters or reuse the default config:
   ```python
   workflow_config = {
       "total_video_seconds": 600,  # 10 minute video
       "total_scenes": 43,          # derived from components' supported FPS
       ...
   }
   ```
3. Run all cells to get cost estimates and resource allocation plan

### Running Energy Estimation

1. Open `energy_estimator.ipynb`
2. Configure similar workflow parameters
3. Run all cells to get energy consumption

### Modifying Algorithms

To experiment with different scheduling strategies:

1. Edit `algorithms.py` to implement your allocation logic
2. Re-run the notebooks to see the impact on cost/energy
3. Compare against baselines in `algorithms_baseline.py`

## Example Workflow: StreamCast

The simulator models a video generation workflow with these stages:

1. **LLM (Gemma)**: Generate scene descriptions and narration
2. **Image Generation (Flux)**: Create key frame images from descriptions
3. **Video Generation (HunyuanFramePack)**: Generate video frames from keyframes
4. **VAE Decoding**: Decode latent representations to pixels
5. **Lip-Sync (FantasyTalking)**: Synchronize lip movements with audio
6. **Upscaling**: Enhance final video resolution

Each stage can be parallelized differently, and the simulator finds the optimal GPU allocation to minimize latency and cost.

## Output

Simulation results include:

- **Optimal GPU Allocation**: How many GPUs assigned to each component
- **Stage Latencies**: Execution time for each workflow stage
- **Total Execution Time**: End-to-end latency
- **Total Cost**: Dollar cost based on GPU pricing
- **Energy Consumption**: Total kWh and carbon emissions (for energy estimator)
- **Bottleneck Analysis**: Which stage limits overall throughput

## Citation

If you use this simulator in your research, please cite our work:

```bibtex
[Citation information to be added]
```

## License

See [LICENSE](../LICENSE) in the root directory.
