from __future__ import annotations

from sim_types import Objective
from sim_types import Policy
from sim_types import GPUType
from sim_types import Model
from sim_types import Solver

from constants import GPU_RESERVED_COST
from constants import GPU_SPOT_COST


# Max devices for each model
# the logic is to allocate devices to each model proportional to their max devices
MAX_DEVICES = {
    Model.GEMMA: 8,
    Model.FLUX: 16,
    Model.HF: 40,
    Model.HF_VAE: 1,
    Model.FT: 40,
    Model.FT_VAE: 1,
}

# Max iterations for the optimization loop to prevent infinite loops in case of non-monotonic allocators or other issues
MAX_ITERATIONS = 100

# Set to True if we want to use up all GPUs if there's no further improvements in the greedy optimization loop
USE_ALL_GPUS = True

# Shorthand for enabling disaggregation for all supported models
STREAMWISE_POLICY = Policy(
    name="streamwise",
    gpu_cost=GPU_SPOT_COST,
    objective=Objective.TTFF_COST,
    disaggregation={
        Model.HF: True,
        Model.FT: False,
    },
    use_upscaler=True,
    hardware=list(GPUType),
)


STREAMWISE_MILP_POLICY = Policy(
    name="streamwise",
    gpu_cost=GPU_SPOT_COST,
    objective=Objective.TTFF_COST,
    disaggregation={
        Model.HF: True,
        Model.FT: False,
    },
    use_upscaler=True,
    hardware=list(GPUType),
    solver=Solver.GUROBI,
)


"""
HexGen policy configuration.
"""
HEXGEN_POLICY = Policy(
    name="hexgen",
    gpu_cost=GPU_RESERVED_COST,
    objective=Objective.TTFF,  # Does not account for cost
    disaggregation={
        Model.HF: True,
        Model.FT: False,
    },  # Dissagregation
    use_upscaler=False,
    hardware=[  # Multiple hardware
        GPUType.A100,
        GPUType.H100,
        GPUType.H200,
        GPUType.GB200,
    ],
    solver=Solver.HEXGEN,
)


"""
Helix policy configuration.
Reference: https://github.com/Thesys-lab/Helix-ASPLOS25
Optimizes models one-by-one following MODEL_ORDER using MILP.
"""
HELIX_POLICY = Policy(
    name="helix",
    gpu_cost=GPU_RESERVED_COST,
    objective=Objective.TTFF,  # Does not account for cost
    disaggregation={
        Model.HF: True,
        Model.FT: False,
    },
    use_upscaler=False,
    hardware=list(GPUType),
    solver=Solver.HELIX,
)


"""
DDiT policy configuration.
Reference: https://arxiv.org/html/2506.13497v1
"""
DDIT_POLICY = Policy(
    name="ddit",
    gpu_cost=GPU_RESERVED_COST,
    objective=Objective.TTFF,
    disaggregation={
        Model.HF: True,
        Model.FT: False,
    },
    use_upscaler=False,
    hardware=list(GPUType),
    solver=Solver.NAIVE,
)


STREAMWISE_ENERGY_POLICY = Policy(
    name="streamwise energy",
    gpu_cost=GPU_SPOT_COST,
    objective=Objective.TIME_ENERGY,
    disaggregation={
        Model.HF: True,
        Model.FT: False,
    },
    use_upscaler=True,
    hardware=list(GPUType),
)

NAIVE_POLICY = Policy(
    name="naive",
    gpu_cost=GPU_RESERVED_COST,
    objective=Objective.TTFF,
    disaggregation={},
    use_upscaler=False,
    hardware=[GPUType.A100],
    solver=Solver.NAIVE,
)


BASELINE_POLICIES = {
    "naive": NAIVE_POLICY,
    "naive disag": Policy(
        "naive disag",
        gpu_cost=GPU_RESERVED_COST,
        objective=Objective.TTFF,
        disaggregation={
            Model.HF: True,
            Model.FT: True,
        },
        use_upscaler=False,
        hardware=[GPUType.A100],
        solver=Solver.NAIVE,
    ),
    "naive upscaler": Policy(
        "naive upscaler",
        gpu_cost=GPU_RESERVED_COST,
        objective=Objective.TTFF,
        disaggregation={},
        use_upscaler=True,  # Changed to True
        hardware=[GPUType.A100],
        solver=Solver.NAIVE,
    ),
    "naive spot": Policy(
        "naive spot",
        gpu_cost=GPU_SPOT_COST,  # Changed to SPOT_COST
        objective=Objective.TTFF,
        disaggregation={},
        use_upscaler=False,
        hardware=[GPUType.A100],
        solver=Solver.NAIVE,
    ),
    "naive ttff*cost allocator": Policy(
        "naive ttff*cost allocator",
        GPU_RESERVED_COST,
        objective=Objective.TTFF_COST,  # Changed to TTFF_COST
        disaggregation={},
        use_upscaler=False,
        hardware=[GPUType.A100],
        solver=Solver.GREEDY,
    ),
    "naive hardware": Policy(
        "naive hardware",
        GPU_RESERVED_COST,
        objective=Objective.TTFF,
        disaggregation={},
        use_upscaler=False,
        hardware=list(GPUType),  # Changed hardware
        solver=Solver.NAIVE,
    ),
}


STREAMWISE_POLICIES = {
    "streamwise": STREAMWISE_POLICY,
    "streamwise no disag": Policy(
        name="streamwise no disag",
        gpu_cost=GPU_SPOT_COST,
        objective=Objective.TTFF_COST,
        disaggregation={},
        use_upscaler=True,
        hardware=list(GPUType),
        solver=Solver.GREEDY,
    ),
    "streamwise no upscaler": Policy(
        name="streamwise no upscaler",
        gpu_cost=GPU_SPOT_COST,
        objective=Objective.TTFF_COST,
        disaggregation={
            Model.HF: True,
            Model.FT: False,
        },
        use_upscaler=False,
        hardware=list(GPUType),
        solver=Solver.GREEDY,
    ),
    "streamwise no spot": Policy(
        name="streamwise no spot",
        gpu_cost=GPU_RESERVED_COST,
        objective=Objective.TTFF_COST,
        disaggregation={
            Model.HF: True,
            Model.FT: False,
        },
        use_upscaler=True,
        hardware=list(GPUType),
        solver=Solver.GREEDY,
    ),
    "streamwise naive allocator": Policy(
        name="streamwise naive allocator",
        gpu_cost=GPU_SPOT_COST,
        objective=Objective.TTFF,
        disaggregation={
            Model.HF: True,
            Model.FT: False,
        },
        use_upscaler=True,
        hardware=list(GPUType),
        solver=Solver.NAIVE,
    ),
    "streamwise A100": Policy(
        name="streamwise single hardware",
        gpu_cost=GPU_SPOT_COST,
        objective=Objective.TTFF_COST,
        disaggregation={
            Model.HF: True,
            Model.FT: False,
        },
        use_upscaler=True,
        hardware=[GPUType.A100],
        solver=Solver.NAIVE,
    ),
}
