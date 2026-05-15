import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from model_provisioner.sim_types import Model
    from model_provisioner.sim_types import GPUType
    from model_provisioner.sim_types import ModelAllocation
    from model_provisioner.sim_types import ProvisioningResult

    from model_provisioner.utils import get_pareto_frontier
    from model_provisioner.utils import find_most_cost_effective_provisioning
    from model_provisioner.utils import find_most_energy_efficient_provisioning
    from model_provisioner.utils import find_pareto_frontier
    from model_provisioner.utils import coalesce_models

    from model_provisioner.models import FTModelAllocation


def test_get_pareto_frontier() -> None:
    """Create a Pareto frontier."""
    # 1 point
    paretor_frontier = get_pareto_frontier(
        [1],
        [1],
    )
    assert paretor_frontier.tolist() == [
        [1, 1],
    ]

    # 3 points
    paretor_frontier = get_pareto_frontier(
        [1, 2, 3],  # TTFF
        [1, 2, 2],  # Cost
    )
    assert paretor_frontier.tolist() == [
        [1, 2],
        [1, 1],
        [3, 1],
    ]

    # 4 points
    paretor_frontier = get_pareto_frontier(
        [1, 2, 3, 2],  # TTFF
        [1, 2, 2, 3],  # Cost
    )
    assert paretor_frontier.tolist() == [
        [1, 3],
        [1, 1],
        [3, 1],
    ]

    # 5 points with maxes
    paretor_frontier = get_pareto_frontier(
        [1, 2, 3, 3, 2],  # TTFF
        [1, 2, 2, 3, 3],  # Cost
        max_x=100,
        max_y=200,
    )
    assert paretor_frontier.tolist() == [
        [1, 100],
        [1, 3],
        [1, 1],
        [3, 1],
        [200, 1],
    ]


def test_find_most() -> None:
    provisioning = ProvisioningResult(
        latencies=[10.0, 20.0, 15.0],
        costs=[100.0, 80.0, 90.0],
        energies=[500.0, 400.0, 450.0],
        ttffs=[5.0, 10.0, 7.5],
        tbfs=[0.1, 0.2, 0.15],
        actual_provision=[
            {"A100": 8, "H100": 0},
            {"A100": 0, "H100": 8},
            {"A100": 4, "H100": 4},
        ],
        config_provision=[
            {"A100": 8, "H100": 0},
            {"A100": 0, "H100": 8},
            {"A100": 4, "H100": 4},
        ],
        model_provision=[],
    )

    idx = find_most_cost_effective_provisioning(provisioning)
    assert 0 <= idx < 3

    idx = find_most_energy_efficient_provisioning(provisioning)
    assert 0 <= idx < 3

    find_pareto_frontier([1], [1], [1])


def test_coalesce_models() -> None:
    models: dict[GPUType, dict[Model, list[ModelAllocation]]] = {}
    coalesced_models = coalesce_models(models)
    assert coalesced_models == {}

    models = {
        GPUType.A100: {
            Model.FT: [
                FTModelAllocation(gpu_type=GPUType.A100, devices=2, replicas=1),
                FTModelAllocation(gpu_type=GPUType.A100, devices=2, replicas=1),
            ],
        },
    }
    coalesced_models = coalesce_models(models)
    assert coalesced_models[GPUType.A100][Model.FT] == [
        FTModelAllocation(gpu_type=GPUType.A100, devices=2, replicas=2),
    ]
