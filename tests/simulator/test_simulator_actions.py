import sys
import os
import pytest

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator", "streamwise"):
    from sim_types import Action
    from sim_types import ActionName
    from sim_types import GPUType
    from sim_types import Model
    from sim_types import Result


def test_action() -> None:
    action = Action(
        name=ActionName.ADD_DEVICE_REPLICA,
        model=Model.FLUX,
        gpu_type=GPUType.A100,
        models={GPUType.A100: {Model.FLUX: [2, 1]}},  # 2 devices, 1 replica
        action_result=Result(
            total_time_s=10.0,
            cost=5.0,
            total_energy=2.0,
        ),
        arrival_time_s=0.0,
    )
    assert action is not None
    assert action.model == Model.FLUX
    assert action.name == ActionName.ADD_DEVICE_REPLICA
    assert action.gpu_type == GPUType.A100
    assert action.models == {GPUType.A100: {Model.FLUX: [2, 1]}}
    assert action.time == 10.0
    assert action.cost == 5.0


def test_action_errors() -> None:
    with pytest.raises(TypeError, match="missing 5 required positional arguments"):
        Action()  # missing parameters

    with pytest.raises(ValueError, match="Model flux .* not supported"):
        Action(
            name=ActionName.ADD_DEVICE_REPLICA,
            model="flux",
            gpu_type=GPUType.A100,
            models={GPUType.A100: {Model.FLUX: [2, 1]}},  # 2 devices, 1 replica
            action_result=Result(
                total_time_s=10.0,
                cost=5.0,
            ),
            arrival_time_s=0.0,
        )

    with pytest.raises(ValueError, match="Device type A100 .* not supported"):
        Action(
            name=ActionName.ADD_DEVICE_REPLICA,
            model=Model.FT,
            gpu_type="A100",
            models={GPUType.A100: {Model.FT: [1, 1]}},  # 1 device, 1 replica
            action_result=Result(
                total_time_s=10.0,
                cost=5.0,
            ),
            arrival_time_s=0.0,
        )

    with pytest.raises(ValueError, match="Action name .* not supported"):
        Action(
            name="add_device_replica",
            model=Model.FT,
            gpu_type=GPUType.A100,
            models={GPUType.A100: {Model.FT: [1, 1]}},  # 1 device, 1 replica
            action_result=Result(
                total_time_s=10.0,
                cost=5.0,
            ),
            arrival_time_s=0.0,
        )
