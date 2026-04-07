import sys
import os
import pytest

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from constants import DEFAULT_WORKFLOW_CONFIG
    from constants import SECONDS_IN_HOUR

    from sim_types import GPUType
    from sim_types import Model

    from data_loading import load_latency_data
    from data_loading import load_power_data

    from evaluator import evaluate_model_allocation

    from policies import STREAMWISE_POLICY

    from models import FluxModelAllocation
    from models import GemmaModelAllocation
    from models import HFModelAllocation
    from models import HFVAEModelAllocation
    from models import FTModelAllocation
    from models import UpscalerModelAllocation
    from models import OthersModelAllocation

    from utils import to_models_df


def test_empty() -> None:
    """No models."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    with pytest.raises(AssertionError, match="Expected at least one instance of Model.GEMMA, but found 0"):
        evaluate_model_allocation(
            models={},
            num_gpus={GPUType.A100: 8},
            workflow=DEFAULT_WORKFLOW_CONFIG,
            latency_data=latency_data,
            power_data=power_data,
            policy=STREAMWISE_POLICY,
        )


def test_8A() -> None:
    """Test with 8 A100 GPUs."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    models = {
        GPUType.A100: {
            Model.GEMMA: [GemmaModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)],
            Model.FLUX: [FluxModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)],
            Model.HF: [HFModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)],
            Model.HF_VAE: [HFVAEModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)],
            Model.FT: [FTModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=2)],
            Model.UPSCALER: [UpscalerModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)],
            Model.OTHERS: [OthersModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)],   # + 1 for Kokoro/YOLO
        }
    }
    result = evaluate_model_allocation(
        models=models,
        num_gpus={GPUType.A100: 8},
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=STREAMWISE_POLICY,
    )
    assert to_models_df(models).to_string() == (
        "                Devices  Replicas  Work  #GPUs  Time (s)  TTFF (s)  Energy (kWh)  Cost ($)\n"
        "A100  gemma           1         1     0      1     33.20      5.94          0.38      4.74\n"
        "      flux            1         1     0      1      9.75      9.75          0.29      4.74\n"
        "      hf              1         1     0      1   1491.06      2.97          0.31      4.74\n"
        "      hf_vae          1         1     0      1    342.97      2.00          0.29      4.74\n"
        "      ft              1         2     0      2  11387.49    132.45          2.62      9.48\n"
        "      upscaler        1         1     0      1   2663.40     15.63          0.41      4.74\n"
        "      others          1         1     0      1     25.80      0.60          0.00      4.74\n"
        "TOTAL                 7         8     0      8  15953.67    169.34          4.30     37.92"
    )
    assert str(result) == (
        "Time:15953.68 s TTFF:15353.68 s Cost:$37.93 TTFF*Cost:582431.29 Energy:4.31 kWh GPUS: 8xA100"
    )

    assert result.gpus_used == {GPUType.A100: 8}
    assert result.gpus_total == {GPUType.A100: 8}
    _assert_equals_approx(result.total_time_s, 15953.68)
    _assert_equals_approx(result.ttff_s, 15353.68)
    _assert_equals_approx(result.first_chunk_time, 169.34)
    _assert_equals_approx(result.total_energy / SECONDS_IN_HOUR / 1000, 4.31)
    _assert_equals_approx(result.cost, 37.93)


def test_16H() -> None:
    """Test with 16 H200 GPUs."""
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    models = {
        GPUType.H200: {
            Model.GEMMA: [GemmaModelAllocation(
                gpu_type=GPUType.H200,
                devices=1, replicas=1)],
            Model.FLUX: [FluxModelAllocation(
                gpu_type=GPUType.H200,
                devices=1, replicas=1)],
            Model.HF: [HFModelAllocation(
                gpu_type=GPUType.H200,
                devices=2, replicas=2)],
            Model.HF_VAE: [HFVAEModelAllocation(
                gpu_type=GPUType.H200,
                devices=1, replicas=1)],
            Model.FT: [FTModelAllocation(
                gpu_type=GPUType.H200,
                devices=2, replicas=2)],
            Model.UPSCALER: [UpscalerModelAllocation(
                gpu_type=GPUType.H200,
                devices=1, replicas=2)],
            Model.OTHERS: [OthersModelAllocation(
                gpu_type=GPUType.H200,
                devices=1, replicas=1)],   # + 1 for Kokoro/YOLO
        }
    }
    result = evaluate_model_allocation(
        models=models,
        num_gpus={GPUType.H200: 16},
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=STREAMWISE_POLICY,
    )
    assert result is not None
    assert result.gpus_used == {GPUType.H200: 14}
    assert result.gpus_total == {GPUType.H200: 16}
    _assert_equals_approx(result.total_time_s, 4062.7)
    _assert_equals_approx(result.ttff_s, 3462.7)
    _assert_equals_approx(result.first_chunk_time, 51.68)
    _assert_equals_approx(result.total_energy / SECONDS_IN_HOUR / 1000, 2.87)
    _assert_equals_approx(result.cost, 66.67)


def test_cost_optimal() -> None:
    """
    Test with 256xA100 + 64xH200 GPUs.
    This is the cost-optimal configuration for the default workflow used in the paper.
    """
    latency_data = load_latency_data("simulator/data/")
    power_data = load_power_data("simulator/data/")

    models = {
        GPUType.A100: {
            Model.GEMMA: [GemmaModelAllocation(
                gpu_type=GPUType.A100,
                devices=8, replicas=1)],
            Model.FLUX: [FluxModelAllocation(
                gpu_type=GPUType.A100,
                devices=16, replicas=1)],
            Model.HF: [
                HFModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=2, replicas=6),
                HFModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=1, replicas=29),
            ],
            Model.HF_VAE: [HFVAEModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=20)],
            Model.FT: [
                FTModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=4, replicas=18),
                FTModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=2, replicas=12),
            ],
            Model.UPSCALER: [
                UpscalerModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=4, replicas=8),
                UpscalerModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=8, replicas=3),
                UpscalerModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=2, replicas=6),
                UpscalerModelAllocation(
                    gpu_type=GPUType.A100,
                    devices=1, replicas=6),
            ],
            Model.OTHERS: [OthersModelAllocation(
                gpu_type=GPUType.A100,
                devices=1, replicas=1)],
        },
        GPUType.H200: {
            Model.HF: [HFModelAllocation(
                gpu_type=GPUType.H200,
                devices=2, replicas=4)],
            Model.HF_VAE: [HFVAEModelAllocation(
                gpu_type=GPUType.H200,
                devices=1, replicas=4)],
            Model.FT: [
                FTModelAllocation(
                    gpu_type=GPUType.H200,
                    devices=2, replicas=13),
                FTModelAllocation(
                    gpu_type=GPUType.H200,
                    devices=24, replicas=1),
            ],
            Model.UPSCALER: [UpscalerModelAllocation(
                gpu_type=GPUType.H200,
                devices=2, replicas=1)],
        }
    }
    result = evaluate_model_allocation(
        models=models,
        num_gpus={GPUType.A100: 256, GPUType.H200: 64},
        workflow=DEFAULT_WORKFLOW_CONFIG,
        latency_data=latency_data,
        power_data=power_data,
        policy=STREAMWISE_POLICY,
    )

    assert to_models_df(models).to_string() == (
        "                Devices  Replicas  Work  #GPUs  Time (s)  TTFF (s)  Energy (kWh)  Cost ($)\n"
        "A100  gemma           8         1     0      8      8.57      1.48          0.24      0.72\n"
        "      flux           16         1     0     16      0.95      0.95          0.09      1.45\n"
        "      hf              2         6     0     12     29.56      2.03          0.08      1.09\n"
        "      hf              1        29     0     29     29.30      2.97          0.17      2.62\n"
        "      hf_vae          1        20     0     20     11.75      2.00          0.11      1.81\n"
        "      ft              4        18     0     72    192.62     58.82          1.61      6.51\n"
        "      ft              2        12     0     24    192.61     79.76          0.54      2.17\n"
        "      upscaler        4         8     0     32     34.75      3.97          0.23      2.90\n"
        "      upscaler        8         3     0     24     34.75      2.02          0.17      2.17\n"
        "      upscaler        2         6     0     12     34.76      7.86          0.08      1.09\n"
        "      upscaler        1         6     0      6     34.77     15.63          0.04      0.54\n"
        "      others          1         1     0      1     25.80      0.60          0.00      0.09\n"
        "H200  hf              2         4     0      8     29.49      0.90          0.06      2.85\n"
        "      hf_vae          1         4     0      4     11.75      0.87          0.03      1.43\n"
        "      ft              2        13     0     26    193.00     34.87          0.81      9.28\n"
        "      ft             24         1     0     24    191.93     14.78          0.72      8.56\n"
        "      upscaler        2         1     0      2     34.75      3.87          0.02      0.71\n"
        "TOTAL                81       134     0    320    304.40     21.60          5.00     45.99"
    )
    assert str(result) == (
        "Time:304.40 s TTFF:21.60 s Cost:$46.00 TTFF*Cost:993.61 "
        "Energy:5.01 kWh GPUS: 256xA100+64xH200"
    )

    assert result.gpus_used == {
        GPUType.A100: 256,
        GPUType.H200: 64,
    }
    assert result.gpus_total == {
        GPUType.A100: 256,
        GPUType.H200: 64,
    }
    _assert_equals_approx(result.total_time_s, 304.40)
    _assert_equals_approx(result.ttff_s, 21.60)
    _assert_equals_approx(result.first_chunk_time, 21.60)
    _assert_equals_approx(result.total_energy / SECONDS_IN_HOUR / 1000, 5.01)
    _assert_equals_approx(result.cost, 46.00)

    assert models[GPUType.A100][Model.OTHERS][0].devices == 1
    assert models[GPUType.A100][Model.OTHERS][0].replicas == 1
    assert models[GPUType.A100][Model.OTHERS][0].time_first == 0.60

    assert models[GPUType.H200][Model.FT][1].devices == 24
    assert models[GPUType.H200][Model.FT][1].replicas == 1
    _assert_equals_approx(models[GPUType.H200][Model.FT][1].time, 191.93)
    _assert_equals_approx(models[GPUType.H200][Model.FT][1].time_first, 14.78)
    _assert_equals_approx(models[GPUType.H200][Model.FT][1].energy / SECONDS_IN_HOUR / 1000, 0.72)
    _assert_equals_approx(models[GPUType.H200][Model.FT][1].cost, 8.56)


def _assert_equals_approx(
    a: float,
    b: float,
    tol: float = 0.01
) -> None:
    assert abs(a - b) < tol, f"Expected {a:.2f} to be approximately equal to {b:.2f} within tolerance {tol}"
