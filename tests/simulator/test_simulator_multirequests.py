import sys
import os

# Add current path
sys.path.append(os.getcwd())

from tests.test_utils import assert_equals_approx
from tests.test_utils import temp_sys_path

with temp_sys_path("simulator"):
    from multirequests import QPM_LIST
    from multirequests import get_replicas
    from multirequests import get_costs
    from multirequests import get_total_costs
    from multirequests import required_replicas
    from multirequests import aggregate_time_per_request_by_quality
    from multirequests import TIME_PER_REQ
    from multirequests import INIT_REPLICAS
    from multirequests import INIT_REPLICAS_BASELINE
    from multirequests import QUALITY_PORTIONS
    from multirequests import TIME_PER_REQ_ADAPTIVE
    from multirequests import get_time_per_request_baseline

    from data_loading import load_latency_data
    from workflows import PODCAST_WORKFLOW

    from constants import GPU_SPOT_COST
    from sim_types import GPUType
    from sim_types import Model


def test_multirequests() -> None:
    video_minutes = 10
    video_seconds = video_minutes * 60

    replicas = get_replicas(video_seconds=video_seconds)
    assert len(replicas)

    costs = get_costs(
        replicas=replicas,
        gpu_costs=GPU_SPOT_COST,
    )

    total_costs = get_total_costs(costs)

    assert len(total_costs) == len(QPM_LIST) == 9


def test_required_replicas_scenes_partition() -> None:
    """Test required_replicas with scenes partition."""
    result = required_replicas(
        name="test",
        video_seconds=600,
        ttff=10.0,
        per_sec=0.5,
        partition="scenes",
        req_per_min=1.0,
    )
    # ttff_total = 10.0, per_sec_total = 600 * 0.5 = 300
    # total_time_per_request = 310, total_time_per_minute = 310 * 1.0 = 310
    # expected = 310 / 60
    assert_equals_approx(result, 310 / 60)


def test_required_replicas_frames_partition_vae() -> None:
    """Test required_replicas with frames partition for VAE."""
    result = required_replicas(
        name="hf_vae",
        video_seconds=600,
        ttff=5.0,
        per_sec=0.1,
        partition="frames",
        req_per_min=2.0,
    )
    # ttff_total = 5.0, per_sec_total = 600 * 0.1 = 60
    # total_time_per_request = 65, total_time_per_minute = 65 * 2.0 = 130
    # expected = 130 / 60
    assert_equals_approx(result, 130 / 60)


def test_required_replicas_frames_partition_upscaler() -> None:
    """Test required_replicas with frames partition for upscaler."""
    result = required_replicas(
        name="upscaler",
        video_seconds=600,
        ttff=8.0,
        per_sec=0.2,
        partition="frames",
        req_per_min=0.5,
    )
    # ttff_total = 8.0, per_sec_total = 600 * 0.2 = 120
    # total_time_per_request = 128, total_time_per_minute = 128 * 0.5 = 64
    # expected = 64 / 60
    assert_equals_approx(result, 64 / 60)


def test_required_replicas_subscenes_partition() -> None:
    """Test required_replicas with subscenes partition."""
    result = required_replicas(
        name="test",
        video_seconds=300,
        ttff=15.0,
        per_sec=0.3,
        partition="subscenes",
        req_per_min=3.0,
    )
    # ttff_total = 15.0, per_sec_total = 300 * 0.3 = 90
    # total_time_per_request = 105, total_time_per_minute = 105 * 3.0 = 315
    # expected = 315 / 60
    assert_equals_approx(result, 315 / 60)


def test_required_replicas_other_partition() -> None:
    """Test required_replicas with other/default partition."""
    result = required_replicas(
        name="test",
        video_seconds=120,
        ttff=2.0,
        per_sec=0.1,
        partition="unknown",
        req_per_min=5.0,
    )
    # ttff_total = 2.0, per_sec_total = 120 * 0.1 = 12
    # total_time_per_request = 14, total_time_per_minute = 14 * 5.0 = 70
    # expected = 70 / 60
    assert_equals_approx(result, 70 / 60)


def test_get_replicas_with_custom_parameters() -> None:
    """Test get_replicas with custom video length and QPM list."""
    video_seconds = 300
    custom_qpm = [1, 5, 10]

    replicas = get_replicas(
        video_seconds=video_seconds,
        requests_per_minute=0.5,
        time_per_req=TIME_PER_REQ,
        init_replicas=INIT_REPLICAS,
        qpms=custom_qpm,
    )

    # Check structure
    assert GPUType.A100 in replicas
    assert Model.FLUX in replicas[GPUType.A100]

    flux_replicas = replicas[GPUType.A100][Model.FLUX]

    # Should have one entry per QPM
    assert len(flux_replicas) == len(custom_qpm)

    # Replicas should increase with higher QPM
    assert flux_replicas[0] <= flux_replicas[1] <= flux_replicas[2]


def test_get_replicas_returns_all_models() -> None:
    """Test that get_replicas()."""
    replicas = get_replicas(video_seconds=10 * 60)

    assert len(replicas) > 0
    for gpu_type in replicas.keys():
        assert gpu_type in replicas
        assert len(replicas[gpu_type]) > 0
        for model in replicas[gpu_type].keys():
            assert model in replicas[gpu_type]
            replica_list = replicas[gpu_type][model]
            for replica_count in replica_list:
                assert replica_count > 0
                # Each successive QPM should have >= replicas than previous
                for i in range(len(replica_list) - 1):
                    assert replica_list[i] <= replica_list[i + 1]


def test_get_costs_structure() -> None:
    """Test that get_costs returns correct structure."""
    replicas = get_replicas(video_seconds=600)
    costs = get_costs(replicas=replicas, gpu_costs=GPU_SPOT_COST)

    # Should match replicas structure
    assert set(costs.keys()) == set(replicas.keys())

    for gpu_type in costs:
        assert set(costs[gpu_type].keys()) == set(replicas[gpu_type].keys())

        for model in costs[gpu_type]:
            # Cost lists should have same length as replica lists
            assert len(costs[gpu_type][model]) == len(replicas[gpu_type][model])


def test_costs() -> None:
    """Test that get_costs correctly calculates costs."""
    replicas = get_replicas(video_seconds=600)
    costs = get_costs(
        replicas=replicas,
        gpu_costs=GPU_SPOT_COST)

    # Verify cost calculations
    for gpu_type in replicas:
        for model in replicas[gpu_type]:
            for i, replica_count in enumerate(replicas[gpu_type][model]):
                expected_cost = replica_count * GPU_SPOT_COST[gpu_type]
                assert abs(costs[gpu_type][model][i] - expected_cost) < 0.01

    for gpu_type in costs:
        for model in costs[gpu_type]:
            for cost in costs[gpu_type][model]:
                assert cost > 0

    # Total costs
    total_costs = get_total_costs(costs)
    assert len(total_costs) == len(QPM_LIST)

    for i in range(len(total_costs)):
        expected_total = 0
        for gpu_type in costs:
            for model in costs[gpu_type]:
                expected_total += costs[gpu_type][model][i]
        assert_equals_approx(total_costs[i], expected_total)

    # Costs should generally increase with QPM
    for i in range(len(total_costs) - 1):
        assert total_costs[i] <= total_costs[i + 1]


def test_get_total_costs_custom() -> None:
    """Test get_total_costs with custom QPM list."""
    custom_qpm = [1, 2, 5]
    replicas = get_replicas(video_seconds=600, qpms=custom_qpm)
    costs = get_costs(replicas=replicas, gpu_costs=GPU_SPOT_COST)
    total_costs = get_total_costs(costs, qpms=custom_qpm)

    assert len(total_costs) == len(custom_qpm)


def test_get_replicas_short_video() -> None:
    """Test get_replicas with very short video."""
    replicas = get_replicas(video_seconds=10)  # 10 seconds

    assert len(replicas) > 0

    for gpu_type in replicas:
        for model in replicas[gpu_type]:
            assert len(replicas[gpu_type][model]) > 0


def test_get_replicas_long_video() -> None:
    """Test get_replicas() with very long video."""
    replicas = get_replicas(video_seconds=60 * 60)  # 1 hour

    assert len(replicas) > 0

    # Replicas for long video should be higher than short video
    short_replicas = get_replicas(video_seconds=60)

    for gpu_type in replicas:
        if gpu_type in short_replicas:
            for model in replicas[gpu_type]:
                if model in short_replicas[gpu_type]:
                    # At least for QPM=1, long video should need more replicas
                    assert replicas[gpu_type][model][0] >= short_replicas[gpu_type][model][0]


def test_aggregate_time_per_request_by_quality() -> None:
    """Test aggregate_time_per_request_by_quality()."""
    time_req_adaptive_agg = aggregate_time_per_request_by_quality(
        time_per_req=TIME_PER_REQ_ADAPTIVE,
        quality_portions=QUALITY_PORTIONS,
    )
    assert GPUType.A100 in time_req_adaptive_agg
    assert GPUType.H100 in time_req_adaptive_agg

    assert Model.GEMMA in time_req_adaptive_agg[GPUType.A100]
    assert Model.FLUX in time_req_adaptive_agg[GPUType.A100]
    assert Model.UPSCALER in time_req_adaptive_agg[GPUType.A100]
    assert Model.UPSCALER in time_req_adaptive_agg[GPUType.H100]

    assert_equals_approx(time_req_adaptive_agg[GPUType.A100][Model.OTHERS], 25.80)
    assert_equals_approx(time_req_adaptive_agg[GPUType.H100][Model.UPSCALER], 48.12)


def test_get_time_per_request_baseline() -> None:
    latency_data = load_latency_data("simulator/data/")

    time_per_req_baseline = get_time_per_request_baseline(
        workflow_config=PODCAST_WORKFLOW,
        latency_data=latency_data,
        init_replicas=INIT_REPLICAS_BASELINE,
    )
    assert GPUType.A100 in time_per_req_baseline
    assert Model.GEMMA in time_per_req_baseline[GPUType.A100]
    assert_equals_approx(time_per_req_baseline[GPUType.A100][Model.FLUX], 9.75)
    assert_equals_approx(time_per_req_baseline[GPUType.A100][Model.FT], 24618.20)
