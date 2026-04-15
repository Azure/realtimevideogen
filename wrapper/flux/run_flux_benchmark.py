# Flux inference with USP


import time
import torch

from diffusers import FluxPipeline

from flux_xfuser import parallelize_transformer

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)


def main() -> None:
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    GPU = torch.cuda.get_device_name(0)

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    # [-h] [--model MODEL] [--download-dir DOWNLOAD_DIR] [--trust-remote-code] [--warmup_steps WARMUP_STEPS]
    # [--use_parallel_vae]
    # [--use_torch_compile] [--use_onediff]
    # [--use_teacache] [--use_fbcache] [--use_ray] [--ray_world_size RAY_WORLD_SIZE]
    # [--dit_parallel_size DIT_PARALLEL_SIZE]
    # [--use_cfg_parallel]
    # [--data_parallel_degree DATA_PARALLEL_DEGREE] [--ulysses_degree ULYSSES_DEGREE] [--ring_degree RING_DEGREE]
    # [--pipefusion_parallel_degree PIPEFUSION_PARALLEL_DEGREE
    # [--num_pipeline_patch NUM_PIPELINE_PATCH] [--attn_layer_num_for_pp [ATTN_LAYER_NUM_FOR_PP ...]]
    # [--tensor_parallel_degree TENSOR_PARALLEL_DEGREE]
    # [--vae_parallel_size VAE_PARALLEL_SIZE] [--split_scheme SPLIT_SCHEME] [--height HEIGHT] [--width WIDTH]
    # [--num_frames NUM_FRAMES] [--img_file_path IMG_FILE_PATH]
    # [--prompt [PROMPT ...]] [--no_use_resolution_binning] [--negative_prompt [NEGATIVE_PROMPT ...]]
    # [--num_inference_steps NUM_INFERENCE_STEPS]
    # [--max_sequence_length MAX_SEQUENCE_LENGTH] [--seed SEED] [--output_type OUTPUT_TYPE]
    # [--enable_sequential_cpu_offload] [--enable_model_cpu_offload] [--enable_tiling]
    # [--enable_slicing] [--use_fp8_t5_encoder] [--use_fast_attn] [--n_calib N_CALIB] [--threshold THRESHOLD]
    # [--window_size WINDOW_SIZE] [--coco_path COCO_PATH]
    # [--use_cache]

    cache_args = {
        "use_teacache": engine_args.use_teacache,
        "use_fbcache": engine_args.use_fbcache,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": input_config.num_inference_steps,
    }

    # pipe = xFuserFluxPipeline.from_pretrained(
    pipe = FluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    initialize_runtime_state(pipe, engine_config)
    get_runtime_state().set_input_parameters(
        height=input_config.height,
        width=input_config.width,
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        max_condition_sequence_length=512,
        split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
    )

    parallelize_transformer(pipe)

    if engine_config.runtime_config.use_torch_compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

        # one step to warmup the torch compiler
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=1,
            output_type=input_config.output_type,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        ).images

    # Run to warm up the model
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt="warmup prompt",
        num_inference_steps=1,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )

    # Actual run
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    input_config.prompt = "a photo of an astronaut riding a horse on mars"
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    torch.cuda.synchronize()  # Ensure all above CUDA ops are done
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    # Write into a file
    if get_world_group().rank == 0:
        with open("flux_parallel_result.csv", "a") as f:
            dp = engine_args.data_parallel_degree
            up = engine_args.ulysses_degree
            rp = engine_args.ring_degree
            tp = engine_args.tensor_parallel_degree
            cfg = engine_config.parallel_config.cfg_degree
            torchcompile = engine_config.runtime_config.use_torch_compile
            teacache = engine_args.use_teacache
            fbcache = engine_args.use_fbcache
            steps = input_config.num_inference_steps
            height = input_config.height
            width = input_config.width
            f.write(f"{GPU},{get_world_group().world_size},{dp},{up},{rp},{tp},{cfg},"
                    f"{torchcompile},{teacache},{fbcache},{steps},{height},{width},{elapsed_time:.2f}\n")

    if get_world_group().rank == 0:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory / 1e9:.2f} GB, "
            f"memory: {peak_memory / 1e9:.2f} GB"
        )
        print(f"VAE: {pipe.vae.elapsed_time:.2f} sec")
        print(f"Transformer: {pipe.transformer.get_elapsed_time():.2f} sec")
        print(f"Scheduler: {pipe.scheduler.elapsed_time:.2f} sec")
    # get_runtime_state().destroy_distributed_env()
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
