# DETAILED TEST COVERAGE SUMMARY - Functions Tested

## ROOT-LEVEL UTILITIES TEST COVERAGE

### test_console_utils.py ✓
**Functions tested:**
- `bytes_to_human()` - Tested with: 0, 512, 1024, 1536, 1.5GB, 1.7TB, 1.6PB, 1.92EB
- `setup_logging()` - Tested with DEBUG and INFO levels

### test_file_utils.py ✓
**Functions tested:**
- `binary_to_base64()` - Binary to base64 conversion
- `base64_to_binary()` - Base64 back to binary
- Error handling: TypeError for invalid inputs

### test_image_utils.py ✓
**Functions tested:**
- `img_to_base64()` - Image to base64 string
- `base64_to_img()` - Base64 string to Image
- `img_to_bytesio()` - Image to BytesIO object
- `get_image_file_info()` - Extracted: width, height, aspect_ratio
- Error handling: TypeError, FileNotFoundError, invalid base64

### test_audio_utils.py ✓
**Functions tested (from media_utils.py):**
- `empty_audio_file()` - Create empty audio, verify duration/samplerate/channels
- `get_audio_file_info()` - Extract audio metadata
- `get_audio_duration()` - Get duration in seconds
- `chunk_audio_base64()` - Chunk audio into base64 chunks
- `base64_to_audio_file()` - Convert base64 to audio file
- `fit_audio_to_duration()` - Fit audio to target duration
- `get_aligned_duration()` - Get aligned duration
- Error handling: FileNotFoundError, TypeError

### test_tts_utils.py ✓
**Functions tested:**
- `get_audio_chunks_by_silences()` - Detect and split audio at silences
- `split_into_sentences_max_duration()` - Split text sentences by max duration
- `strip_audio_file_silence()` - Remove silence from audio
- `generate_waveform_plt()` - Generate waveform visualization
- `merge_chunks()` - Merge audio chunks back together

### test_video_utils.py ✓
**Functions tested (from media_utils.py):**
- `get_frame_with_text()` - Create frame with text
- `video_frames_to_base64()` - Frames to base64
- `base64_to_video_frames()` - Base64 to frames
- `chunk_video_binary()` - Chunk video binary
- `save_video_frames()` - Save frames as video
- `save_video_audio()` - Save audio track
- `add_text_to_frame()` - Add text overlay
- `concatenate_videos()` - Merge multiple videos
- `change_video_fps()` - Resample video fps
- `get_video_file_info()` - Extract video metadata
- `get_video_duration()` - Get duration
- `get_video_frames()` - Extract all frames
- `get_video_frame()` - Extract specific frame
- `get_video_fps()` - Get frame rate
- `get_video_num_frames()` - Get frame count
- `get_video_with_text()` - Video with text overlay
- `get_video_frames_at_fps()` - Extract frames at specific fps
- `get_video_size()` - Get dimensions
- `get_font_size()` - Calculate font size
- `get_ffmpeg_version()` - Get ffmpeg version

### test_k8s_utils.py ✓
**Functions tested:** (Kubernetes utilities)
- Cluster connection
- Pod/node/service listing and querying
- Load balancer operations

### test_quart_utils.py ✓
**Functions tested:** (Quart/Flask utilities)
- Route handling
- JSON formatting
- Template rendering

### test_pdf_utils.py ✓
**Functions tested:** (PDF utilities)
- PDF reading/extraction
- Text extraction
- Page operations

### test_tensor_utils.py ✓
**Functions tested:** (Tensor utilities)
- Tensor operations
- Shape manipulation
- Device handling

### test_utils.py ✓
**Functions tested:** (General utilities)
- Helper functions
- Utility functions

---

## SIMULATOR MODULE TEST COVERAGE

### test_simulator_types.py ✓
**Classes/Functions tested:**
- `Model` - Enum: GEMMA, FLUX, HF, HF_VAE, FT, FT_VAE, UPSCALER, OTHERS
- `GPUType` - Enum: A100, H100, H200, GB200
- `QualityLevel` - Enum: ORIGINAL, HIGH, MEDIUM, LOW
- `ModelAllocation` - Factory pattern, JSON serialization
- Model list to JSON conversion
- Workflow to JSON conversion
- Policy to JSON conversion

**Test cases:** 
- Serialize models dict to JSON with nested structure
- Verify GPU type comparison
- Quality level handling

### test_workflows.py ✓
**Functions/Classes tested:**
- `build_workflow_config()` - Build custom workflow configs
- `_get_num_subscenes()` - Calculate scene count
- `_get_num_scenes()` - Calculate total scenes
- `_video_gen_work()` - Calculate video generation work
- Pre-built workflows:
  - `PODCAST_WORKFLOW`
  - `SHORTS_WORKFLOW`
  - `MOVIE_WORKFLOW`
  - `ANIMATED_STORY_WORKFLOW`
  - `LECTURE_WORKFLOW`
  - `DUBBING_WORKFLOW`
  - `EDITING_WORKFLOW`
  - `VIDEO_CHAT_WORKFLOW`

**Constants tested:**
- `MAX_FT_FRAMES`
- `SUBSCENE_SECONDS`
- `SUBSCENES_PER_SCENE`
- `FPS`, `NUM_STEPS`, `FRAMES_OPTIONS`

### test_simulator_policies.py ✓
**Policies tested:**
- `STREAMWISE_POLICY` - Verify name, gpu_cost, objective (TTFF_COST)
- `BASELINE_POLICIES` - 6 baselines: naive, naive disag, naive ttff*cost, naive upscaler, naive spot, naive hardware

### test_simulator_actions.py ✓
**Classes/Functions tested:**
- `Action` class - ActionName, model, gpu_type, models config, result, arrival_time
- Error handling: TypeError for missing args, ValueError for invalid enum values

**Test cases:**
- 2 test_action() functions with comprehensive validation
- 1 test_action_errors() function testing invalid inputs

### test_evaluator.py ✓
**Functions tested:**
- `evaluate_model_allocation()` - Evaluate allocations with latency/power data
- Error handling: AssertionError for missing GEMMA model

**Models validated:**
- GemmaModelAllocation
- FluxModelAllocation
- HFModelAllocation
- HFVAEModelAllocation
- FTModelAllocation
- UpscalerModelAllocation
- OthersModelAllocation

### test_data_loading.py ✓
**Functions tested:**
- `load_latency_data()` - Load model latency profiles
- `load_power_data()` - Load power consumption data
- `load_adaptive_quality_data()` - Load quality level data (LOW, MEDIUM)

**Test cases:**
- Successful data loading from simulator/data/
- FileNotFoundError for missing data
- Quality level variations

### test_simulator_provisioning.py ✓
**Classes/Functions tested:**
- `ProvisioningResult` - Results computation
- Provisioning calculations with GPU configs

### test_simulator_utils.py ✓
**Functions tested:**
- `get_pareto_frontier()` - Compute pareto efficient points
- `find_most_cost_effective_provisioning()` - Find cheapest provisioning
- `find_most_energy_efficient_provisioning()` - Find most efficient
- `find_pareto_frontier()` - Pareto frontier computation
- `coalesce_models()` - Model aggregation

**Test cases:**
- Pareto frontier with 1, 3, 4, 5 points
- Pareto with max_x and max_y constraints
- Finding most cost/energy effective from ProvisioningResult

### test_simulator_multirequests.py ✓
**Functions tested:**
- Multi-request handling
- Request queuing and processing

### test_simulator_baseline.py ✓
**Functions tested:**
- Baseline allocation strategies
- Comparison with other allocators

### test_simulator_energy.py ✓
**Functions tested:**
- Energy calculation
- Energy optimization
- Cost-energy tradeoff

### test_simulator_plotutils.py ✓
**Functions tested:**
- Visualization functions
- Plot generation
- Data formatting for plotting

### test_greedy.py ✓
**Allocator tested: GreedyAllocator**

**Test configurations:**
- 8 A100 + 8 H100 GPUs: 50-60 min TTFF, 45-55 min total time
- 32 A100 + 32 H100 GPUs: Linear scaling
- 16 A100 only: Single GPU type
- 64 A100 only: High A100 count
- 64 H100 only: High H100 count

**Metrics validated:**
- TTFF (Time To First Frame)
- TBF (Time Between Frames) 
- total_time_s
- tbf_s
- GPU utilization

### test_helix.py ✓
**Allocator tested: HelixAllocator**
- Helix-specific allocation strategy

### test_hexgen.py ✓
**Allocator tested: HexGenAllocator**
- HexGen allocation algorithm

### test_milp.py ✓
**Allocator tested: MILPAllocator**
- MILP (Mixed Integer Linear Programming) solver
- Optimization under constraints

### test_simulator.py ✓
**Integration tests:**
- End-to-end simulator workflow
- 8 A100 + 8 H100 configuration
- Result validation: TTFF < Total Time
- Performance assertions

---

## STREAMWISE MODULE TEST COVERAGE

### test_streamwise.py ✓
**HTTP routes tested:**
- GET / - Index page
- Service endpoints:
  - `get_services()` - List services
  - `get_service_timestamps()` - Service timestamps
  - `get_service_health()` - Health status
  - `get_service_files()` - Service file listing
  - `get_k8s_pod_events()` - Pod events
  - `get_k8s_container_logs()` - Container logs
  - `parse_vllm_metrics()` - LLM metrics parsing
  - `get_health_and_files_async()` - Async health/files

**Test cases:**
- HTTP status code validation
- JSON response parsing
- K8s mock integration
- Service account manager integration

### test_streamwise_files.py ✓
**File operations tested:**
- Video file info endpoint
- Image file info endpoint
- File operations
- `get_file_info()` function
- HTTP status codes (200, 500, etc.)

### test_streamwise_pod.py ✓
**Pod operations tested:**
- Pod listing
- Pod metrics
- Pod events
- Pod status queries

### test_streamwise_node.py ✓
**Node operations tested:**
- Node listing
- Node status
- Node resource queries

### test_streamwise_job.py ✓
**Job operations tested:**
- Job creation/listing
- Job status monitoring
- Job completion tracking

---

## STREAMWISE_APP TEST COVERAGE (15 apps)

| App | Test File | Test Coverage |
|-----|-----------|----------------|
| StreamChat | test_streamchat.py | App initialization, HTTP routes, endpoints |
| StreamCast | test_streamcast.py | Podcast casting functionality |
| StreamAnimate | test_streamanimate.py | Animation generation |
| StreamEdit | test_streamedit.py | Video editing operations |
| StreamLecture | test_streamlecture.py | Lecture video generation |
| StreamPersona | test_streampersona.py | Avatar/persona handling |
| StreamDub | test_streamdub.py | Audio dubbing |
| StreamMovie | test_streammovie.py | Movie generation |
| StreamShort | test_streamshort.py | Short video generation |
| SceneSegment | test_scene.py | Scene/segment operations, timestamps, images, descriptions |
| Character | test_characters.py | Character management |
| Client | test_client.py | API client |
| ServiceManager | test_service_manager.py | Service management |
| LMMGenerator | test_lmm_generator.py | LMM integration |
| VideoChunking | test_gen_video_chunked.py | Chunked video generation |

---

## CRITICAL GAPS - UNTESTED MODULES

### simulator/models.py (710 lines) - CRITICAL
**NOT tested directly. Indirect testing through:**
- test_evaluator.py (uses model allocations)
- test_greedy.py (uses model allocations)
- test_helix.py (uses model allocations)
- test_hexgen.py (uses model allocations)
- test_milp.py (uses model allocations)

**Functions NOT directly tested:**
- `get_model_allocation()` factory function
- All 7 ModelAllocation subclasses:
  - `GemmaModelAllocation.calculate_time()`
  - `FluxModelAllocation.calculate_time()`
  - `HFModelAllocation.calculate_time()`
  - `HFVAEModelAllocation.calculate_time()`
  - `FTModelAllocation.calculate_time()`
  - `UpscalerModelAllocation.calculate_time()`
  - `OthersModelAllocation.calculate_time()`
- `_calculate_total_time()` helper
- Energy calculations per model
- Cost calculations per model

### streamwise/service_manager.py (418 lines) - CRITICAL
**Functions NOT tested:**
- `parse_vllm_metrics()` - Parse vLLM inference metrics
- `get_k8s_pod_events()` - Get pod events from K8s
- `get_k8s_container_logs()` - Get container logs
- `get_services()` - List services (untested directly)
- `get_services_ns()` - List services in namespace
- `get_service_timestamps()` - Service timestamps (untested directly)
- `get_service_health()` - Service health check (untested directly)
- `get_service_files()` - Service file info (untested directly)
- `get_health_and_files_async()` - Async health and files

### simulator/auto_model_allocator.py (107 lines) - HIGH
**Functions NOT tested:**
- All functions in auto_model_allocator
- Depends on untested models.py

### simulator/model_allocator.py (~300 lines) - HIGH
**Functions NOT tested:**
- Base allocator class
- Abstract methods and implementations

### simulator/naive_baseline.py (~200 lines) - MEDIUM
**Functions NOT tested:**
- Naive baseline allocation strategies
- Comparison baselines

### streamwise/service_account_manager.py (~150 lines) - HIGH
**Functions NOT tested:**
- K8s service account operations
- Account creation/deletion
- RBAC configuration

---

## SUMMARY BY PRIORITY

### NEED TO CREATE IMMEDIATELY
1. `tests/simulator/test_models.py` (200-300 lines)
2. `tests/streamwise/test_service_manager.py` (200-250 lines)
3. Extend `tests/test_media_utils.py` (300-400 lines)

### SHOULD ADD NEXT
4. `tests/simulator/test_auto_model_allocator.py` (100 lines)
5. `tests/simulator/test_naive_baseline.py` (100-150 lines)
6. `tests/streamwise/test_service_account_manager.py` (100-150 lines)

### COULD ADD LATER
7. `tests/test_ppt_utils.py` (80-100 lines)
8. `tests/simulator/test_model_allocator.py` (150-200 lines)

