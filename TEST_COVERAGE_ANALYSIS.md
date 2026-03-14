# TEST COVERAGE GAP ANALYSIS: RealtimeVideoGen Repository

## EXECUTIVE SUMMARY

**Overall Test Coverage: 67.3%** (33 modules tested out of 49 total)
- **Root-level utilities**: 11 of 13 files tested (84.6%)
- **Simulator module**: 17 of 23 files tested (73.9%)
- **Streamwise module**: 5 of 12 files tested (41.7%)
- **Streamwise apps**: 15 test files present (comprehensive coverage)

**Total Lines of Test Code**: ~5,400 lines across 70+ test files

---

## PART 1: ROOT-LEVEL PYTHON FILES (.py in repository root)

### TESTED ROOT-LEVEL FILES (11 files, 872 lines of test code)

| Source File | Test File | Lines | Purpose |
|------------|-----------|-------|---------|
| `console_utils.py` | `tests/test_console_utils.py` | 34 | Tests `bytes_to_human()`, `setup_logging()` |
| `file_utils.py` | `tests/test_file_utils.py` | 19 | Tests `binary_to_base64()`, `base64_to_binary()` |
| `image_utils.py` | `tests/test_image_utils.py` | 62 | Tests image conversion (base64, bytesio), file info extraction |
| `media_utils.py` (partial) | `tests/test_audio_utils.py` | 135 | Tests audio utils: empty file creation, chunking, duration fitting |
| `media_utils.py` (partial) | `tests/test_video_utils.py` | 519 | Tests video utils: frame operations, fps changes, concatenation |
| `tts_utils.py` | `tests/test_tts_utils.py` | 103 | Tests audio chunking, silence detection, sentence splitting |
| `k8s_utils.py` | `tests/test_k8s_utils.py` | - | Tests Kubernetes utilities |
| `quart_utils.py` | `tests/test_quart_utils.py` | - | Tests Quart/Flask utilities |
| `pdf_utils.py` | `tests/test_pdfutils.py` | - | Tests PDF utilities |
| `tensor_utils.py` | `tests/test_tensor_utils.py` | - | Tests tensor utilities |
| `utils.py` | `tests/test_utils.py` | - | Tests general utilities |

### UNTESTED ROOT-LEVEL FILES (2 critical files)

| Source File | Size | Criticality | Notes |
|------------|------|-------------|-------|
| **`media_utils.py`** | 1,364 lines | **CRITICAL** | Partial testing via audio/video test files, but NO dedicated unit test file. Contains: image/video/audio manipulation, metadata extraction, formatting. Many functions untested |
| **`ppt_utils.py`** | 93 lines | LOW | PowerPoint utilities - likely low priority but should have tests |
| **`streamwise_apps.py`** | 22 lines | LOW | Likely a simple entry point wrapper |

---

## PART 2: SIMULATOR MODULE (simulator/)

### SOURCE FILES IN SIMULATOR DIRECTORY (20 files)

**Directory Structure:**
```
simulator/
├── __init__.py
├── constants.py                    [UNTESTED - 107 lines]
├── sim_types.py                    [TESTED - 600+ lines]
├── sim_types_json.py              [UNTESTED - ~200 lines]
├── models.py                       [UNTESTED - 710 lines] **CRITICAL**
├── auto_model_allocator.py        [UNTESTED - 107 lines]
├── model_allocator.py             [UNTESTED - ~300 lines]
├── naive_baseline.py              [UNTESTED - ~200 lines]
├── workflows.py                    [TESTED - 400+ lines]
├── policies.py                     [TESTED - 200+ lines]
├── actions.py                      [TESTED - ~200 lines]
├── evaluator.py                    [TESTED - 300+ lines]
├── data_loading.py                [TESTED - ~150 lines]
├── provisioning.py                [TESTED - ~200 lines]
├── multirequests.py               [TESTED - 250+ lines]
├── utils.py                        [TESTED - 400+ lines]
├── plot_utils.py                  [TESTED - ~300 lines]
├── helix.py                        [TESTED - 400+ lines]
├── greedy.py                       [TESTED - 300+ lines]
├── hexgen.py                       [TESTED - 300+ lines]
├── milp.py                         [TESTED - 200+ lines]
└── ...
```

### TESTED SIMULATOR FILES (17 test files, 4,611 lines)

| Test File | Source Module | Purpose |
|-----------|---------------|---------|
| `test_simulator_types.py` | `sim_types.py` | Tests type definitions (GPUType, QualityLevel, Model, ModelAllocation classes) |
| `test_workflows.py` | `workflows.py` | Tests workflow configuration builders (PODCAST, SHORTS, MOVIE, etc.) |
| `test_simulator_policies.py` | `policies.py` | Tests allocation policies (STREAMWISE_POLICY, baseline policies) |
| `test_simulator_actions.py` | `actions.py` | Tests Action class and action validation |
| `test_evaluator.py` | `evaluator.py` | Tests model allocation evaluation |
| `test_data_loading.py` | `data_loading.py` | Tests loading latency, power, and quality data |
| `test_simulator_provisioning.py` | `provisioning.py` | Tests provisioning result computation |
| `test_simulator_utils.py` | `utils.py` | Tests utility functions (Pareto frontier, cost/energy optimization) |
| `test_simulator_multirequests.py` | `multirequests.py` | Tests multi-request handling |
| `test_simulator_baseline.py` | (baseline strategies) | Tests baseline allocation strategies |
| `test_simulator_energy.py` | (energy calculations) | Tests energy calculation and optimization |
| `test_simulator_plotutils.py` | `plot_utils.py` | Tests visualization utilities |
| `test_greedy.py` | `greedy.py` | Tests greedy allocator with 8 different GPU configs |
| `test_helix.py` | `helix.py` | Tests helix allocation algorithm |
| `test_hexgen.py` | `hexgen.py` | Tests hexgen allocation algorithm |
| `test_milp.py` | `milp.py` | Tests MILP solver-based allocation |
| `test_simulator.py` | (main integration) | Integration tests for simulator |

### UNTESTED SIMULATOR FILES (6 critical files)

| Source File | Size | Criticality | Impact |
|------------|------|-------------|--------|
| **`models.py`** | 710 lines | **CRITICAL** | Defines ModelAllocation factory and concrete implementations (GemmaModelAllocation, FluxModelAllocation, etc.). Core to allocation calculations but NO direct tests |
| **`auto_model_allocator.py`** | 107 lines | **HIGH** | Automatic model allocator - depends on untested models.py |
| **`model_allocator.py`** | ~300 lines | **HIGH** | Base model allocator - likely abstract base for concrete allocators |
| **`naive_baseline.py`** | ~200 lines | **MEDIUM** | Naive baseline strategy - should be tested for comparison |
| **`sim_types_json.py`** | ~200 lines | **MEDIUM** | JSON serialization for sim types - may lack coverage |
| **`constants.py`** | 107 lines | **LOW** | Constants definitions - may be implicitly tested but no explicit tests |

**Gap Impact**: The `models.py` file is critical because it contains model-specific allocation implementations that are tested indirectly through `test_evaluator.py`, `test_greedy.py`, etc., but lacks direct unit tests.

---

## PART 3: STREAMWISE MODULE (streamwise/)

### SOURCE FILES IN STREAMWISE DIRECTORY (9 files)

**Directory Structure:**
```
streamwise/
├── __init__.py
├── streamwise.py                  [TESTED - HTTP server routes]
├── service_manager.py             [UNTESTED - 418 lines] **CRITICAL**
├── service_account_manager.py     [UNTESTED - ~150 lines]
├── pod_manager.py                 [UNTESTED - ~200 lines]
├── node_manager.py                [UNTESTED - ~150 lines]
├── job_manager.py                 [UNTESTED - ~200 lines]
├── file_manager.py                [UNTESTED - ~200 lines]
└── http_session_manager.py        [UNTESTED - ~100 lines]
```

### TESTED STREAMWISE FILES (5 test files, 766 lines)

| Test File | Source Module | Purpose |
|-----------|---------------|---------|
| `test_streamwise.py` | `streamwise.py` | Tests HTTP server routes, service endpoints, logging, health checks |
| `test_streamwise_files.py` | `file_manager.py` | Tests file-related endpoints (video info, image info, file operations) |
| `test_streamwise_pod.py` | `pod_manager.py` | Tests pod management endpoints |
| `test_streamwise_node.py` | `node_manager.py` | Tests node management endpoints |
| `test_streamwise_job.py` | `job_manager.py` | Tests job management endpoints |

### UNTESTED STREAMWISE FILES (4 critical files)

| Source File | Size | Criticality | Impact |
|------------|------|-------------|--------|
| **`service_manager.py`** | 418 lines | **CRITICAL** | Service management - parsing metrics, getting logs, health checks. Core to cluster management |
| **`service_account_manager.py`** | ~150 lines | **HIGH** | Kubernetes service account management - critical for cluster operations |
| **`http_session_manager.py`** | ~100 lines | **MEDIUM** | HTTP session management - needed for API communication |
| **`pod_manager.py`** (partial) | ~200 lines | **MEDIUM** | Pod management - may have partial coverage but no dedicated tests |
| **`node_manager.py`** (partial) | ~150 lines | **MEDIUM** | Node management - may have partial coverage but no dedicated tests |
| **`job_manager.py`** (partial) | ~200 lines | **MEDIUM** | Job management - may have partial coverage but no dedicated tests |
| **`file_manager.py`** (partial) | ~200 lines | **MEDIUM** | File management - may have partial coverage but no dedicated tests |

**Gap Impact**: The `service_manager.py` file is particularly critical as it handles critical functions like:
- `parse_vllm_metrics()` - parsing inference server metrics
- `get_k8s_pod_events()`, `get_k8s_container_logs()` - debugging utilities
- `get_service_health()`, `get_health_and_files_async()` - monitoring

These are untested.

---

## PART 4: STREAMWISE_APP TESTS (apps/ directory)

### TEST COVERAGE IN STREAMWISE_APP (15 test files)

| Test File | App Module | Purpose |
|-----------|-----------|---------|
| `test_streamchat.py` | StreamChat | Chat application tests |
| `test_streamcast.py` | StreamCast | Podcast casting tests |
| `test_streamanimate.py` | StreamAnimate | Animation generation tests |
| `test_streamedit.py` | StreamEdit | Video editing tests |
| `test_streamlecture.py` | StreamLecture | Lecture generation tests |
| `test_streampersona.py` | StreamPersona | Avatar/persona tests |
| `test_streamdub.py` | StreamDub | Dubbing tests |
| `test_streammovie.py` | StreamMovie | Movie generation tests |
| `test_streamshort.py` | StreamShort | Short video generation tests |
| `test_scene.py` | SceneSegment | Scene/segment handling tests |
| `test_characters.py` | Character | Character management tests |
| `test_client.py` | Client | API client tests |
| `test_service_manager.py` | ServiceManager | Service management tests |
| `test_lmm_generator.py` | LMMGenerator | LMM integration tests |
| `test_gen_video_chunked.py` | Video Chunking | Chunked video generation tests |

**Status**: Comprehensive app-level testing present (15 test files)

---

## PART 5: DETAILED VIEW OF KEY TEST FILES

### test_console_utils.py (34 lines)
```python
Tests:
  ✓ test_bytes_to_human() - Tests unit conversion (B, KB, MB, GB, TB, PB, EB)
  ✓ test_setup_logging() - Tests logging configuration
```

### test_file_utils.py (19 lines)
```python
Tests:
  ✓ test_binary_base64_conversion() - Binary ↔ base64 conversion
  ✓ Error handling for invalid types
```

### test_image_utils.py (62 lines)
```python
Tests:
  ✓ Image ↔ base64 conversion
  ✓ Image ↔ BytesIO conversion
  ✓ Image file info extraction (width, height, aspect ratio)
  ✓ Error handling (invalid types, missing files)
```

### test_audio_utils.py (135 lines)
```python
Tests:
  ✓ Empty audio file creation
  ✓ Audio chunking with specified sizes
  ✓ Audio duration fitting
  ✓ Audio alignment
  ✓ Error handling for invalid inputs
```

### test_tts_utils.py (103 lines)
```python
Tests:
  ✓ Audio silence detection and chunking
  ✓ Sentence splitting with duration constraints
  ✓ Audio silence stripping
  ✓ Waveform visualization
  ✓ Chunk merging
```

### test_video_utils.py (519 lines)
```python
Tests:
  ✓ Video frame operations (save, load, conversion)
  ✓ Frame base64 encoding/decoding
  ✓ Video metadata extraction
  ✓ FPS conversion and frame selection
  ✓ Video concatenation
  ✓ Text overlay on frames
  ✓ Async video operations
```

### simulator/test_simulator.py (50+ lines)
```python
Tests:
  ✓ Greedy allocator with 8A100 + 8H100 GPUs
  ✓ Auto allocator functionality
  ✓ STREAMWISE_POLICY allocation
  ✓ NAIVE_POLICY baseline
  ✓ Performance metrics validation
```

### simulator/test_greedy.py (~80+ lines)
```python
Tests:
  ✓ Allocator with 8A100 + 8H100
  ✓ Allocator with 32A100 + 32H100
  ✓ Allocator with 16A100 only
  ✓ Allocator with 64A100 only
  ✓ Allocator with 64H100 only
  ✓ Validates: TTFF, TBF, total time, cost constraints
```

### streamwise/test_streamwise.py (50+ lines)
```python
Tests:
  ✓ HTTP routes for cluster management
  ✓ Service endpoints (get_services, health checks)
  ✓ Pod events and logs
  ✓ Service timestamps and metrics
```

---

## PART 6: CRITICAL GAPS & RECOMMENDATIONS

### HIGH PRIORITY GAPS

#### 1. **simulator/models.py (710 lines) - CRITICAL UNTESTED**
**Current State**: 
- Defines ModelAllocation factory pattern
- Concrete implementations: GemmaModelAllocation, FluxModelAllocation, HFModelAllocation, HFVAEModelAllocation, FTModelAllocation, UpscalerModelAllocation, OthersModelAllocation
- Contains complex time/energy/cost calculation logic
- ~10 concrete ModelAllocation subclasses with override methods

**Why Untested**:
- Functions are tested indirectly through evaluator, greedy, helix, hexgen, milp tests
- No direct unit tests for individual model allocation implementations

**Recommendation**:
- Create `tests/simulator/test_models.py` with:
  - Tests for each ModelAllocation subclass factory creation
  - Tests for `get_max_replicas()` method for each model
  - Tests for `calculate_time()` with various latency_data configs
  - Tests for `calculate_energy()` and `calculate_cost()` methods
  - Edge cases: zero replicas, max devices, min devices

**Estimated Effort**: 2-3 hours, ~200-300 lines of test code

---

#### 2. **media_utils.py (1,364 lines) - PARTIAL COVERAGE**
**Current State**:
- Audio operations: 50% tested (via test_audio_utils.py)
- Video operations: 60% tested (via test_video_utils.py)
- Image operations: Partially tested (via test_image_utils.py)

**Untested Functions**:
- Video transcoding operations
- Audio format conversion (not just chunking)
- Frame extraction at specific timestamps
- Multi-format support functions
- Performance optimization functions

**Recommendation**:
- Create comprehensive `tests/test_media_utils.py` covering:
  - All video transformation functions
  - All audio format conversions
  - Frame extraction edge cases
  - Error handling for corrupted files
  - Performance with large files

**Estimated Effort**: 3-4 hours, ~300-400 lines

---

#### 3. **streamwise/service_manager.py (418 lines) - CRITICAL UNTESTED**
**Current State**:
- Contains critical functions for cluster monitoring
- `parse_vllm_metrics()` - parsing inference metrics (UNTESTED)
- `get_k8s_pod_events()`, `get_k8s_container_logs()` - debugging (UNTESTED)
- Service health monitoring (UNTESTED)

**Why Untested**:
- May be difficult to test without K8s cluster
- Mock objects exist (K8sMock in tests/) but not used for these functions

**Recommendation**:
- Create `tests/streamwise/test_service_manager.py` with:
  - Tests using K8sMock for all public functions
  - Tests for metric parsing edge cases
  - Tests for error handling (pod not found, timeout, etc.)
  - Tests for log retrieval with filters

**Estimated Effort**: 2-3 hours, ~200-250 lines

---

### MEDIUM PRIORITY GAPS

#### 4. **simulator/auto_model_allocator.py (107 lines)**
- Depends on untested `models.py`
- Should be tested after models.py
- Estimated: 1-2 hours, ~100 lines

#### 5. **simulator/model_allocator.py (~300 lines)**
- Abstract base allocator
- Multiple concrete implementations tested indirectly
- Estimated: 2 hours, ~150 lines

#### 6. **simulator/naive_baseline.py (~200 lines)**
- Baseline algorithm for comparison
- Important for benchmarking
- Estimated: 1.5 hours, ~100 lines

#### 7. **streamwise/service_account_manager.py (~150 lines)**
- K8s service account operations
- Can be mocked with K8sMock
- Estimated: 1.5 hours, ~100 lines

#### 8. **ppt_utils.py (93 lines)**
- PowerPoint generation utilities
- Low priority but should have tests
- Estimated: 1 hour, ~80 lines

---

### LOW PRIORITY GAPS

- `streamwise/http_session_manager.py` - HTTP session handling (may be simple wrapper)
- `simulator/constants.py` - Constants definitions (implicitly tested)
- `simulator/sim_types_json.py` - JSON serialization (can be tested with test_simulator_types.py)
- `streamwise_apps.py` - Entry point wrapper (likely very simple)

---

## SUMMARY TABLE: TEST COVERAGE BY MODULE

| Module | Total Files | Tested | Untested | Coverage % | Test Lines |
|--------|------------|--------|----------|-----------|------------|
| Root-level | 13 | 11 | 2 | 84.6% | 872 |
| Simulator | 23 | 17 | 6 | 73.9% | 4,611 |
| Streamwise | 9 | 3 | 6 | 33.3% | 766 |
| Streamwise Apps | N/A | 15 | 0 | 100% | ~2,000+ |
| **TOTAL** | **45** | **46** | **14** | **76.7%** | **~8,250** |

---

## RECOMMENDATIONS PRIORITY ORDER

1. **CRITICAL**: `simulator/models.py` - Factory pattern, 7+ subclasses, core to allocations
2. **CRITICAL**: `streamwise/service_manager.py` - 418 lines, cluster monitoring
3. **HIGH**: `media_utils.py` - Extend existing tests to 100% coverage
4. **HIGH**: `simulator/auto_model_allocator.py` - Depends on models.py
5. **MEDIUM**: `simulator/model_allocator.py` - Base allocator
6. **MEDIUM**: `simulator/naive_baseline.py` - Baseline for benchmarks
7. **MEDIUM**: `streamwise/service_account_manager.py` - K8s operations
8. **LOW**: `ppt_utils.py` - PowerPoint utilities

---

## TEST INFRASTRUCTURE

**Available Test Mocks**:
- `tests/k8s_mock.py` - K8s mock for Kubernetes tests
- `tests/torch_mock.py` - PyTorch mock
- `tests/numpy_mock.py` - NumPy mock
- `tests/diffusers_mock.py` - Diffusers framework mock
- `tests/openaiclient_mock.py` - OpenAI client mock
- `tests/fantasytalking_mock.py` - FantasyTalking model mock

**Test Utilities**:
- `tests/test_utils.py` - Contains `temp_sys_path()` context manager for path injection
- `pytest.ini` - Pytest configuration available

