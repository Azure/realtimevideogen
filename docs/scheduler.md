
# Request scheduler
The request scheduler orchestrates execution using a live, iterative version of our greedy algorithm informed by the request DAG.

## Multiple requests
Following the [YARN](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html) philosophy, each user request is managed by a _request scheduler_.
To coordinate multiple requests, model instances maintain local queues that prioritize tasks by deadline.
For example, the image generation model may process an early scene from a new request before a later scene from an earlier request if it has a tighter deadline.
The _scheduler_ monitors these queues and re-schedules as resources become available.

## DAG generation
Most of the DAG is generated at runtime.
For example, we do not know the number of scenes and shots until the screenplay is generated.
We start with a sketch of the DAG (e.g., 10-minute video with 30 seconds per shot) and as stages are generated, we update the DAG.

## Deadlines
The _request scheduler_ computes deadlines for each DAG node based on the SLO and expected runtimes.
For example, a real-time video podcast with a TTFF of 5 seconds and 10-minute duration sets the final node’s deadline at $t_{now}+605$, with dependent nodes scheduled recursively.

Execution starts with dependency-free nodes, prioritized by deadline.
As nodes finish, their dependents are triggered.
For example, screenplay generation precedes audio and image generation, with images prioritized due to tighter deadlines.
The _request scheduler_ also favors earlier scenes and shots.
Deadlines are attached to each submission for fine-grained, instance-level scheduling.

## Instance selection
Each DAG node is assigned to the model instance with the shortest expected runtime (e.g., 8xH200 vs 4xA100).
Lightweight tasks (e.g., prompt filtering, safety checks) are typically routed to cheaper GPUs or CPUs, while early-stage tasks may use higher-end instances.
If no instance is available, the _request scheduler_ queues the request and reschedules as resources free up.

## Adaptive quality
Scheduling begins with the target quality (e.g., 1280x800, 20 de-noising steps) and degrades incrementally (e.g., 640x400, 10 steps) if deadlines are at risk of not being met.
Initial and intermediate stages may run at lower quality, while final outputs are rendered at higher resolution.
If not enough, we switch to static content.

## Model constraints
Our greedy algorithm accounts for model-specific limits (e.g., maximum generation length).
For example, Fantasy Talking supports up to 3.5 seconds of video/audio at 23 FPS before drifting.
To handle longer shots, we first generate a full clip (e.g., 30 seconds) with FramePack at medium quality, segment it at speech pauses, and re-sync audio using Fantasy Talking.
Resolution constraints are also considered to ensure suitable aspect ratios.

## Evictions and failures
[Spot VMs](https://learn.microsoft.com/en-us/azure/architecture/guide/spot/spot-eviction) typically provide a 30-second eviction notice.
After receiving this notice, we stop sending new requests to instances on affected resources.
For other failures, we monitor the model instances liveness and avoid sending requests to unresponsive ones.
Requests running on failed resources are resubmitted.

## Caching
StreamWise supports reusing intermediate results across requests to improve efficiency.
Common assets (e.g., static backgrounds, text/image embeddings, or previously generated segments) are cached and reused.
This can be extended to support diffusion-aware caching methods (e.g., [AdaCache](https://github.com/AdaCache-DiT/AdaCache), [NIRVANA](https://github.com/iDEA-iSAIL-Lab-UIUC/NIRVANA), [MoDM](https://arxiv.org/abs/2503.11972)).
