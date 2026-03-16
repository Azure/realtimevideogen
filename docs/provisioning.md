# Provisioning hardware and models
We frame hardware and model selection for a workload (e.g., a 2-minute medium-quality video podcast) as an optimization problem.
After selecting a configuration, the hardware and model provisioners handle setup accordingly.

## Auto-scaling
To adapt to changing demand, StreamWise runs the optimization periodically.
We start from the current configuration and add the costs for provisioning the resources and the additional time to load the models.
In this way, StreamWise may scale-out to ten H200 servers during daytime and scale-in to a single A100 server at night.

## Optimization process
It consists of two phases:

### Initial provisioning
We start with a cost-efficient baseline configuration that leverages inexpensive models and lower-cost GPUs.
Each model instance (e.g., Flux for image generation, Wan for video) is assigned a single GPU (e.g., A100).
A greedy algorithm simulates how user requests would be processed in this setup.
It represents requests as directed acyclic graphs (DAG) and prioritizes node assignments along the critical path to available resources.
Latency and cost estimates are derived from model profiles generated during on-boarding.

### Iterative refinement
Next, we apply a genetic algorithm to evolve the initial configuration by systematically exploring the latency-cost trade-off space.
For each setting, we use the greedy algorithm to estimate the latency and cost.
If a solution is unfeasible (e.g., no image generation models), it gets discarded.
This refinement process includes:
(1) adding or removing hardware resources (including Spot);
(2) switching GPU types;
(3) switching model for a task;
(4) adjusting the number of model instances; and
(5) modifying GPU allocation per instance (i.e., model parallelism).
We also add domain-specific heuristics to guide the evolution.
If the cost exceeds the budget, we switch to Spot VMs and scale-in VMs.
If latency is too high, we scale-out and try faster GPUs.

## Optimization objective
We minimize cost x TTFF ($ x seconds).
When targeting specific SLOs (e.g., real-time with a 10-second TTFF), we steer the evolution process toward configurations that satisfy this.
If we cannot find feasible solutions, it returns the closest solution.
During exploration, we navigate the Pareto frontier between latency and cost.
By default, the system balances both objectives, but this behavior can be tuned to prioritize one.


## Optimization extensions
To support several advanced refinements, we extend our basic optimization framework.

### Quality
StreamWise can choose between generating video directly at high quality or producing a lower-resolution version and using a separate model for up-scaling.
For example, we can run Fantasy Talking at 640x400 and then upscale it to 1280x800 using Real-ESRGAN.

### Disaggregation
We can disaggregate compute-intensive model components such as DiT and VAE into separate components.
FramePack DiT streams latent outputs to the VAE for decoding, which are then passed to the audio sync model.
This enables pipelined execution, independent scaling, and fine-grained resource allocation.
After disaggregation, each component is represented as a distinct DAG node used by the provisioning and optimization algorithms.

### Spot
These instances offer significant cost savings (e.g., up to 50%), naturally biasing the optimization toward their use.
However, they are subject to eviction.
We proportionally increase the number of allocated resources to the eviction risk.
This ensures that the cost reflects this risk, striking a balance between affordability and reliability.

### Multi-region
Since GPU availability varies across regions over time, StreamWise continuously monitors and aggregates this availability.
We consider both inter-region latency and bandwidth when calculating generation latency and cost.
For example, small image transfers can tolerate inter-region latency, but components like DiT and VAE should remain co-located within the same region for performance reasons.

## Hardware provisioner
We use VMs as the underlying resource pool for the K8s cluster.
The _hardware provisioner_ interacts with the cloud provider (e.g., [Azure](https://azure.microsoft.com/en-us)) via standard APIs to add/remove VMs to the cluster.
It supports heterogeneous hardware (e.g., V100, A100, H100, H200) and Spot VMs, all interconnected within a virtual network.
Where applicable, we configure [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand) and [NCCL](https://developer.nvidia.com/nccl), and set up network peering to enable cross-region.
We expose all these features (GPU, region, evictable) as K8s node labels to be used for pod scheduling via affinity rules.

## Model provisioner
The provisioning optimization completes in ~90 ms and outputs the model instances to deploy (e.g., two FantasyTalking instances on 8 x H100s and 2 x A100s).
The provisioner uses K8s interfaces to launch these instances on the designated hardware.
This process may take several minutes, as it involves pulling Docker images, loading model weights, and performing warm-up runs.
To reduce startup latency, we cache frequently used images in each region.
Model-to-hardware mapping is managed via K8s pod affinity rules.
We also support partial-GPU deployments, allowing lightweight models (e.g., Kokoro) to share a GPU using [MPS](https://docs.nvidia.com/deploy/mps/index.html) and [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html).
