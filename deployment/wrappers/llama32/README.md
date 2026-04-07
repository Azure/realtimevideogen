# LLM
We use a pre-created vLLM image.
Took example from [here](https://medium.com/@amri369/self-host-llama-3-1-8-b-in-ec2-using-vllm-and-docker-1aefe4584b9a).

## Deploy
Set `HF_TOKEN` in [`set_properties.sh`](../../../set_properties.sh) for access to gated models.
```bash
source ../../../set_properties.sh
```

### Image management
For ACR setup and credentials, see [Deployment README](../../README.md#azure-container-registry-acr).

To use the `vllm-openai` image from our Azure Container Registry:
```bash
docker pull vllm/vllm-openai:v0.9.1
docker tag vllm/vllm-openai:v0.9.1 $ACR_URL/vllm/vllm-openai:v0.9.1
docker push $ACR_URL/vllm/vllm-openai:v0.9.1
```


### Llama
For Llama we require file `llama3_chat_template.tmpl`:
```jinja2
{% set sep = '\n' %}
{% for message in messages %}
{% if message['role'] == 'system' %}
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% endif %}
{% endfor %}
<|start_header_id|>assistant<|end_header_id|>
```
And then we can launch the container:
```bash
docker network create mynet
docker run \
  --runtime nvidia \
  --gpus all \
  -v $HF_HOME:/root/.cache/huggingface \
  -v $(pwd)/llama3_chat_template.tmpl:/chat_template.tmpl \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --network mynet \
  -p 18087:8000 \
  --ipc=host \
  vllm/vllm-openai:v0.9.1 \
  --model meta-llama/Meta-Llama-3.1-8B \
  --chat-template /chat_template.tmpl \
  --tensor-parallel-size 8
```

### Gemma
For running with the Gemma model:
```bash
docker run \
  --runtime nvidia \
  --gpus '"device=0,1"' \
  -v $HF_HOME:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --network mynet \
  -p 18086:8000 \
  --ipc=host \
  vllm/vllm-openai:v0.9.1 \
  --model google/gemma-3-27b-it \
  --tensor-parallel-size 2 \
  --structured-output-backend xgrammar
```


## Inference

### Llama
Then we can manually query:
```bash
HOST=localhost
PORT=8000
curl -s http://$HOST:$PORT/v1/completions -H "Content-Type: application/json" -d '{
"model": "meta-llama/Meta-Llama-3.1-8B",
"prompt": "Hello",
"max_tokens": 7,
"temperature": 0
}' | jq .choices[0].text
```

### Gemma
```bash
HOST=localhost
PORT=8000
curl -s http://$HOST:$PORT/v1/completions -H "Content-Type: application/json" -d '{
"model": "google/gemma-3-27b-it",
"prompt": "Hello",
"max_tokens": 7,
"temperature": 0
}' | jq .choices[0].text
```
