# Whisper

Audio to text via OpenAI Whisper large-v3, served through the vLLM OpenAI-compatible API
(`/v1/audio/transcriptions`) on port 8000.

The Docker image (`vllm-librosa`) extends `vllm/vllm-openai` with `librosa` and `soundfile`
for audio I/O.

## Build and push

Set the Azure Container Registry URL in [`set_properties.sh`](../../../set_properties.sh):

```bash
source ../../../set_properties.sh
```

Build the image and push it to ACR:

```bash
cd deployment/wrappers/whisper
bash setup_image.sh
```

This produces and pushes `$ACR_URL/vllm-librosa:v0.9.1`.

## Run

```bash
source ../../../set_properties.sh
export HF_HOME="/mnt/cache/huggingface"   # directory for model weights

docker run \
  --runtime nvidia \
  --gpus all \
  -v "$HF_HOME":/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  "$ACR_URL/vllm-librosa:v0.9.1" \
  --model openai/whisper-large-v3
```

## Inference

### curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "model=openai/whisper-large-v3" \
  -F "file=@audio.mp3"
```

### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="n/a")
with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=f,
        response_format="json",
        language="en",
    )
print(result.text)
```