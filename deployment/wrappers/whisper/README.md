# Whisper
Audio to text.

## Deployment
Original docker image:
```bash
docker pull onerahmet/openai-whisper-asr-webservice:v1.9.1-gpu
```

Upload to ACR:
```bash
source set_properties.sh

IMAGE="onerahmet/openai-whisper-asr-webservice"
TAG="v1.9.1-gpu"

docker tag $IMAGE:$TAG $ACR_URL/$IMAGE:$TAG
docker push "$ACR_URL/$IMAGE:$TAG"
```

Run:
```bash
docker run "$ACR_URL/$IMAGE:$TAG"
```