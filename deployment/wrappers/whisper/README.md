# Whisper
Audio to text.

## Deployment
Original docker image:
```bash
docker pull onerahmet/openai-whisper-asr-webservice:v1.9.1-gpu
```

Upload to ACR:
```bash
ACR_NAME="inigogrtgen"
ACR_FULL_NAME="$ACR_NAME-cjd9f3dydte2bzbb"
ACR_URL="$ACR_FULL_NAME.azurecr.io"
IMAGE="onerahmet/openai-whisper-asr-webservice"
TAG="v1.9.1-gpu"

docker tag $IMAGE:$TAG $ACR_URL/$IMAGE:$TAG
docker push "$ACR_URL/$IMAGE:$TAG"
```

Run:
```bash
docker run "$ACR_URL/$IMAGE:$TAG"
```