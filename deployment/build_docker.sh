#!/usr/bin/env bash

# Generate the Docker images.

DEPLOYMENT_DIR=.
# shellcheck source=deployment/set_properties.sh.template
source "$DEPLOYMENT_DIR/set_properties.sh"
# shellcheck source=deployment/setup_lib.sh
source "$DEPLOYMENT_DIR/setup_lib.sh"

ensure_acr_login "$ACR_NAME"

mapfile -t IMAGES < <(jq -r 'keys[]' ../services.json)

# Build the wrappers and apps images
for IMAGE in "${IMAGES[@]}"; do
  for DIR_TYPE in "wrappers" "apps"; do
    TARGET_DIR="$DIR_TYPE/$IMAGE"
    if [ -d "$TARGET_DIR" ]; then
      (
        cd "$TARGET_DIR" || exit
        if [ -f setup_image.sh ]; then
          echo "Build $IMAGE on $TARGET_DIR..."
          bash setup_image.sh > /dev/null 2>&1
          STATUS=$?
          if [ $STATUS -ne 0 ]; then
            echo "❌ Cannot build $IMAGE"
          fi
        fi
      )
    fi
  done
done

# Push the images
ensure_acr_login "$ACR_NAME"

for IMAGE in "${IMAGES[@]}"; do
  TAG=$(docker image ls "$ACR_URL/$IMAGE" --format "{{.Tag}}" \
        | grep -v -E '^(latest|<none>)$' \
        | sort -V \
        | tail -n 1)
  if [[ -n "$TAG" ]]; then
    echo "Push $ACR_URL/$IMAGE:$TAG..."
    docker push "$ACR_URL/$IMAGE:$TAG" > /dev/null 2>&1
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
      echo "❌ Cannot push $ACR_URL/$IMAGE:$TAG"
    fi
  else
    echo "No valid tag found for $IMAGE"
  fi
done
