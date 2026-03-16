#!/usr/bin/env bash

# Generate the Docker images.
# Build order: base image first, then all wrappers and apps.

DEPLOYMENT_DIR=.
# shellcheck source=deployment/set_properties.sh.template
source "$DEPLOYMENT_DIR/set_properties.sh"
# shellcheck source=deployment/setup_lib.sh
source "$DEPLOYMENT_DIR/setup_lib.sh"

ensure_acr_login "$ACR_NAME"

mapfile -t IMAGES < <(jq -r 'keys[]' ../services.json)

build_image() {
  local IMAGE="$1"
  local TARGET_DIR="$2"
  (
    cd "$TARGET_DIR" || exit
    if [ -f setup_image.sh ]; then
      echo "🔨 Building $IMAGE from $TARGET_DIR..."
      LOG_FILE=$(mktemp /tmp/build_"$IMAGE"_XXXXXX.log)
      if bash setup_image.sh > "$LOG_FILE" 2>&1; then
        echo "✅ Built $IMAGE"
      else
        echo "❌ Cannot build $IMAGE — build log:"
        cat "$LOG_FILE"
      fi
      rm -f "$LOG_FILE"
    fi
  )
}

# Build the base image first — all wrapper and app images depend on it
if [ -d "base" ] && [ -f "base/setup_image.sh" ]; then
  build_image "base" "base"
fi

# Build the wrappers, apps, and any other images (e.g. streamwise)
for IMAGE in "${IMAGES[@]}"; do
  [ "$IMAGE" = "base" ] && continue  # already built above
  built=false
  for TARGET_DIR in "wrappers/$IMAGE" "apps/$IMAGE" "$IMAGE"; do
    if [ -d "$TARGET_DIR" ] && [ -f "$TARGET_DIR/setup_image.sh" ]; then
      build_image "$IMAGE" "$TARGET_DIR"
      built=true
      break
    fi
  done
  if [ "$built" = false ]; then
    echo "⚠️ No setup_image.sh found for $IMAGE (skipping build)"
  fi
done

# Push the images
ensure_acr_login "$ACR_NAME"

for IMAGE in "${IMAGES[@]}"; do
  TAG=$(docker image ls "$ACR_URL/$IMAGE" --format "{{.Tag}}" \
        | grep -v -E '^(latest|<none>)$' \
        | sort -V \
        | tail -n 1)
  if [[ -n "$TAG" ]]; then
    echo "🔨 Pushing $ACR_URL/$IMAGE:$TAG..."
    if docker push "$ACR_URL/$IMAGE:$TAG" > /dev/null 2>&1; then
      echo "✅ Pushed $ACR_URL/$IMAGE:$TAG"
    else
      echo "❌ Cannot push $ACR_URL/$IMAGE:$TAG"
    fi
  else
    echo "⚠️ No valid tag found for $IMAGE (skipping push)"
  fi
done
