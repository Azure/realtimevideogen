#!/usr/bin/env bash
# CI utility functions for the Docker workflow.

# wait_for_service <name> <health_url> <container> [retries=30] [delay=2] [jq_path=.status]
# Polls the health endpoint until the jq_path field equals "ok".
wait_for_service() {
  local name="$1"
  local health_url="$2"
  local container="$3"
  local retries="${4:-30}"
  local delay="${5:-2}"
  local jq_path="${6:-.status}"
  local i STATUS
  for i in $(seq 1 "$retries"); do
    STATUS=$(curl -sf "$health_url" 2>/dev/null | jq -r "${jq_path} // empty" 2>/dev/null)
    if [ "$STATUS" = "ok" ]; then
      echo "✅ $name is ready"
      return 0
    fi
    echo "Waiting... ($i/$retries) status=${STATUS:-loading}"
    sleep "$delay"
  done
  echo "❌ $name did not become ready in time" >&2
  docker logs "$container" >&2 || true
  return 1
}

# download_pages <name> <base_url> <output_dir> <path:dest> [<path:dest> ...]
# Downloads each <base_url><path> to <output_dir>/<dest>.
# Parent directories are created automatically.
download_pages() {
  local name="$1"
  local base_url="$2"
  local output_dir="$3"
  shift 3
  local pair url_path dest dest_dir
  while [[ $# -gt 0 ]]; do
    pair="$1"
    url_path="${pair%%:*}"
    dest="${pair##*:}"
    dest_dir="$(dirname "$output_dir/$dest")"
    mkdir -p "$dest_dir"
    curl -sf "${base_url}${url_path}" -o "$output_dir/$dest" \
      || { echo "❌ Failed to download $name $dest" >&2; return 1; }
    shift
  done
  echo "✅ Downloaded $name pages"
}
