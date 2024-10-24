version=$1

docker manifest create ghcr.io/xiaozheyao/scratchpad:${version}-dev --amend ghcr.io/xiaozheyao/scratchpad:${version}dev-amd64 --amend ghcr.io/xiaozheyao/scratchpad:${version}dev-arm64
docker manifest push ghcr.io/xiaozheyao/scratchpad:$1-dev
