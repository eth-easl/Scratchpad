# get cpu arch
arch=$(uname -m)
version=$1
# if version is not provided, raise error
if [ -z "$version" ]; then
    echo "Please provide version number"
    exit 1
fi
echo "Building image for $arch, version $version"
podman build -f docker/Dockerfile.$arch-cuda . -t ghcr.io/xiaozheyao/scratchpad:${version}dev-$arch --build-arg ARCH=$arch
podman push ghcr.io/xiaozheyao/scratchpad:${version}dev-$arch