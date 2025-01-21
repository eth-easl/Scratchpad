# get cpu arch
arch=$(uname -m)
version=$1
buildtool=$2
upload=$3
# if version is not provided, raise error
if [ -z "$version" ]; then
    echo "Please provide version number"
    exit 1
fi
echo "Building image for $arch, version $version"
DOCKER_BUILDKIT=0 $buildtool build -f docker/Dockerfile.$arch-cuda . -t ghcr.io/xiaozheyao/scratchpad:${version}dev-$arch --build-arg ARCH=$arch

if [ "$upload" = "upload" ]; then
    echo "Uploading image to ghcr.io"
    $buildtool push ghcr.io/xiaozheyao/scratchpad:${version}dev-$arch
fi
