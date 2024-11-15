# get cpu arch
arch=$(uname -m)
version=$1
buildtool=$2
# if version is not provided, raise error
if [ -z "$version" ]; then
    echo "Please provide version number"
    exit 1
fi
echo "Building image for $arch, version $version"
$buildtool build -f docker/Dockerfile.$arch-cuda . -t ghcr.io/xiaozheyao/scratchpad:${version}dev-$arch --build-arg ARCH=$arch
$buildtool push ghcr.io/xiaozheyao/scratchpad:${version}dev-$arch
