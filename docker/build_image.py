#!/usr/bin/env python3
import os
import platform
import subprocess
import argparse


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Build Docker image for Scratchpad")
    parser.add_argument("version", help="Version number for the image")
    parser.add_argument(
        "--buildtool", default="docker", help="Build tool to use (default: docker)"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload the image after building"
    )
    args = parser.parse_args()

    # Get CPU architecture
    arch = platform.machine()

    # Print build information
    print(f"Building image for {arch}, version {args.version}")

    # Build docker image
    build_cmd = [
        args.buildtool,
        "build",
        "-f",
        f"docker/Dockerfile.{arch}-cuda",
        ".",
        "-t",
        f"ghcr.io/xiaozheyao/scratchpad:{args.version}dev-{arch}",
        "--build-arg",
        f"ARCH={arch}",
    ]

    # Set environment variable for Docker buildkit
    build_env = os.environ.copy()
    build_env["DOCKER_BUILDKIT"] = "0"

    # Execute build command
    subprocess.run(build_cmd, env=build_env, check=True)

    # Upload image if specified
    if args.upload:
        print("Uploading image to ghcr.io")
        push_cmd = [
            args.buildtool,
            "push",
            f"ghcr.io/xiaozheyao/scratchpad:{args.version}dev-{arch}",
        ]
        subprocess.run(push_cmd, check=True)


if __name__ == "__main__":
    main()
