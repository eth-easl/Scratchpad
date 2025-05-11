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
    parser.add_argument(
        "--multistage", action="store_true", help="Use multi-stage Dockerfile"
    )
    args = parser.parse_args()
    # if version not start with v, add v in front of it
    if not args.version.startswith("v"):
        args.version = "v" + args.version

    # Get CPU architecture
    arch = platform.machine()

    # Print build information
    print(f"Building image for {arch}, version {args.version}")

    # Define and check for required patch directories
    patch_base_dir = ".buildcache/patches"
    required_patch_dirs = [
        os.path.join(patch_base_dir, "flashinfer"),
        os.path.join(patch_base_dir, "triteia"),
    ]

    for dir_path in required_patch_dirs:
        if not os.path.isdir(dir_path):
            print(f"Warning: Patch directory '{dir_path}' not found. Creating it now.")
            print(
                "         The Docker build will proceed. If patches are intended for this build,"
            )
            print(
                f"         ensure they are placed in '{dir_path}' for future builds if needed."
            )
            os.makedirs(dir_path, exist_ok=True)

    # Determine Dockerfile name based on --multistage flag
    dockerfile_name = (
        f"Dockerfile.{'multistage.' if args.multistage else ''}{arch}-cuda"
    )

    # Build docker image
    build_cmd = [
        args.buildtool,
        "build",
        "-f",
        f"meta/docker/{dockerfile_name}",
        ".",
        "-t",
        f"ghcr.io/xiaozheyao/scratchpad:{args.version}dev-{arch}",
        "--build-arg",
        f"ARCH={arch}",
    ]

    # Set environment variable for Docker buildkit
    build_env = os.environ.copy()
    # build_env["DOCKER_BUILDKIT"] = "0"

    # Execute build command
    subprocess.run(build_cmd, env=build_env, check=True)

    # --- Copy wheels from built image to local .buildcache/wheels ---
    print("Attempting to copy built wheels to local .buildcache/wheels...")
    local_wheels_dir = os.path.join(".buildcache", "wheels")
    os.makedirs(local_wheels_dir, exist_ok=True)

    image_name_tag = f"ghcr.io/xiaozheyao/scratchpad:{args.version}dev-{arch}"
    # Use a relatively unique name for the temporary container
    temp_container_name = (
        f"scratchpad_wheel_extractor_{args.version.replace('.', '_')}_{arch}"
    )

    try:
        # Create a temporary container from the built image (final stage, which has /wheels copied from builder)
        create_cmd = [
            args.buildtool,
            "create",
            "--name",
            temp_container_name,
            image_name_tag,
        ]
        print(f"Running: {' '.join(create_cmd)}")
        subprocess.run(create_cmd, check=True, capture_output=True, text=True)

        # Copy the /wheels directory from the container to the local path
        # The ":/wheels/." copies the contents of /wheels into local_wheels_dir
        cp_cmd = [
            args.buildtool,
            "cp",
            f"{temp_container_name}:/wheels/.",
            local_wheels_dir,
        ]
        print(f"Running: {' '.join(cp_cmd)}")
        subprocess.run(cp_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully copied wheels to {local_wheels_dir}")

    except subprocess.CalledProcessError as e:
        print(f"Error during wheel extraction: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        print("Proceeding without extracted wheels.")
    except FileNotFoundError:
        print(
            f"Error: '{args.buildtool}' command not found. Please ensure it is installed and in your PATH."
        )
        print("Proceeding without extracted wheels.")
    finally:
        # Remove the temporary container
        rm_cmd = [args.buildtool, "rm", temp_container_name]
        print(f"Running: {' '.join(rm_cmd)}")
        # Run rm even if previous steps failed, to ensure cleanup. check=False to avoid error if container doesn't exist.
        subprocess.run(rm_cmd, check=False, capture_output=True, text=True)
    # --- End of wheel copying ---

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
