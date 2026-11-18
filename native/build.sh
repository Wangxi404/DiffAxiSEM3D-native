#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
build=${AXISEM3D_BUILD_DIR:-"$root/build"}
source_dir="$build/upstream"
patch_file="$root/native/extension.patch"
revision=bc67bb2d76deb2fb208d14dff7b7b25c57d9f35f
signature="$revision $(git hash-object "$patch_file")"
mkdir -p "$build"

if [[ ! -f "$source_dir/.diff-revision" ]]; then
    repo=${AXISEM3D_REPOSITORY:-"$build/upstream.git"}
    if [[ -z "${AXISEM3D_REPOSITORY:-}" ]]; then
        git init --bare --quiet "$repo"
        if ! git -C "$repo" cat-file -e "$revision^{commit}" 2>/dev/null; then
            git -C "$repo" fetch --depth=1 --no-tags https://github.com/AxiSEMunity/AxiSEM3D.git "$revision"
        fi
    fi
    staging=$(mktemp -d "$build/.source-XXXXXX")
    trap 'rm -rf "$staging"' EXIT
    git -C "$repo" archive "$revision" CMakeLists.txt VERSION LICENSE cmake external src | tar -xf - -C "$staging"
    mkdir -p "$staging/src/core/diff"
    (cd "$staging" && patch -s -f -p1 -i "$patch_file")
    printf '%s\n' "$signature" > "$staging/.diff-revision"
    if [[ -e "$source_dir" ]]; then
        echo "The native source directory already exists: $source_dir" >&2
        exit 1
    fi
    mv "$staging" "$source_dir"
    trap - EXIT
elif [[ "$(<"$source_dir/.diff-revision")" != "$signature" ]]; then
    echo "Native source changed. Set AXISEM3D_BUILD_DIR to a fresh directory." >&2
    exit 1
fi

if [[ "${1:-}" == "--prepare-only" ]]; then
    exit 0
fi
cmake -S "$source_dir" -B "$build" "$@" \
    -DCMAKE_BUILD_TYPE=Release -DCOMPILE_DEBUG_EXECUTABLE=OFF \
    -DNPOL_SEM=2 -DUSE_DOUBLE=ON -DSERIAL_BUILD=OFF \
    -DUSE_PARALLEL_NETCDF=OFF -DWITH_HDF5=OFF
cmake --build "$build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}" --target axisem3d
