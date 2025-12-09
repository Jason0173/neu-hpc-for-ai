#!/usr/bin/env bash
set -euo pipefail

# NVSHMEM installation target directory
NVSHMEM_PREFIX="/usr/local/nvshmem"

echo "[install_nvshmem.sh] NVSHMEM will be installed to: ${NVSHMEM_PREFIX}"

# 1. Find NVSHMEM tarball in /tmp and /project (prefer /tmp, as it may be mounted)
TARBALL=""
# Check /tmp first (mount location)
for p in /tmp/nvshmem*.txz /tmp/nvshmem*.tar.xz /tmp/nvshmem*.tar.gz; do
  if [ -f "$p" ]; then
    TARBALL="$p"
    break
  fi
done
# If not found in /tmp, check /project
if [ -z "$TARBALL" ]; then
  for p in /project/nvshmem*.txz /project/nvshmem*.tar.xz /project/nvshmem*.tar.gz; do
    if [ -f "$p" ]; then
      TARBALL="$p"
      break
    fi
  done
fi

if [ -z "$TARBALL" ]; then
  echo "[install_nvshmem.sh] ERROR: No NVSHMEM tarball found in /tmp or /project"
  exit 1
fi

echo "[install_nvshmem.sh] Using tarball: $TARBALL"

# 2. Extract to temporary directory
WORKDIR="/tmp/nvshmem_install"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

echo "[install_nvshmem.sh] Extracting to $WORKDIR ..."
tar -xf "$TARBALL" -C "$WORKDIR"

echo "[install_nvshmem.sh] After extract, contents of $WORKDIR:"
ls -R "$WORKDIR"

# 3. Find the actual extracted subdirectory (exclude WORKDIR itself)
EXTRACTED_DIR=$(find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d -name "nvshmem*" | head -n 1)
if [ -z "$EXTRACTED_DIR" ]; then
  # Some versions may not have directory name starting with nvshmem, fallback: take any subdirectory
  EXTRACTED_DIR=$(find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
fi

if [ -z "$EXTRACTED_DIR" ]; then
  echo "[install_nvshmem.sh] ERROR: Could not find extracted NVSHMEM directory in $WORKDIR"
  exit 1
fi

echo "[install_nvshmem.sh] Extracted dir: $EXTRACTED_DIR"

# 4. Clean original installation directory and copy
rm -rf "$NVSHMEM_PREFIX"
mkdir -p "$NVSHMEM_PREFIX"

echo "[install_nvshmem.sh] Copying files from $EXTRACTED_DIR to $NVSHMEM_PREFIX ..."
cp -r "$EXTRACTED_DIR"/* "$NVSHMEM_PREFIX"/

# 5. Print include directory contents to confirm nvshmem.h exists
echo "[install_nvshmem.sh] Contents of $NVSHMEM_PREFIX/include:"
ls -R "$NVSHMEM_PREFIX/include" || echo "include dir not found!"

echo "[install_nvshmem.sh] NVSHMEM installed successfully."
