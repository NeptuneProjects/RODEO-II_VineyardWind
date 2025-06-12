#!/usr/bin/env bash
# This script retrieves the example data from the remote server.

# Run from the project root directory:
# REMOTE="<user>@<host>" ./retrieve_data.sh

echo "Connecting ${REMOTE}"

# Define source and destination paths:
REMOTE_DIR="/project/users/ytlin/VW_DAS"
REMOTE_DAS_DIR=${REMOTE_DIR}/20231201
REMOTE_ACOUST_DIR=${REMOTE_DIR}/SHRUs

DEST="data"
DEST_DAS_DIR=${DEST}/das
DEST_ACOUST_DIR=${DEST}/acoustic

# Make directories to store data if they don't exist:
mkdir -p ${DEST}
mkdir -p ${DEST_DAS_DIR}
mkdir -p ${DEST_ACOUST_DIR}

# Function to download files
download_files() {
    local remote_dir=$1
    local dest_dir=$2
    shift 2
    local files=("$@")

    echo "Downloading files from ${remote_dir} to ${dest_dir}:"
    mkdir -p ${dest_dir}
    for file in "${files[@]}"; do
        scp ${REMOTE}:${remote_dir}/${file} ${dest_dir}/
    done
}

# Download DAS data:
DAS_FILES=(
    "GEODES_UTC_20231201_215953.757.tdms"
    "GEODES_UTC_20231201_220023.757.tdms"
    "GEODES_UTC_20231201_220053.757.tdms"
)
download_files ${REMOTE_DAS_DIR} ${DEST_DAS_DIR} "${DAS_FILES[@]}"

# Download acoustic data for VLA1:
VLA1=VLA1_shru905
REMOTE_VLA1_DIR=${REMOTE_ACOUST_DIR}/${VLA1}
DEST_VLA1_DIR=${DEST_ACOUST_DIR}/${VLA1}
VLA1_FILES=(
    "12012125.D23"
)
download_files ${REMOTE_VLA1_DIR} ${DEST_VLA1_DIR} "${VLA1_FILES[@]}"

# Download acoustic data for VLA2:
VLA2=VLA2_shru920
REMOTE_VLA2_DIR=${REMOTE_ACOUST_DIR}/${VLA2}
DEST_VLA2_DIR=${DEST_ACOUST_DIR}/${VLA2}
VLA2_FILES=(
    "12012143.D23"
)
download_files ${REMOTE_VLA2_DIR} ${DEST_VLA2_DIR} "${VLA2_FILES[@]}"
