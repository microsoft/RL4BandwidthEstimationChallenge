#!/usr/bin/bash

# ***** Emulated Dataset for ACM MMSys 2024 Bandwidth Estimation Challenge *****

BLOB_NAMES=(
    emulated_dataset/emulated_dataset_chunk_0.zip
    emulated_dataset/emulated_dataset_chunk_1.zip
    emulated_dataset/emulated_dataset_chunk_2.zip
)

###############################################################

AZURE_URL="https://dnschallengepublic.blob.core.windows.net/bwechallenge-2023"

mkdir -p ./emulated_dataset/

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"

    # DRY RUN: print HTTP headers WITHOUT downloading the files
    curl -s -I "$URL" | head -n 1

    # Actually download the files - UNCOMMENT it when ready to download
    # curl "$URL" -o "$BLOB"

    # Same as above, but using wget
    # wget "$URL" -O "$BLOB"
done