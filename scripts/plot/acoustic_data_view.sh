#!/bin/bash
# Plot acoustic data for specific sensors and time intervals
# Usage: ./acoustic_data_view.sh

python ./scripts/plot/acoustic_data_view.py \
    --sensor 3dvha \
    --start "2023-12-01T21:00:00" \
    --end "2023-12-01T23:30:00" \
    --multi \
    --interval 60 \
    --savefig \
    --target_fs 250.0 \
    --filter highpass \
    --filt-freq 10.0

python ./scripts/plot/acoustic_data_view.py \
    --sensor vla1 \
    --start "2023-12-01T21:26:00" \
    --end "2023-12-02T01:13:00" \
    --multi \
    --interval 60 \
    --savefig \
    --target_fs 250.0 \
    --filter highpass \
    --filt-freq 10.0

python ./scripts/plot/acoustic_data_view.py \
    --sensor vla2 \
    --start "2023-12-01T21:44:00" \
    --end "2023-12-02T01:31:00" \
    --multi \
    --interval 60 \
    --savefig \
    --target_fs 250.0 \
    --filter highpass \
    --filt-freq 10.0
