#!/bin/bash
time_list="time_range_file.txt"

# Check if the file exists and is not empty
if [[ ! -s "$time_list" ]]; then
    echo "Error: $time_list does not exist or is empty."
    exit 1
fi

# Read each line
while IFS= read -r t || [[ -n "$t" ]]; do
    echo "Submitting job with time: $t"
    sbatch mte_val_job.sbatch "$t"
done < "$time_list"