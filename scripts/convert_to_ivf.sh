#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 input.mp4"
    exit 1
fi

input_file="$1"
output_file="${input_file%.*}.ivf"

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found"
    exit 1
fi

ffmpeg -i "$input_file" -c:v libvpx -an -f ivf "$output_file"

if [ $? -eq 0 ]; then
    echo "Successfully converted '$input_file' to '$output_file'"
else
    echo "Error converting file"
    exit 1
fi
