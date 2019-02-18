#!/bin/bash
set -eu

OUTPUT_DIRPATH="data/images"
CSV_FILEPATH="data/book-data.csv"
python3 download_images.py ${OUTPUT_DIRPATH} ${CSV_FILEPATH}