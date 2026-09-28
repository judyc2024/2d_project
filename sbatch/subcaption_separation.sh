#!/usr/bin/env bash

source ~/.bashrc
set -euo pipefail

conda activate 2dproject

python subcaption_separation.py --run