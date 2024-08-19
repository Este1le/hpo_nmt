#!/usr/bin/env bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lr
cd /exp/xzhang/hpo/Learning-Curve-Extrapolation/asha_scripts/
python training.py --config lcranknet_configs.yaml