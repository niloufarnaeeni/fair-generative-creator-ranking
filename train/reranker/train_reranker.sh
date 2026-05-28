#!/bin/bash

set -euo pipefail

CONFIG_DIR="config"

accelerate launch train_reranker.py --config "${CONFIG_DIR}/paper_ntp.yaml"
accelerate launch train_reranker.py --config "${CONFIG_DIR}/paper_fairgroup.yaml"
accelerate launch train_reranker.py --config "${CONFIG_DIR}/paper_suppgroup.yaml"
accelerate launch train_reranker.py --config "${CONFIG_DIR}/paper_suppgroup_creator.yaml"
