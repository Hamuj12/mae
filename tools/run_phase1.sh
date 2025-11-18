#!/bin/bash

torchrun --nproc_per_node=2 dual_yolo_mae/train_phase1.py --config configs/phase1_template.yaml --run_name phase1_test --output outputs/phase1_test
