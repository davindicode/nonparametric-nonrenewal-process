#!/bin/bash

cd ./scripts/


# 28-140313
# 12-120806
python3 HDC.py --session_id 12-120806 --phase wake --edge_bins 100 --cv_folds 5 --cv -1 0 1 2 3 4 --ncvx 2 --batch_size 500 --bin_size 500 --likelihood IPexp --mapping svgp8 --x_mode hdWP16 --lr 1e-2 --jitter 1e-3 --gpu 0
