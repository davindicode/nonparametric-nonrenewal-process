#!/bin/bash

cd ./scripts/

  
python3 synthetic.py --datatype 0 --cv_folds 5 --cv -1 0 1 2 3 4 --likelihood IGexp --filter selfsvgp8 --mapping svgp32 --x_mode hd-isi1 --ncvx 2 --lr 1e-2 --jitter 1e-4 --hist_len 10 --batch_size 200000 --gpu 0


python3 synthetic.py --datatype 0 --cv_folds 5 --cv -1 0 1 2 3 4 --likelihood IGexp --mapping svgp16 --x_mode hd --ncvx 2 --lr 1e-3 --jitter 1e-4 --batch_size 200000 --gpu 0


python3 synthetic.py --datatype 0 --cv_folds 5 --cv -1 0 1 2 3 4 --likelihood IPPexp --mapping svgp32 --x_mode hd-isi1 --ncvx 2 --lr 1e-3 --jitter 1e-4 --batch_size 200000 --gpu 0