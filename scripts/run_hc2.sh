#!/bin/bash

cd ./scripts/

python3 CA.py --dataset maze15_half- --cv_folds 5 --cv -1 0 1 2 3 4 --ncvx 2 --batch_size 100000 --likelihood IPPexp --mapping svgp64 --x_mode th-pos-isi1 --margin_epochs 100 --loss_margin -100 --jitter 1e-4 --lr 1e-3 --gpu 1


python3 CA.py --dataset maze15_half- --cv_folds 5 --cv -1 0 1 2 3 4 --ncvx 2 --batch_size 100000 --likelihood IPPexp --filter selfrcb8-100.-1.-1. --mapping svgp32 --x_mode th-pos --margin_epochs 100 --loss_margin -100 --jitter 1e-4 --lr 1e-3 --gpu 0

python3 CA.py --dataset maze15_half- --cv_folds 5 --cv -1 0 1 2 3 4 --ncvx 2 --batch_size 100000 --likelihood IPPexp --filter fullrcb8-100.-1.-1. --mapping svgp32 --x_mode th-pos --margin_epochs 100 --loss_margin -100 --jitter 1e-4 --lr 1e-3 --gpu 1