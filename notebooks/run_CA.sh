#!/bin/bash

cd ./scripts/



python3 CA.py --dataset maze15_half- --cv -1 0 1 2 3 4 --gpu 0 --likelihood Uew3 --x_mode th-hd-pos-s-t --num_induc 64 --ncvx 2 --jitter 1e-4 --lr 1e-2 --binsize 50 --batchsize 15000 --gpu 1

python3 CA.py --dataset maze15_half- --cv -1 0 1 2 3 4 --gpu 0 --likelihood Uew3 --x_mode th-hd-pos-s-t --z_mode R1 --num_induc 72 --ncvx 2 --lr 1e-2


python3 CA.py --modes 0 --cv_folds 5 --cv -1 0 1 2 3 4 --dataset maze15_half- --ncvx 2 --batchsize 15000 --jitter 1e-4 --lr 1e-2 --binsize 10 --gpu 0
