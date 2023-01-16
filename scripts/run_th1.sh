#!/bin/bash

cd ./scripts/

  
python3 th1.py --session_name "Mouse28_140313_wake"
  
  
  
python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --mapping svgp32 --x_mode hd-isi1 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 250000 --jitter 1e-4 --gpu 0


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --mapping svgp40 --x_mode hd-isi2 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --mapping svgp48 --x_mode hd-isi3 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --gpu 1

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 --likelihood IPPexp --mapping svgp64 --x_mode hd-isi5 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 130000 --jitter 1e-4 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 3 4 -1 --likelihood IPPexp --mapping svgp64 --x_mode hd-isi5 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 130000 --jitter 1e-4 --gpu 1





python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --filter selfrcb8-100.-1.-1. --mapping svgp16 --x_mode hd --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --hist_len 100 --gpu 0


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --filter fullrcb8-100.-1.-1. --mapping svgp16 --x_mode hd --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --hist_len 100 --gpu 1