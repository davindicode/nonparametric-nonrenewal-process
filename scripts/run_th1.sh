#!/bin/bash

cd ./fit/


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.1 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -100 --batch_size 5000 --observed_covs hd --observations factorized_gp-32-100 --likelihood PP-log --device 0
  
--latent_covs matern32d2-100-diagonal-fixed_grid
  
--observations nonparam_pp_gp-32-matern32-10-100
  
rate_renewal_gp
mod_renewal_gp





  
  
python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --mapping svgp32 --x_mode hd-isi1 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 250000 --jitter 1e-4 --gpu 0


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --mapping svgp40 --x_mode hd-isi2 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --mapping svgp48 --x_mode hd-isi3 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --gpu 1

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 --likelihood IPPexp --mapping svgp64 --x_mode hd-isi5 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 130000 --jitter 1e-4 --gpu 0

python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 3 4 -1 --likelihood IPPexp --mapping svgp64 --x_mode hd-isi5 --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 130000 --jitter 1e-4 --gpu 1





python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --filter selfrcb8-100.-1.-1. --mapping svgp16 --x_mode hd --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --hist_len 100 --gpu 0


python3 HDC.py --session_id 12-120806 --phase wake --cv_folds 5 --cv 0 1 2 3 4 -1 --likelihood IPPexp --filter fullrcb8-100.-1.-1. --mapping svgp16 --x_mode hd --ncvx 2 --lr 1e-3 --margin_epochs 100 --loss_margin -100 --batch_size 200000 --jitter 1e-4 --hist_len 100 --gpu 1