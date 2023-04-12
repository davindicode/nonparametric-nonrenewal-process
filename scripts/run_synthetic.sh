#!/bin/bash

cd ./fit/

  
python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --max_epochs 3000 --observed_covs x-y-theta --observations factorized_gp-32-100 --likelihood PP-log --jitter 1e-6 --device 0

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 100000 --observed_covs x-y-theta --observations rate_renewal_gp-32-100 --likelihood gamma-log --device 0

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --observed_covs x-y-theta --observations mod_renewal_gp-32-100 --likelihood gamma-log --device 0


python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --max_epochs 3000 --observed_covs x-y-theta --observations factorized_gp-32-100 --likelihood PP-log --filter_type rcb-1-10.-1.-1.-self-H500 --jitter 1e-6 --device 0

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 100000 --observed_covs x-y-theta --observations rate_renewal_gp-32-100 --likelihood gamma-log --filter_type rcb-1-10.-1.-1.-self-H500 --device 0



python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y-theta --observations nonparam_pp_gp-32-matern32-300-12. --likelihood isi4 --device 0


--latent_covs matern12d2-100-diagonal_sites-diagonal_cov-fixed_grid_locs

--latent_covs matern32d2-100-diagonal-fixed_grid
  
--filter_type rcb-1-10.-1.-1.-selfH500