#!/bin/bash

cd ./scripts/

  
python3 synthetic.py spikes --data_path ../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --max_epochs 3000 --observed_covs x-y-theta --observations factorized_gp-32-100 --likelihood PP-log --jitter 1e-6 --device 0

python3 synthetic.py spikes --data_path ../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 100000 --observed_covs x-y-theta --observations rate_renewal_gp-32-100 --likelihood gamma-log --device 0

python3 synthetic.py spikes --data_path ../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --observed_covs x-y-theta --observations mod_renewal_gp-32-100 --likelihood gamma-log --device 0

--filter_type rcb-1-10.-1.-1.-self-H500


python3 synthetic.py spikes --data_path ../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y-theta --observations nonparam_pp_gp-8-matern32-100-12. --likelihood isi3 --device 0



python3 synthetic.py spikes --data_path ../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y-theta --latent_covs matern12d2-100-diagonal_sites-diagonal_cov-fixed_grid_locs --observations nonparam_pp_gp-8-matern32-8-spatial_full-fixed_grid-100-12. --likelihood isi3 --device 0


python3 synthetic.py spikes --data_path ../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y-theta --observations nonparam_pp_gp-8-matern32-100-12.-8-spatial_full-fixed_grid --likelihood isi3 --device 0



--latent_covs matern32d2-100-diagonal-fixed_grid
  
  
--filter_type rcb-1-10.-1.-1.-selfH500