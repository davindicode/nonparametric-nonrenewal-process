#!/bin/bash

cd ./fit/


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.1 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type rcb-8-3.-1.-1.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --device 0



python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.1 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood expon-log --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.1 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations mod_renewal_gp-8-1000 --likelihood expon-log --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-32-matern32-1000-12. --likelihood isi4 --freeze_params obs_model.log_refract_amp --device 0



--latent_covs matern32d2-100-diagonal-fixed_grid
 