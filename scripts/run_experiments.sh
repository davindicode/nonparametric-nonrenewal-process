#!/bin/bash

cd ./fit/


# exponential and rate-rescaled renewal
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood gamma-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood lognorm-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood invgauss-log --device 0



python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations factorized_gp-32-1000 --likelihood PP-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations rate_renewal_gp-32-1000 --unroll 10 --likelihood gamma-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations rate_renewal_gp-32-1000 --unroll 10 --likelihood lognorm-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations rate_renewal_gp-32-1000 --unroll 10 --likelihood invgauss-log --device 0


# spike history filters
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type rcb-8-3.-1.-1.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --device 0



python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations factorized_gp-32-1000 --likelihood PP-log --filter_type rcb-8-3.-1.-1.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --device 0

# modulated
# python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.1 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations mod_renewal_gp-8-1000 --likelihood gamma-log --device 0

# python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.1 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations mod_renewal_gp-8-1000 --unroll 10 --likelihood lognorm-log --freeze_params obs_model.renewal.sigma --device 0


# BNPP
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-32-matern32-1000-12. --likelihood isi4 --freeze_params obs_model.log_refract_amp --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations nonparam_pp_gp-96-matern32-1000-12. --likelihood isi4 --freeze_params obs_model.log_refract_amp --jitter 1e-5 --device 0


#--freeze_params obs_model.gp.induc_locs
#--latent_covs matern32d2-100-diagonal-fixed_grid
 
 
 
 
 
 
 
# exponential and rate-rescaled renewal
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations factorized_gp-32-1000 --likelihood PP-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations rate_renewal_gp-32-1000 --likelihood gamma-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations rate_renewal_gp-32-1000 --likelihood lognorm-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations rate_renewal_gp-32-1000 --likelihood invgauss-log --device 0



python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta-speed --observations factorized_gp-40-1000 --likelihood PP-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta-speed --observations rate_renewal_gp-40-1000 --likelihood gamma-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta-speed --observations rate_renewal_gp-40-1000 --likelihood lognorm-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta-speed --observations rate_renewal_gp-40-1000 --likelihood invgauss-log --device 0


# spike history filters
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations factorized_gp-32-1000 --filter_type rcb-8-3.-1.-1.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --likelihood PP-log --device 0


python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta-speed --observations factorized_gp-40-1000 --filter_type rcb-8-3.-1.-1.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --likelihood PP-log --device 0


# BNPP
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations nonparam_pp_gp-64-matern32-1000-12. --likelihood isi4 --freeze_params obs_model.log_refract_amp --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_L-to-R --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta-speed --observations nonparam_pp_gp-72-matern32-1000-12. --likelihood isi4 --freeze_params obs_model.log_refract_amp --device 0






  
python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --max_epochs 3000 --observed_covs x-y-theta --observations factorized_gp-32-100 --likelihood PP-log --jitter 1e-6 --device 0

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 100000 --observed_covs x-y-theta --observations rate_renewal_gp-32-100 --likelihood gamma-log --device 0

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --observed_covs x-y-theta --observations mod_renewal_gp-32-100 --likelihood gamma-log --device 0


python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --max_epochs 3000 --observed_covs x-y-theta --observations factorized_gp-32-100 --likelihood PP-log --filter_type rcb-1-10.-1.-1.-self-H500 --jitter 1e-6 --device 0

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 100000 --observed_covs x-y-theta --observations rate_renewal_gp-32-100 --likelihood gamma-log --filter_type rcb-1-10.-1.-1.-self-H500 --device 0



python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --array_type float32 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y-theta --observations nonparam_pp_gp-32-matern32-300-12. --likelihood isi4 --device 0
