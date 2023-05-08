#!/bin/bash


### fitting ###
cd ./fit/


# synthetic
python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --max_epochs 3000 --observed_covs x-y --observations factorized_gp-16-1000 --likelihood PP-log --device 1

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --max_epochs 3000 --observed_covs x-y --observations rate_renewal_gp-16-1000 --likelihood gamma-log --unroll 10 --joint_samples --device 1

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --observed_covs x-y --observations factorized_gp-16-1000 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood PP-log --device 1

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y --observations nonparam_pp_gp-48-matern32-matern32-1000-n2. --likelihood isi4 --device 1

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y --observations nonparam_pp_gp-48-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1


# th1
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --likelihood gamma-log --unroll 10 --joint_samples --jitter 1e-5 --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --joint_samples --likelihood invgauss-log --device 0

# NaN, set jitter?
# python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --likelihood lognorm-log --unroll 10 --joint_samples --jitter 1e-5 --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood gamma-log --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --joint_samples --device 0

# NaN, set jitter?
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood invgauss-log --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --joint_samples --jitter 1e-5 --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type svgp-8-n2.-10.-self-H150 --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern32-1000-n2. --likelihood isi4 --device 1

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern52-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern52-1000-n2. --likelihood isi4 --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern12-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern12-1000-n2. --likelihood isi4 --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern32-1000-n2. --likelihood isi4 --device 1



# hc3
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-32-1000 --likelihood PP-log --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood gamma-log --unroll 10 --joint_samples --jitter 1e-5 --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood invgauss-log --unroll 10 --joint_samples --jitter 1e-5 --device 1

# python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood lognorm-log --unroll 10 --joint_samples --device 1


python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-32-1000 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood PP-log --device 1

# NaN?
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --unroll 100 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood gamma-log --joint_samples --jitter 1e-5 --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --unroll 100 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood invgauss-log --joint_samples --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-40-1000 --filter_type svgp-8-n2.-10.-self-H150 --likelihood PP-log --device 1


python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern12-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern12-1000-n2. --likelihood isi4 --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern52-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern52-1000-n2. --likelihood isi4 --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern32-1000-n2. --likelihood isi4 --device 1





### analysis ###
cd ../analysis/

python3 analyze_synthetic.py --device 0
python3 analyze_th1.py --device 0
python3 analyze_hc3.py --device 0



### plotting ###
cd ../plots/

# main
python3 plot_schematic.py
python3 plot_synthetic.py
python3 plot_real.py

# appendix
python3 plot_baselines.py
python3 plot_BNPP_checks.py
python3 plot_real_details.py

# logo
python3 plot_logo.py