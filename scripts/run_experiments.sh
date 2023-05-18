#!/bin/bash


# use --device DEVICE_NUM to run on GPU devices, use --force_cpu to run on CPU

### fitting ###
cd ./fit/


# synthetic
python3 synthetic.py spikes --data_path ../../data/saves/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --max_epochs 3000 --observed_covs x-y --observations factorized_gp-16-1000 --likelihood PP-log --device 1

python3 synthetic.py spikes --data_path ../../data/saves/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 30000 --max_epochs 3000 --observed_covs x-y --observations rate_renewal_gp-16-1000 --likelihood gamma-log --unroll 10 --lik_int_method MC-1 --device 0

python3 synthetic.py spikes --data_path ../../data/saves/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y --observations factorized_gp-16-1000 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood PP-log --device 1

python3 synthetic.py spikes --data_path ../../data/saves/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y --observations nonparam_pp_gp-48-matern32-matern32-1000-n2. --likelihood isi4 --device 1

python3 synthetic.py spikes --data_path ../../data/saves/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y --observations nonparam_pp_gp-48-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1


# th1
python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type svgp-6-n2.-10.-self-H150 --device 0



python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 30000 --observed_covs hd --observations rate_renewal_gp-8-1000 --likelihood gamma-log --unroll 10 --lik_int_method MC-1 --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 30000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood gamma-log --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --lik_int_method MC-1 --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 30000 --observed_covs hd --observations rate_renewal_gp-8-1000 --likelihood invgauss-log --unroll 10 --lik_int_method MC-1 --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 30000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood invgauss-log --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --lik_int_method MC-1 --jitter 1e-6 --device 0



python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern32-1000-n2. --likelihood isi4 --device 1

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern52-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern52-1000-n2. --likelihood isi4 --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern12-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern12-1000-n2. --likelihood isi4 --device 0

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 th1.py spikes --data_path ../../data/saves/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern32-1000-n2. --likelihood isi4 --device 1



# hc3
python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-32-1000 --likelihood PP-log --device 1

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-32-1000 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood PP-log --device 1

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 19 1234 1837 1637 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-40-1000 --filter_type svgp-6-n2.-10.-self-H150 --likelihood PP-log --jitter 1e-5 --device 1  # stable seeds from many tried random seeds (due to numerical instability)



python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 30000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood gamma-log --unroll 10 --lik_int_method MC-1 --jitter 1e-6 --device 0

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 30000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --unroll 10 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood gamma-log --lik_int_method MC-1 --jitter 1e-6 --device 1

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 30000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood invgauss-log --unroll 10 --lik_int_method MC-1 --jitter 1e-6 --device 0

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 30000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --unroll 10 --filter_type rcb-8-10.-20.-4.5-9.-self-H150 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood invgauss-log --lik_int_method MC-1 --jitter 1e-6 --device 1




python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --device 1

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern12-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern12-1000-n2. --likelihood isi4 --device 1

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern52-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern52-1000-n2. --likelihood isi4 --device 1

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 hc3.py spikes --data_path ../../data/saves/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern32-1000-n2. --likelihood isi4 --device 1





### analysis ###
cd ../analysis/

python3 analyze_synthetic.py --tasks 0 --batch_size 50000 --device 0
python3 analyze_synthetic.py --tasks 1 --batch_size 100 --device 0

python3 analyze_th1.py --tasks 0 --batch_size 10000 --device 0
python3 analyze_th1.py --tasks 1 --batch_size 30000 --device 0
python3 analyze_th1.py --tasks 2 --batch_size 100 --device 0
python3 analyze_th1.py --tasks 3 --batch_size 1000 --device 0

python3 analyze_hc3.py --tasks 0 --batch_size 10000 --device 0
python3 analyze_hc3.py --tasks 1 --batch_size 30000 --device 0
python3 analyze_hc3.py --tasks 2 --batch_size 100 --device 0
python3 analyze_hc3.py --tasks 3 --batch_size 1000 --device 0



### plotting ###
cd ../plots/

# main
python3 plot_schematic.py
python3 plot_synthetic.py
python3 plot_real.py

# appendix
python3 plot_baselines.py
python3 plot_NPNR_checks.py
python3 plot_real_details.py

# logo
python3 plot_logo.py