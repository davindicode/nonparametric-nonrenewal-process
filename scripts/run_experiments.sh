#!/bin/bash

cd ./fit/


### th1 ###

# exponential and rate-rescaled renewal
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood gamma-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood lognorm-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood invgauss-log --device 0



python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 100 --likelihood gamma-log --joint_samples --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 100 --likelihood lognorm-log --joint_samples --device 1

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 100 --likelihood invgauss-log --joint_samples --device 0



python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations factorized_gp-32-1000 --likelihood PP-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations rate_renewal_gp-32-1000 --unroll 10 --joint_samples --likelihood gamma-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations rate_renewal_gp-32-1000 --unroll 10 --likelihood lognorm-log --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations rate_renewal_gp-32-1000 --unroll 10 --likelihood invgauss-log --device 0


# spike history filters
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type rcb-8-0.-3.-1.-1.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type rcb-16-17.-36.-6.-30.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations rate_renewal_gp-8-1000 --unroll 10 --likelihood lognorm-log --filter_type rcb-16-17.-36.-6.-30.-self-H500 --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations factorized_gp-8-1000 --likelihood PP-log --filter_type svgp-8-1.-30.-self-H500 --freeze_params obs_model.spikefilter.gp.kernel. --device 0 --double_arrays



python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations factorized_gp-32-1000 --likelihood PP-log --filter_type rcb-8-3.-1.-1.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --device 0


# BNPP
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern32-1000-n6. --likelihood isi4 --freeze_params obs_model.mean_amp --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern32-matern32-1000-n2. --likelihood isi4 --device 0



python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern32-1000-n6. --likelihood isi4 --freeze_params obs_model.mean_amp --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations nonparam_pp_gp-40-matern12-matern32-1000-n2. --likelihood isi4 --device 0




python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations nonparam_pp_gp-96-matern32-1000-12. --likelihood isi4 --freeze_params obs_model.log_refract_amp --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations nonparam_pp_gp-96-matern32-1000-2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd-x-y-speed --observations nonparam_pp_gp-96-matern32-1000-2. --likelihood isi4 --device 0


# modulated
python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations mod_renewal_gp-8-1000 --likelihood expon-log --freeze_params obs_model.log_scale_tau --device 0


python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations mod_renewal_gp-8-1000 --likelihood gamma-log --freeze_params obs_model.log_scale_tau --device 0

python3 th1.py spikes --data_path ../../data/th1/ --session_name Mouse28_140313_wake_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs hd --observations mod_renewal_gp-8-1000 --unroll 10 --likelihood lognorm-log --freeze_params obs_model.log_scale_tau --device 0


#--latent_covs matern32d2-100-diagonal-fixed_grid




### hc3 ###

# exponential and rate-rescaled renewal
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-32-1000 --likelihood PP-log --device 0


python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood gamma-log --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood lognorm-log --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood invgauss-log --device 1


python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood gamma-log --joint_samples --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood lognorm-log --joint_samples --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations rate_renewal_gp-32-1000 --likelihood invgauss-log --joint_samples --device 1



python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-theta --observations factorized_gp-40-1000 --likelihood PP-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-theta --observations rate_renewal_gp-40-1000 --likelihood gamma-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations rate_renewal_gp-40-1000 --likelihood lognorm-log --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations rate_renewal_gp-40-1000 --likelihood invgauss-log --device 1


# spike history filters
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations factorized_gp-32-1000 --filter_type rcb-8-17.-36.-6.-30.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood PP-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-theta --observations factorized_gp-32-1000 --filter_type rcb-8-17.-36.-6.-30.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --likelihood PP-log --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-theta --observations factorized_gp-32-1000 --filter_type rcb-8-17.-36.-6.-30.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood PP-log --device 1




python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-theta --observations factorized_gp-32-1000 --filter_type svgp-10-3.-10.-self-H500 --likelihood PP-log --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-theta --observations factorized_gp-40-1000 --filter_type svgp-10-3.-10.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c --likelihood PP-log --device 0


# BNPP
python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n6. --likelihood isi4 --freeze_params obs_model.mean_amp --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --device 1


python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern32-1000-n6. --likelihood isi4 --freeze_params obs_model.mean_amp --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-hd-theta --observations nonparam_pp_gp-64-matern12-matern32-1000-n2. --likelihood isi4 --device 1




python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-theta --observations nonparam_pp_gp-64-matern32-matern32-1000-n2. --likelihood isi4 --device 0




python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-hd-theta --observations nonparam_pp_gp-72-matern32-matern32-1000-n12. --likelihood isi4 --freeze_params obs_model.mean_amp --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-hd-theta --observations nonparam_pp_gp-72-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 1

python3 hc3.py spikes --data_path ../../data/hc3/ --session_name ec014.29_ec014.468_isi5 --select_frac 0.0 0.5 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -10 --batch_size 10000 --observed_covs x-speed-hd-theta --observations nonparam_pp_gp-72-matern32-matern32-1000-n2. --likelihood isi4 --device 1





### synthetic ###
python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --max_epochs 3000 --observed_covs x-y --observations factorized_gp-16-1000 --likelihood PP-log --device 0


python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --max_epochs 3000 --observed_covs x-y --observations rate_renewal_gp-16-1000 --likelihood gamma-log --device 0


python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.98 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 50000 --observed_covs x-y --observations factorized_gp-16-1000 --filter_type rcb-8-17.-36.-6.-30.-self-H500 --freeze_params obs_model.spikefilter.a obs_model.spikefilter.log_c obs_model.spikefilter.phi --likelihood PP-log --device 0



python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y --observations nonparam_pp_gp-48-matern32-matern32-1000-n2. --likelihood isi4 --device 0

python3 synthetic.py spikes --data_path ../../data/synthetic/ --session_name syn_data_seed123 --select_frac 0.0 1.0 --seeds 1 2 3 --lr_start 1e-2 --lr_decay 0.998 --lr_end 1e-4 --margin_epochs 100 --loss_margin -1 --batch_size 10000 --observed_covs x-y --observations nonparam_pp_gp-48-matern32-matern32-1000-n2. --likelihood isi4 --freeze_params obs_model.log_warp_tau --device 0




### analysis ###
python3 analyze_synthetic.py --device 0
python3 analyze_th1.py --device 0
python3 analyze_hc3.py --device 0