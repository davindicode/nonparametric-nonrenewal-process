import numpy as np
import torch
import torch.optim as optim

import subprocess
import os
import argparse



import pickle

import sys
sys.path.append("../..")

from neuroprob import stats, tools, neural_utils
import neuroprob.models as mdl

import models





def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run diode simulations."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    
    parser.add_argument('--modes', nargs='+', type=int)
    parser.add_argument('--gpu', default=0, type=int)
    
    args = parser.parse_args()
    return args




def main():
    parser = init_argparse()
    
    dev = tools.PyTorch(gpu=parser.gpu)

    session_id = 'hc5_15'
    data = np.load('./checkpoint/{}.npz'.format(session_id))
    spktrain = data['spktrain']
    x_t = data['x_t']
    y_t = data['y_t']
    s_t = data['s_t']
    dir_t = data['dir_t']
    hd_t = data['hd_t']
    theta_t = data['theta_t']
    arena = data['arena']

    sample_bin = 0.0008

    left_x, right_x, bottom_y, top_y = arena
    units_used = spktrain.shape[0]

    # binning
    def binning(bin_size, spktrain):
        tbin, resamples, rc_t, (rhd_t, rth_t, rx_t, ry_t) = neural_utils.BinTrain(bin_size, sample_bin, spktrain, 
                                                            spktrain.shape[1], (np.unwrap(hd_t), np.unwrap(theta_t), x_t, y_t), 
                                                            average_behav=True, binned=True)


        rw_t = (rhd_t[1:]-rhd_t[:-1])/tbin
        rw_t = np.concatenate((rw_t, rw_t[-1:]))

        rvx_t = (rx_t[1:]-rx_t[:-1])/tbin
        rvy_t = (ry_t[1:]-ry_t[:-1])/tbin
        rs_t = np.sqrt(rvx_t**2 + rvy_t**2)
        rs_t = np.concatenate((rs_t, rs_t[-1:]))
        rtime_t = np.arange(resamples)*tbin

        units_used = rc_t.shape[0]
        rcov = (rx_t, ry_t, tools.WrapPi(rth_t, True), rs_t, tools.WrapPi(rhd_t, True), rw_t, rtime_t)
        return rcov, units_used, tbin, resamples, rc_t
    
    
    rcov, units_used, tbin, resamples, rc_t = binning(50, spktrain) # 40 ms
    
    
    # GP with timeshift regressors model fit
    nonconvex_trials = 2

    modes_tot = [#('IP', 'hd', None, 8), 
                 ('IP', 'pos_th', None, 24)]
    
    modes = [modes_tot[m] for m in parser.modes]

    S = 20
    shifts = np.arange(-S, S+1)
    T_s = resamples - 2*S

    rc_t_s = rc_t[:, S:S+T_s]

    folds = 10
    cv_runs = [2, 4, 6, None]

    for shift in shifts:
        print(shift)

        rcov_s = [cc[shift+S:shift+S+T_s] for cc in rcov]

        for m in modes:
            ll_mode, r_mode, spk_cpl, num_induc = m
            print(m)

            cv_set = neural_utils.SpikeTrainCV(folds, rc_t_s, T_s, rcov_s)
            for kcv in cv_runs:
                if kcv is not None:
                    ftrain, fcov, vtrain, vcov = cv_set[kcv]
                else:
                    ftrain = rc_t
                    fcov = rcov

                lowest_loss = np.inf # nonconvex pick the best
                for kk in range(nonconvex_trials):

                    retries = 0
                    while True:
                        try:
                            glm, _ = models.set_glm(r_mode, ll_mode, spk_cpl, fcov, units_used, tbin, ftrain, num_induc, 
                                                    inv_link='exp', batch_size=150000)
                            glm.to(dev)

                            # fit
                            sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
                            opt_tuple = (optim.Adam, 100, sch)
                            opt_lr_dict = {'default': 1e-2}
                            glm.set_optimizers(opt_tuple, opt_lr_dict)#, nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

                            annealing = lambda x: 1.0#min(1.0, 0.002*x)

                            losses = glm.fit(3000, loss_margin=-1e1, stop_iters=100, anneal_func=annealing, cov_samples=1, ll_samples=10)
                            break
                        except (RuntimeError, AssertionError):
                            print('Retrying...')
                            if retries == 2: # max retries
                                print('Stopped after max retries.')
                                raise ValueError
                            retries += 1

                    if losses[-1] < lowest_loss:
                        lowest_loss = losses[-1]

                        plt.figure()
                        plt.plot(losses)
                        plt.xlabel('epoch')
                        plt.ylabel('NLL')
                        plt.show()

                        # save model
                        model_name = 'GPR_{}_shift={}_cv={}'.format(session_id, shift, kcv)
                        torch.save({'glm': glm.state_dict()}, './checkpoint/' + model_name)

                
        
if __name__ == "__main__":
    main()