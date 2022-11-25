import numpy as np
import sys
from scipy import signal # for convolution

from tqdm.autonotebook import tqdm

from .tools import ConsecutiveArrays, TrueIslands



def place_fields(sp_sess_rate, covariates, bin_cov, sep_t_spike, 
                 highest_rate=False, centre_size=3, sm_size=5, connect=True):
    r"""
    Detect single place field modes and smoothen activity bins
    Connect place fields or leave from the raw rate data
    
    References:
    
    [1] `Place cell discharge is extremely variable during individual passes of the rat through the firing field`
    
    André A. Fenton and Robert U. Muller
    
    """
    sys.setrecursionlimit(5000)
    units_used, bins_x, bins_y = sp_sess_rate.shape
    track_samples = len(covariates[0])
    tg_x = np.digitize(covariates[0], bin_cov[0])-1
    tg_y = np.digitize(covariates[1], bin_cov[1])-1
    
    step_wall = centre_size // 2
    c_filter = np.ones((centre_size, centre_size)) / centre_size**2

    temp_tab = np.empty((bins_x, bins_y))
    explore_tab = np.empty((bins_x, bins_y))

    def check_bounds(xl, yl, lx=0, ly=0, hx=bins_x-1, hy=bins_y-1):
        if xl < lx or yl < ly or xl > hx or yl > hy:
            return False
        return True

    def DFS(u, xl, yl):
        NN = [[xl-1,yl], [xl+1,yl], [xl,yl-1], [xl,yl+1]]
        visited[xl, yl] = fields
        for k_1 in NN:
            if check_bounds(k_1[0], k_1[1]) and sp_sess_rate[u, k_1[0], k_1[1]] > 0 and visited[k_1[0], k_1[1]] == 0:
                DFS(u, k_1[0], k_1[1])

    def find_edge(xl, yl):
        NN = [[xl-1,yl], [xl+1,yl], [xl,yl-1], [xl,yl+1]]
        neigh = 0
        for k_1 in NN:
            if check_bounds(k_1[0], k_1[1]) is False:
                return False
            if temp_tab[k_1[0], k_1[1]] == 1:
                neigh += 1

        if neigh == 4:
            temp_tab[xl, yl] = 1 # fill these up
        elif neigh > 1:
            return True

        return False

    def surrounded(xl, yl, lx, ly, hx, hy):
        NN = [[xl-1,yl], [xl+1,yl], [xl,yl-1], [xl,yl+1]]
        surr = True
        explore_tab[xl, yl] = 1
        visstep = 0
        for k_1 in NN:
            if check_bounds(k_1[0], k_1[1], lx, ly, hx, hy) is False:
                surr = False
                continue
            if explore_tab[k_1[0], k_1[1]] == 1:
                continue

            a = surrounded(k_1[0], k_1[1], lx, ly, hx, hy)
            if surr and a is False:
                surr = False

        return surr

    def fill_up(xl, yl): # neighbour filling
        NN = [[xl-1,yl], [xl+1,yl], [xl,yl-1], [xl,yl+1]]
        temp_tab[xl, yl] = 1
        for k_1 in NN:
            if temp_tab[k_1[0], k_1[1]] == 1:
                continue
            fill_up(k_1[0], k_1[1])

    # place field detection and smoothing
    place_field = np.zeros((units_used, bins_x, bins_y))
    place_centre = np.zeros((units_used, 2)) # bin of highest mean rate centre area

    iterator = tqdm(range(units_used))
    for u in iterator:
        visited = np.zeros((bins_x, bins_y)) # field id count
        fields = 1 # count now goes from 1 to fields

        for xl in range(bins_x):
            for yl in range(bins_y):
                if sp_sess_rate[u, xl, yl] > 0 and visited[xl, yl] == 0:
                    DFS(u, xl, yl)
                    fields += 1

        if highest_rate: # select the field centre with highest rate
            max_f = 0
            use_f = 0
            for f in range(1, fields+1):
                sr = sp_sess_rate[u] * (visited == f) # centre convolution
                sr_c = signal.convolve2d(sr, c_filter, boundary='symm', mode='valid')
                if (np.max(sr_c) > max_f):
                    max_f = np.max(sr_c)
                    use_f = f
                    gg = np.where(sr_c == max_f)
                    place_centre[u, 0] = gg[0][0]+step_wall
                    place_centre[u, 1] = gg[1][0]+step_wall
        else: # in the paper, they select largest field
            max_size = 0
            use_f = 0
            for f in range(1, fields+1):
                if (visited == f).sum() > max_size:
                    use_f = f
                    sr = sp_sess_rate[u] * (visited == f) # centre convolution
                    sr_c = signal.convolve2d(sr, c_filter, boundary='symm', mode='valid')
                    gg = np.where(sr_c == np.max(sr_c))
                    place_centre[u, 0] = gg[0][0]+step_wall
                    place_centre[u, 1] = gg[1][0]+step_wall

        # smoothen and connect the used placed field
        if connect:
            temp_tab = (visited == use_f)
            gz = zip(np.where(visited == 0)[0], np.where(visited == 0)[1])
            field_u = np.where(temp_tab)
            lx = field_u[0].min()
            hx = field_u[0].max()
            ly = field_u[1].min()
            hy = field_u[1].max()

            unvis = [pp for pp in gz if check_bounds(pp[0], pp[1], lx, ly, hx, hy)] # all zero activity units
            explore_tab = np.copy(temp_tab)
            for uv in unvis:
                if explore_tab[uv[0], uv[1]] == 1:
                    continue
                if surrounded(uv[0], uv[1], lx, ly, hx, hy):
                    fill_up(uv[0], uv[1])

            place_field[u] = temp_tab
        
        else:
            place_field[u] = (visited == use_f)
            
    return place_field, place_centre
        
        
        
def analyze_passes(sp_sess_rate, covariates, bin_cov, sep_t_spike, sample_bin, place_field, place_centre, pass_minsize):
    r"""
    Analyze the passes through place fields
    
    :param pass_minsize: count of minimum pass sizes in time steps
    """
    units_used, bins_x, bins_y = sp_sess_rate.shape
    track_samples = len(covariates[0])
    tg_x = np.digitize(covariates[0], bin_cov[0])-1
    tg_y = np.digitize(covariates[1], bin_cov[1])-1
    
    neurpass_spikes = []
    neurpass_avg = []
    neurpass_len = []
    neurpass_start = []
    neurpass_centre = []

    in_field = np.zeros((units_used, track_samples))
    for t in range(track_samples):
        in_field[:, t] = place_field[:, tg_x[t], tg_y[t]]

    iterator = tqdm(range(units_used))
    for u in iterator:
        # label and select reliable passes
        i_ind, i_size = TrueIslands(in_field[u])
        idz = np.where(i_size < pass_minsize)
        i_ind = np.delete(i_ind, idz)
        i_size = np.delete(i_size, idz)
        passes = len(i_ind)
        i_spikes = []
        i_phases = []
        i_avg = np.zeros(passes)
        i_centre = np.zeros(passes, dtype=bool)

        # compute spikes observed per pass
        start = i_ind
        end = (i_ind+i_size)
        for p in range(passes):
            idx = np.where((sep_t_spike[u] >= start[p]) & (sep_t_spike[u] < end[p]))[0]
            stimes = sep_t_spike[u][idx]
            i_spikes.append(stimes)
            for t in range(i_ind[p], i_ind[p]+i_size[p]):
                i_avg[p] += sp_sess_rate[u, tg_x[t], tg_y[t]] * sample_bin

            vec_centre = np.array(list(zip(place_centre[u, 0] - tg_x[start[p]:end[p]], 
                                           place_centre[u, 1] - tg_y[start[p]:end[p]])))
            if np.power(vec_centre, 2).sum(1).min() <= 2:
                i_centre[p] = True


        neurpass_spikes.append(i_spikes) # time in track sample of spike
        neurpass_avg.append(i_avg)
        neurpass_len.append(i_size)
        neurpass_start.append(i_ind)
        neurpass_centre.append(i_centre)
        
    return neurpass_spikes, neurpass_avg, neurpass_len, neurpass_start, neurpass_centre





def motion_drift(sample_bin, deltalen, behav_tuple, x_bin, y_bin, spiketimes):
    """
    motion drift and LED correction, assumed zero degrees in dir is along the x-axis
    [Theta phase–specific codes for two-dimensional position, trajectory and heading in the hippocampus]
    behav_tuple takes the form (dir_t, x_t, y_t), bins_tuple similarly
    """
    units = len(spiketimes)
    x_bins = len(x_bin)-1
    y_bins = len(y_bin)-1
    dd = [[0,7], [1,2], [3,4], [5,6]]
    dirs = 4
    dir_bins = 8
    dir_bin = np.linspace(0, 2*np.pi+1e-3, dir_bins+1)

    sess_rate, _ = IPP_model(sample_bin, behav_tuple, (dir_bin, x_bin, y_bin), spiketimes)

    dir_rate = np.empty((units, dirs, x_bins, y_bins))
    for u in range(units):
        for k in range(dirs):
            dir_rate[u, k] = sess_rate[u, dd[k]].sum(0)

    # correlation matrix EW and NS shifts
    lag = np.arange(-deltalen, deltalen+1)
    dir_corr = np.empty((units, 2, len(lag)))
    for ind, l in enumerate(lag):
        if l < deltalen:
            x_1 = dir_rate[:, 0, deltalen:-deltalen, :]
            x_2 = dir_rate[:, 2, deltalen+l:-deltalen+l, :]
        else:
            x_1 = dir_rate[:, 0, deltalen:-deltalen, :]
            x_2 = dir_rate[:, 2, deltalen+l:, :]
        std_1 = x_1.std(axis=(1,2))
        std_2 = x_2.std(axis=(1,2))
        std_1[std_1 == 0.] = -1. # division by zero as numerator is also zero, avoid it
        std_2[std_2 == 0.] = -1.
        
        dir_corr[:, 0, ind] = (((x_1 - x_1.mean(axis=(1,2), keepdims=True))*(x_2 - x_2.mean(axis=(1,2), keepdims=True))).mean(axis=(1,2)) / \
                            std_1 / std_2).mean(-1)

        if l < deltalen:
            x_1 = dir_rate[:, 1, :, deltalen:-deltalen]
            x_2 = dir_rate[:, 3, :, deltalen+l:-deltalen+l]
        else:
            x_1 = dir_rate[:, 1, :, deltalen:-deltalen]
            x_2 = dir_rate[:, 3, :, deltalen+l:]
        std_1 = x_1.std(axis=(1,2))
        std_2 = x_2.std(axis=(1,2))
        std_1[std_1 == 0.] = -1.
        std_2[std_2 == 0.] = -1.

        dir_corr[:, 1, ind] = (((x_1 - x_1.mean(axis=(1,2), keepdims=True))*(x_2 - x_2.mean(axis=(1,2), keepdims=True))).mean(axis=(1,2)) / \
                             std_1 / std_2).mean(-1)
            
    return dir_rate, dir_corr



def temporal_shift(sample_bin, eval_times, behav_tuple, bins_tuple, spiketimes, dev, folds=5):
    """
    Shift the spike trains with respect to the behaviour.
    
    References:
    
    [1] `The firing of hippocampal place cells predicts the future position of freely moving rats`,
    R. U. Muller, J. L. Kubie (1989)
    
    """
    units = len(spiketimes)
    shifts = len(eval_times)
    track_samples = len(behav_tuple[0])
    bin_size = 1

    def rate_func(behav_tuple, bins_tuple, spiketimes):
        return IPP_model(sample_bin, behav_tuple, bins_tuple, spiketimes)[0], \
                np.prod([len(b) for b in bins_tuple])*len(spiketimes)
    
    NLL = np.empty((shifts, units))
    CVs = np.empty((shifts, folds, units))

    model = mdl.nonparametrics.histogram(bins_tuple, neurons=units)
    model.set_params(sample_bin)
    
    shape = np.ones(units)
    renewal_dist = mdl.renewal.Gamma(units, 'identity', shape)
    renewal_dist.set_params(sample_bin)
    
    glm = mdl.inference.nll_optimized([model], renewal_dist)
    glm.to(dev)

    iterator = tqdm(range(shifts))
    for k in iterator:
        # shift the spike trains
        used_t_spike = [(spiketimes[u]+eval_times[k])[((spiketimes[u]+eval_times[k]) >= 0) & \
                                                       ((spiketimes[u]+eval_times[k]) < track_samples)] for u in range(units)]
        
        tbin, _, rc_t, _ = BinTrain(bin_size, sample_bin, used_t_spike, track_samples)
        fit_set, valid_set = SpikeTrainCV(folds, used_t_spike, track_samples, behav_tuple)
        cv_set = []
        for f, v in zip(fit_set, valid_set):
            _, v_size, v_spk, _ = \
                    BinTrain(bin_size, sample_bin, v[0], len(v[1][0]), _)
            cv_set.append((f[1], f[0], v[1], v_spk, v_size))
        
        n = np.arange(units)[:, None]
        _, _, NLL[k], CVs[k] = \
            eval_hist_model(glm, tbin, rate_func, cv_set, behav_tuple, model.bins_cov, rc_t, used_t_spike, neuron=n)
        
    return NLL, CVs



def local_rates(sample_bin, covariates, spiketimes, cov_bins=None, eval_loc_tuples=None):
    """
    Computes the firing rate per covariate bin traversal, PSTH. Alternatively, compute the data 
    where the covariates are within a radius to target evaluation locations.
    
    :param float sample_bin:
    :param tuple covariates:
    :param tuple cov_bins:
    :param list spiketimes:
    :param list eval_loc_tuples: list of tuples of (list targets, float radius)
    :returns: rates of shape (units, traversals), bin indices per traversal of shape (units, traversals, ...)
    :rtype: tuple
    """
    units = len(spiketimes)
    
    if cov_bins is not None:
        c_bins = ()
        tg_c = ()
        sg_c = [() for u in range(units)]
        for k, cov in enumerate(covariates):
            c_bins += (len(cov_bins[k])-1,)
            tg_c += (np.digitize(cov, cov_bins[k])-1,)
            for u in range(units):
                sg_c[u] += (np.digitize(cov[spiketimes[u]], cov_bins[k])-1,)

        # get bin traversals
        track_samples = len(covariates[0])
        glob_index = tg_c[0]
        fac = 1
        for k, cc in enumerate(c_bins[1:]):
            fac *= cc
            glob_index += tg_c[k+1]*fac

        diff = glob_index[1:]-glob_index[:-1]
        switchtimes = np.concatenate(([0], np.where(diff != 0)[0] + 1, [track_samples]))

        # get time spent in each bin traversal
        trav_times = (switchtimes[1:] - switchtimes[:-1])*sample_bin
        travs = len(trav_times)

        rate_sample = np.empty((units, travs))
        bins_sample = np.empty((units, travs, len(cov_bins))).astype(int)
        for k, trv in enumerate(trav_times):
            for u in range(units):
                act = ((spiketimes[u] >= switchtimes[k]) & (spiketimes[u] < switchtimes[k+1])).sum()
                rate_sample[u, k] = act/trv
                for l, tg in enumerate(tg_c):
                    bins_sample[u, k, l] = tg[switchtimes[k]]

        return rate_sample, bins_sample

    elif eval_loc_tuples is not None:
        # within some radius, look for the rates for each neuron per trajectory
        for eloc in eval_loc_tuples:
            radius = eloc[1]
            cov_tar = eloc[0]

            rd = np.power(np.array(cov_list) - np.array(cov_tar), 2).sum()
            indr = (rd < radius**2)
            isind, issize = tools.TrueIslands(indr)
            icont = (isind + issize // 2)

            #images = []
            #for s in range(steps):
            unit_arr = [30]
            for u in range(len(unit_arr)):
                # place cells with trajectory passage and spikes
                fig, ax = tools.draw_2d(np.transpose(smth_rate[u]), (7,12), origin='lower', aspect='equal', 
                                        cmap='plasma', vmax=smth_rate[u].max(), ticktitle='Firing rate (Hz)')
                tools.decorate_ax(ax)#, xlim=[0, bins_x], ylim=[0, bins_y])

                xscal = (bins_x-5)/bins_x # scaling due to smoothing resize
                yscal = (bins_y-5)/bins_y
                circle = plt.Circle((x_tar/delta_bin_x*xscal, y_tar/delta_bin_y*yscal), \
                                    radius/delta_bin_x*xscal, color='w', fill=False) # assume ratio 1
                ax.add_artist(circle)

                trajs = len(r_store[u][k])
                rates_m = np.empty((trajs))
                rates_sd = np.empty((trajs))
                for tr in range(trajs):
                    i = icont[tr]
                    ax.plot(x_t[i:i+traj_len]/delta_bin_x*xscal, y_t[i:i+traj_len]/delta_bin_y*yscal, color='w')
                    #ax.text(x_t[i]*ww-1.0, y_t[i]*hh-0.7, str(tr+1), color='lime', fontdict={'weight': 'bold', 'size': 12})
                    rates_m[tr] = r_store[u][k][tr].mean()
                    rates_sd[tr] = r_store[u][k][tr].std(ddof=1)/np.sqrt(len(lens))

                spec = gridspec.GridSpec(ncols=1, nrows=2,
                                     height_ratios=[3, 1])
                ax2 = fig.add_subplot(spec[1])
                ax2.errorbar(np.arange(1, trajs+1), rates_m, linestyle='', yerr=rates_sd, marker='+', capsize=5)
                ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax2.set_ylabel('Local firing rate')
                ax2.set_xlabel('Trajectory')
                ax2.grid()
                plt.show()
                
                #images.append(tools.render_image(fig))
                #plt.close(fig)
                #plt.imshow(tu.render_image(fig))
                #plt.show()

                #tools.generate_gif(images, './groups.gif', fps=25)
                return
                
    else:
        raise ValueError('Need either binning or evaluation locations.')