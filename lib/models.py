import sys
sys.path.append("..")

import lib, jax
import jax.numpy as jnp








def get_model(kernel, mapping, likelihood, x_dims, y_dims, tbin, 
              obs_mc=20, lik_gh=20, seed=123, dtype=jnp.float32):
    # likelihood
    if likelihood == 'Normal':
        var_y = 1.*jnp.ones(y_dims)
        lik = lib.likelihoods.Gaussian(y_dims, variance=var_y)
        f_dims = y_dims
    elif likelihood == 'hNormal':
        lik = lib.likelihoods.HeteroscedasticGaussian(y_dims)#, autodiff=False)
        f_dims = y_dims*2
    elif likelihood == 'Poisson':
        lik = lib.likelihoods.Poisson(y_dims, tbin)#, autodiff=True)
        f_dims = y_dims
    lik.set_approx_integration(approx_int_method='GH', num_approx_pts=lik_gh)

    # mapping
    if mapping == 'Id':
        x_dims = f_dims
        obs = lib.observations.Identity(f_dims, lik)
    elif mapping == 'Lin':
        C = 0.1*jax.random.normal(jax.random.PRNGKey(seed), 
                                  shape=(f_dims, x_dims))#jnp.ones((f_dims, x_dims))
        #C = C/np.linalg.norm(C)
        b = 0.*jnp.ones((f_dims,))
        obs = lib.observations.Linear(lik, C, b)
    elif mapping == 'bLin':
        mean_f = jnp.zeros(f_dims)
        scale_C = 1.*jnp.ones((1, x_dims)) # shared across f_dims
        blin_site_params = {'K_eta_mu': jnp.ones((f_dims, x_dims, 1)), 
                            #jax.random.normal(jax.random.PRNGKey(seed), shape=(f_dims, x_dims, 1)),  
                            'chol_K_prec_K': 1.*jnp.eye(x_dims)[None, ...].repeat(f_dims, axis=0)}
        obs = lib.observations.BayesianLinear(mean_f, scale_C, blin_site_params, lik, 
                                              jitter=1e-5)
    elif mapping == 'SVGP':
        len_fx = 1.*jnp.ones((f_dims, x_dims)) # GP lengthscale
        var_f = 1.*jnp.ones(f_dims) # kernel variance
        kern = lib.kernels.SquaredExponential(f_dims, variance=var_f, lengthscale=len_fx)
        mean_f = jnp.zeros(f_dims)
        num_induc = 10
        induc_locs = jax.random.normal(jax.random.PRNGKey(seed), shape=(f_dims, num_induc, x_dims))
        svgp_site_params = {'K_eta_mu': 1.*jax.random.normal(jax.random.PRNGKey(seed), shape=(f_dims, num_induc, 1)),  
                            'chol_K_prec_K': 1.*jnp.eye(num_induc)[None, ...].repeat(f_dims, axis=0)}
        obs = lib.observations.SVGP(kern, mean_f, induc_locs, svgp_site_params, lik, 
                                    jitter=1e-5)


    obs.set_approx_integration(approx_int_method='MC', num_approx_pts=obs_mc)

    # state space LDS
    if kernel == 'Mat32':
        var_x = 1.0*jnp.ones(x_dims)  # GP variance
        len_x = 10.0*jnp.ones((x_dims, 1))  # GP lengthscale
        kernx = lib.kernels.Matern32(x_dims, variance=var_x, lengthscale=len_x)
        
    elif kernel == 'LEG':
        N, R, B, Lam = lib.kernels.LEG.initialize_hyperparams(jax.random.PRNGKey(seed), 3, x_dims)
        kernx = lib.kernels.LEG(N, R, B, Lam)
        
    state_space = lib.latents.LinearDynamicalSystem(kernx, diagonal_site=True)
    
    
    model = lib.inference.CVI_SSGP(state_space, obs, dtype=dtype)
    name = '{}x_{}y_{}ms_'.format(x_dims, y_dims, int(tbin*1000)) + kernel + '_' + mapping + '_' + likelihood
    return model, name




def split_params_func_GD(all_params, split_param_func=None):
    params, site_params = all_params['hyp'], all_params['sites']
    
    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)
    
    # site params
    learned_sp = lib.utils.copy_pytree(site_params)
    fixed_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)
    
    return {'hyp': learned, 'sites': learned_sp}, {'hyp': fixed, 'sites': fixed_sp}



def split_params_func_NGDx(all_params, split_param_func=None):
    params, site_params = all_params['hyp'], all_params['sites']
    
    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)
    
    # site params
    learned_sp = lib.utils.copy_pytree(site_params)
    fixed_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)
    for k in site_params['state_space'].keys():
        learned_sp['state_space'][k] = jnp.empty(0)
        fixed_sp['state_space'][k] = site_params['state_space'][k]
    
    return {'hyp': learned, 'sites': learned_sp}, {'hyp': fixed, 'sites': fixed_sp}




def split_params_func_NGDu(all_params, split_param_func=None):
    params, site_params = all_params['hyp'], all_params['sites']
    
    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)
    
    # site params
    learned_sp = lib.utils.copy_pytree(site_params)
    fixed_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)
    for k in site_params['observation'].keys():
        learned_sp['observation'][k] = jnp.empty(0)
        fixed_sp['observation'][k] = site_params['observation'][k]
    
    return {'hyp': learned, 'sites': learned_sp}, {'hyp': fixed, 'sites': fixed_sp}




def split_params_func_NGD(all_params, split_param_func=None):
    params, site_params = all_params['hyp'], all_params['sites']
    
    # params
    if split_param_func is not None:
        learned, fixed = split_param_func(params)
    else:
        learned = lib.utils.copy_pytree(params)
        fixed = jax.tree_map(lambda x: jnp.empty(0), params)
    
    # site params
    learned_sp = jax.tree_map(lambda x: jnp.empty(0), site_params)
    fixed_sp = lib.utils.copy_pytree(site_params)
    
    return {'hyp': learned, 'sites': learned_sp}, {'hyp': fixed, 'sites': fixed_sp}



def overwrite_GD(all_params, site_params):
    return all_params


def overwrite_NGDx(all_params, site_params):
    all_params['sites']['state_space'] = site_params['state_space'] # overwrite with NGDs
    return all_params


def overwrite_NGDu(all_params, site_params):
    all_params['sites']['observation'] = site_params['observation'] # overwrite with NGDs
    return all_params


def overwrite_NGD(all_params, site_params):
    all_params['sites'] = site_params # overwrite with NGDs
    return all_params