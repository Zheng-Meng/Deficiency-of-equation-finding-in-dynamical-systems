import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks

def f_real(x, t, rho=28.0, beta=8.0 / 3.0, sigma=10.0):
    res = np.zeros_like(x)
    res[0] = sigma * (x[1] - x[0])
    res[1] = x[0] * (rho - x[2]) - x[1]
    res[2] = x[0] * x[1] - beta * x[2]
    
    return res

def f_sindy1(x, t, rho=25.167, beta=2.679):
    res = np.zeros_like(x)
    res[0] = -5.311 * x[0] + 6.177 * x[1] - 0.136 * x[0] * x[2] + 0.113 * x[1] * x[2]
    res[1] = rho * x[0] - 0.927 * x[0] * x[2]
    res[2] = 0.349 * x[1] - beta * x[2] + 0.951 * x[0] * x[1]
    return res

def f_sindy2(x, t, rho=28.361, beta=2.691):
    res = np.zeros_like(x)
    res[0] = -4.023 * x[0] + 5.958 * x[1] - 0.167 * x[0] * x[2] + 0.113 * x[1] * x[2]
    res[1] = rho * x[0] - 3.051 * x[1] - 1.026 * x[0] * x[2] + 0.103 * x[1] * x[2]
    res[2] = -0.105 * x[0] + 0.176 * x[1] - beta * x[2] + 0.942 * x[0] * x[1]
    return res

def f_sindy3(x, t, rho=24.176, beta=2.691):
    res = np.zeros_like(x)
    res[0] = -9.806 * x[0] + 9.846 * x[1]
    res[1] = rho * x[0] - 0.901 * x[0] * x[2]
    res[2] = -0.115 * x[0] + 0.184 * x[1] - beta * x[2] + 0.942 * x[0] * x[1]
    return res


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _local_minima(series, distance=5):
    # Local minima via peaks on inverted signal
    peaks, _ = find_peaks(-series, distance=distance)
    return series[peaks]


def _compute_bifurcation_data(
    f_handle,
    param_name,
    param_values,
    base_params,
    x0,
    t_span,
    transient=0.5,
    min_distance=5,
):
    results = []
    start_idx = int(len(t_span) * transient)
    for p in param_values:
        params = dict(base_params)
        params[param_name] = float(p)
        sol = odeint(f_handle, x0, t_span, args=tuple(params.values()))
        data = sol[start_idx:, :]
        mins = _local_minima(data[:, 1], distance=min_distance)
        results.append(mins)
    return results


def _plot_bifurcation_panel(param_values, minima, title_prefix, save_path):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for p, mins in zip(param_values, minima):
        if mins.size == 0:
            continue
        ax.plot([p] * len(mins), mins, 'k.', markersize=1)
    ax.set_xlabel(title_prefix)
    ax.set_ylabel('local minima of y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def run_bifurcation_study():
    print('Running bifurcation study...')
    save_data_dir = './save_data'
    save_fig_dir = './save_results'
    _ensure_dir(save_data_dir)
    _ensure_dir(save_fig_dir)
    print('Saving data to:', save_data_dir)
    print('Saving figures to:', save_fig_dir)

    t0 = 0.0
    tend = 2000.0
    dt = 0.02
    # starting from a random initial condition
    x0 = np.array([1.5 + np.random.uniform(-0.5, 0.5), -1.5 + np.random.uniform(-0.5, 0.5), 20.0 + np.random.uniform(-5.0, 5.0)])
    t_span = np.arange(t0, tend, dt)

    models = {
        'real': (f_real, {'rho': 28.0, 'beta': 8.0 / 3.0, 'sigma': 10.0}),
        'sindy1': (f_sindy1, {'rho': 25.167, 'beta': 2.679}),
        'sindy2': (f_sindy2, {'rho': 28.361, 'beta': 2.691}),
        'sindy3': (f_sindy3, {'rho': 24.176, 'beta': 2.691}),
    }

    # Parameter ranges (centered around default values)
    rho_range = np.linspace(9.0, 70.0, 150)
    beta_range = np.linspace(0.1, 5.0, 500)

    all_data = {}
    for model_name, (f_handle, base_params) in models.items():
        all_data[model_name] = {}
        # for param_name, param_values in [('rho', rho_range), ('beta', beta_range)]:
        for param_name, param_values in [('beta', beta_range)]:
            print('Running bifurcation study for:', param_name)
            minima = _compute_bifurcation_data(
                f_handle=f_handle,
                param_name=param_name,
                param_values=param_values,
                base_params=base_params,
                x0=x0,
                t_span=t_span,
            )
            all_data[model_name][param_name] = {
                'param_values': param_values,
                'minima': minima,
            }   
            save_path = os.path.join(
                save_fig_dir,
                f'bifurcation_{model_name}_{param_name}.png'
            )
            _plot_bifurcation_panel(
                param_values=param_values,
                minima=minima,
                title_prefix=param_name,
                save_path=save_path,
            )

    data_path = os.path.join(save_data_dir, 'bifurcation_lorenz.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(all_data, f)
if __name__ == '__main__':
    run_bifurcation_study()






















