import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks

def f_example1(x, t, beta=2.701):
    return np.array([
        -9.815 * x[0] + 9.322 * x[1] + 0.107 * x[2],
        26.331 * x[0] - 0.237 * x[1] - 0.958 * x[0] * x[2],
        0.444 * x[1] - beta * x[2] + 0.956 * x[0] * x[1]
    ])

def f_example2(x, t, beta=2.703):
    return np.array([
        -9.731 * x[0] + 9.609 * x[1],
        31.072 * x[0] - 3.569 * x[1] - 1.092 * x[0] * x[2] + 0.103 * x[1] * x[2],
        -0.104 * x[0] + 0.344 * x[1] - beta * x[2] + 0.947 * x[0] * x[1]
    ])

def f_example3(x, t, beta=2.670):
    return np.array([
        -4.933 * x[0] + 6.115 * x[1] - 0.145 * x[0] * x[2] + 0.114 * x[1] * x[2],
        29.493 * x[0] - 3.173 * x[1] - 1.062 * x[0] * x[2] + 0.103 * x[1] * x[2],
        0.262 * x[1] - beta * x[2] + 0.943 * x[0] * x[1]
    ])

def f_example4(x, t, beta=2.681):
    return np.array([
        -5.654 * x[0] + 6.624 * x[1] - 0.128 * x[0] * x[2] + 0.105 * x[1] * x[2],
        25.078 * x[0] - 0.930 * x[0] * x[2],
        0.293 * x[1] - beta * x[2] + 0.942 * x[0] * x[1]
    ])

def f_example5(x, t, beta=2.691):
    return np.array([
        -9.806 * x[0] + 9.846 * x[1],
        24.176 * x[0] - 0.901 * x[0] * x[2],
        -0.113 * x[0] + 0.184 * x[1] - beta * x[2] + 0.942 * x[0] * x[1]
    ])

def f_example6(x, t, beta=2.719):
    return np.array([
        -1.939 * x[0] + 5.172 * x[1] - 0.215 * x[0] * x[2] + 0.124 * x[1] * x[2],
        22.199 * x[0] + 0.388 * x[1] - 0.846 * x[0] * x[2],
        - beta * x[2] + 0.945 * x[0] * x[1]
    ])


# additional for 30% missing data
def f_example7(x, t, beta=2.708):
    return np.array([
        -9.836 * x[0] + 9.692 * x[1],
        24.836 * x[0] - 0.921 * x[0] * x[2],
        0.278 * x[1] - beta * x[2] + 0.947 * x[0] * x[1]
    ])

def f_example8(x, t, beta=2.614):
    return np.array([
        -6.287 * x[0] + 6.849 * x[1] - 0.109 * x[0] * x[2] + 0.101 * x[1] * x[2],
        25.355 * x[0] - 0.937 * x[0] * x[2],
        -0.148 * x[0] + 0.365 * x[1] - beta * x[2] + 0.950 * x[0] * x[1]
    ])

def f_example9(x, t, beta=2.719):
    return np.array([
        -9.741 * x[0] + 9.745 * x[1],
        30.713 * x[0] - 3.710 * x[1] - 1.083 * x[0] * x[2] + 0.109 * x[1] * x[2],
        -0.310 * x[0] + 0.275 * x[1] - beta * x[2] + 0.944 * x[0] * x[1]
    ])


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
        'example1': (f_example1, {}),
        'example2': (f_example2, {}),
        'example3': (f_example3, {}),
        'example4': (f_example4, {}),
        'example5': (f_example5, {}),
        'example6': (f_example6, {}),
        'example7': (f_example7, {}),
        'example8': (f_example8, {}),
        'example9': (f_example9, {}),
    }

    # Parameter ranges (centered around default values)
    beta_range = np.linspace(0.1, 5.0, 500)

    all_data = {}
    for model_name, (f_handle, base_params) in models.items():
        all_data[model_name] = {}
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

    data_path = os.path.join(save_data_dir, 'bifurcation_lorenz_additional.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(all_data, f)
if __name__ == '__main__':
    run_bifurcation_study()


































