import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks


def f_real(x, t, a=1.0, b=1.0, c=1.0):
    return np.array([
        a * x[1] * x[2],
        b * x[0] - x[1],
        1 - c * x[0] * x[1]
    ])

def f_example1(x, t, a=0.953, b=0.969, c=0.864):
    return np.array([
        -0.164 * x[1] + a * x[1] * x[2],
        b * x[0] - 0.972 * x[1],
        0.871 - c * x[0] * x[1]
    ])

def f_example2(x, t, a=1.023, b=1.024, c=0.912):
    return np.array([
        0.123 * x[2] + a * x[1] * x[2],
        b * x[0] - 1.013 * x[1],
        0.776 - 0.237 * x[1] - c * x[0] * x[1]
    ])

def f_example3(x, t, a=0.977, b=1.009, c=0.917):
    return np.array([
        a * x[1] * x[2],
        -0.173 + b * x[0] - 1.016 * x[1],
        0.851 - c * x[0] * x[1]  # 2313.101 - 975.424 - 1336.826 = 0.851
    ])


def f_example4(x, t, a=0.921, b=0.957, c=0.911):
    return np.array([
        0.103 * x[0] - 0.262 * x[1] + a * x[1] * x[2],
        b * x[0] - 1.013 * x[1],
        0.905 - c * x[0] * x[1]  # 1635.500 - 1634.595 = 0.905
    ])

def f_example5(x, t, a=0.856, b=0.946, c=0.886):
    return np.array([
        -0.227 * x[1] - 0.127 * x[2] + a * x[1] * x[2],
        -0.38 + b * x[0] - 0.961 * x[1],
        0.958 + 0.609 * x[1] - c * x[0] * x[1]
    ])


def f_example6(x, t, a=0.954, b=0.964, c=0.864):
    return np.array([
        -0.158 * x[1] + a * x[1] * x[2],
        -0.141 + b * x[0] - 0.964 * x[1],
        0.792 - c * x[0] * x[1]  # 3393.071 - 1810.866 - 1581.413 = 0.792
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
    param_order=('a', 'b', 'c'),
):
    results = []
    start_idx = int(len(t_span) * transient)
    for p in param_values:
        params = dict(base_params)
        params[param_name] = float(p)
        sol = odeint(f_handle, x0, t_span, args=tuple(params[key] for key in param_order))
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
    tend = 15000.0
    dt = 0.1
    # starting from a random initial condition
    x0 = np.array([0.1 + np.random.uniform(-0.05, 0.05), 0.1 + np.random.uniform(-0.05, 0.05), 0.1 + np.random.uniform(-0.05, 0.05)])
    t_span = np.arange(t0, tend, dt)

    models = {
        'real': (f_real, {'a': 1.0, 'b': 1.0, 'c': 1.0}),
        'example1': (f_example1, {'a': 0.953, 'b': 0.969, 'c': 0.864}),
        'example2': (f_example2, {'a': 1.023, 'b': 1.024, 'c': 0.912}),
        'example3': (f_example3, {'a': 0.977, 'b': 1.009, 'c': 0.917}),
        'example4': (f_example4, {'a': 0.921, 'b': 0.957, 'c': 0.911}),
        'example5': (f_example5, {'a': 0.856, 'b': 0.946, 'c': 0.886}),
        'example6': (f_example6, {'a': 0.954, 'b': 0.964, 'c': 0.864}),
    }

    param_ranges = {
        # 'a': np.linspace(0.1, 6.0, 200),
        'b': np.linspace(0.1, 6.0, 500),
        # 'c': np.linspace(0.1, 6.0, 200),
    }

    all_data = {}
    for model_name, (f_handle, base_params) in models.items():
        all_data[model_name] = {}
        for param_name, param_values in param_ranges.items():
            print('Running bifurcation study for:', model_name, param_name)
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
                f'bifurcation_sprott1_{model_name}_{param_name}.png'
            )
            _plot_bifurcation_panel(
                param_values=param_values,
                minima=minima,
                title_prefix=param_name,
                save_path=save_path,
            )

    data_path = os.path.join(save_data_dir, 'bifurcation_sprott1.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(all_data, f)
if __name__ == '__main__':
    run_bifurcation_study()






















