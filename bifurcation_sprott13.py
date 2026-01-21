import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


def f_real(x, t, a=2.0, b=1.0, c=2.0):
    return np.array([
        -a * x[1],
        x[0] + b * x[2] ** 2,
        1 + x[1] - c * x[2],
    ])

def f_example1(x, t, a=1.922, b=0.936, c=1.902):
    return np.array([
        0.597 - a * x[1] - 0.227 * x[2],
        -0.633 + 0.970 * x[0] - 0.322 * x[2] + b * x[2] ** 2,
        1.018 + 0.954 * x[1] - c * x[2],
    ])
    
def f_example2(x, t, a=1.991, b=0.919, c=1.950):
    return np.array([
        0.529 - a * x[1],
        -0.458 + 0.925 * x[0] + 0.154 * x[1] - 0.618 * x[2] + b * x[2] ** 2,
        0.975 + 0.961 * x[1] - c * x[2],
    ])

def f_example3(x, t, a=1.942, b=0.939, c=1.914):
    return np.array([
        0.613 - a * x[1] - 0.182 * x[2],
        -0.588 + 0.976 * x[0] - 0.271 * x[2] + b * x[2] ** 2,
        0.993 + 0.955 * x[1] - c * x[2],
    ])

def f_example4(x, t, a=1.956, b=0.854, c=1.933):
    return np.array([
        0.655 - a * x[1] - 0.130 * x[2],
        0.883 * x[0] + 0.233 * x[1] - 0.918 * x[2] + b * x[2] ** 2,
        0.983 + 0.957 * x[1] - c * x[2],
    ])

def f_example5(x, t, a=1.979, b=0.886, c=1.967):
    return np.array([
        0.692 - a * x[1],
        -0.179 + 0.915 * x[0] + 0.194 * x[1] - 0.808 * x[2] + b * x[2] ** 2,
        1.020 + 0.961 * x[1] - c * x[2],
    ])

def f_example6(x, t, a=1.912, b=0.892, c=1.900):
    return np.array([
        0.612 - a * x[1] - 0.234 * x[2],
        -0.527 + 0.888 * x[0] + 0.198 * x[1] - 0.868 * x[2] + b * x[2] ** 2,
        0.985 + 0.960 * x[1] - c * x[2],
    ])


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _local_minima(series, distance=5):
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
    t0, t1 = float(t_span[0]), float(t_span[-1])
    for value in param_values:
        params = dict(base_params)
        params[param_name] = float(value)

        def rhs(t, x):
            return f_handle(x, t, *[params[key] for key in param_order])

        sol = solve_ivp(rhs, (t0, t1), x0, t_eval=t_span, rtol=1e-6, atol=1e-9)
        if not sol.success:
            results.append(np.array([]))
            continue
        data = sol.y.T[start_idx:, :]
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
    save_data_dir = './save_data'
    save_fig_dir = './save_results'
    _ensure_dir(save_data_dir)
    _ensure_dir(save_fig_dir)

    t0 = 0.0
    tend = 15000.0
    dt = 0.1
    x0 = np.array(
        [
            0.5 + np.random.uniform(-0.25, 0.25),
            0.5 + np.random.uniform(-0.25, 0.25),
            0.5 + np.random.uniform(-0.25, 0.25),
        ]
    )
    t_span = np.arange(t0, tend, dt)

    models = {
        'real': (f_real, {'a': 2.0, 'b': 1.0, 'c': 2.0}),
        'example1': (f_example1, {'a': 1.979, 'b': 0.886, 'c': 1.967}),
        'example2': (f_example2, {'a': 1.956, 'b': 0.854, 'c': 1.933}),
        'example3': (f_example3, {'a': 1.991, 'b': 0.919, 'c': 1.950}),
        'example4': (f_example4, {'a': 1.942, 'b': 0.939, 'c': 1.914}),
        'example5': (f_example5, {'a': 1.922, 'b': 0.936, 'c': 1.902}),
        'example6': (f_example6, {'a': 1.912, 'b': 0.892, 'c': 1.900}),
    }

    param_ranges = {
        'a': np.linspace(0.1, 4.0, 500),
        # 'b': np.linspace(0.1, 6.0, 200),
        # 'c': np.linspace(0.1, 12.0, 200),
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
            save_path = os.path.join(save_fig_dir, f'bifurcation_sprott13_{model_name}_{param_name}.png')
            _plot_bifurcation_panel(
                param_values=param_values,
                minima=minima,
                title_prefix=param_name,
                save_path=save_path,
            )

    data_path = os.path.join(save_data_dir, 'bifurcation_sprott13.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(all_data, f)


if __name__ == '__main__':
    run_bifurcation_study()
