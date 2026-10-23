#!/usr/bin/env python3
"""Sample a thermal-regime catalog from an explicitly synthetic Calor dataset."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calor import evaluate_temperature_ratio, infer_temperature_regimes
from calor.paths import get_data_path, get_visuals_path
from tdpy.cli import add_plot_arguments
from tdpy.plotting import apply_plot_theme, configure_plot_style, plot_background_colors, plot_style
from tdpy.verbosity import print


def run_example(output_directory, number_steps=12000, number_chains=4):
    temperature = np.linspace(500.0, 3300.0, 32)  # [K]
    truth_base = np.array([np.log(0.95), 0.15])
    truth_catalog = np.array([[np.log(1700.0 / 1000.0), -0.75]])
    uncertainty = np.full(temperature.size, 0.025)
    truth = evaluate_temperature_ratio(temperature, truth_base, truth_catalog)
    observed = truth + np.random.default_rng(17).normal(size=temperature.size) * uncertainty
    posterior = infer_temperature_regimes(
        temperature, observed, uncertainty, number_steps=number_steps,
        burn_steps=number_steps // 3, number_chains=number_chains,
        thinning=max(1, number_steps // 600), seed=23,
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        'simulated': True, 'temperature_kelvin': temperature.tolist(), 'observed_ratios': observed.tolist(),
        'ratio_uncertainties': uncertainty.tolist(), 'truth_base': truth_base.tolist(),
        'truth_catalog': truth_catalog.tolist(), 'count_prior_mean': posterior['count_prior_mean'],
        'break_temperature_bounds': posterior['break_temperature_bounds'],
        'break_count_probabilities': posterior['break_count_probabilities'].tolist(),
        'base_samples': posterior['base_samples'].tolist(),
        'catalog_samples': [catalog.tolist() for catalog in posterior['catalog_samples']],
    }
    path = output_directory / 'temperature_regime_posterior.json'
    print(f'Writing to {path}...')
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return temperature, observed, uncertainty, posterior, truth_base, truth_catalog


@plot_style()
def plot_result(result, output_path, typeplotback=None):
    if typeplotback is not None:
        configure_plot_style(typeplotback)
    background, foreground = plot_background_colors()
    temperature, observed, uncertainty, posterior, truth_base, truth_catalog = result
    grid = np.linspace(temperature.min(), temperature.max(), 300)  # [K]
    stride = max(1, len(posterior['catalog_samples']) // 1000)
    models = np.array([evaluate_temperature_ratio(grid, base, catalog)
                       for base, catalog in zip(posterior['base_samples'][::stride], posterior['catalog_samples'][::stride])])
    lower, median, upper = np.percentile(models, [16, 50, 84], axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={'width_ratios': [2, 1]},
                                facecolor=background, constrained_layout=True)
    axes[0].errorbar(temperature, observed, yerr=uncertainty, fmt='.', color=foreground, label='Synthetic measurements')
    axes[0].plot(grid, evaluate_temperature_ratio(grid, truth_base, truth_catalog), '--', color='#A51417', label='Input trend')
    axes[0].fill_between(grid, lower, upper, color='#007360', alpha=0.25)
    axes[0].plot(grid, median, color='#007360', label='PCAT posterior median')
    axes[0].set(xlabel='Irradiation temperature [K]', ylabel=r'$\mathcal{R}=T_{\rm day}/T_{\rm max}$',
                 xlim=(temperature.min(), temperature.max()))
    axes[0].legend(frameon=True, fancybox=True, framealpha=1.0)
    probabilities = posterior['break_count_probabilities']
    axes[1].bar(np.arange(len(probabilities)), probabilities, color='#007360')
    axes[1].set(xlabel='Number of trend breaks', ylabel='Posterior probability', ylim=(0, 1))
    axes[1].set_xticks(np.arange(len(probabilities)))
    for axis in axes:
        axis.grid(False)
    figure.suptitle('Synthetic thermal-regime inference with PCAT')
    apply_plot_theme(figure)
    print(f'Writing to {output_path}...')
    figure.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=background)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--output-directory', type=Path, default=get_data_path() / 'temperature_regimes')
    parser.add_argument('--figure-path', type=Path)
    arguments = parser.parse_args()
    result = run_example(arguments.output_directory, arguments.steps, arguments.chains)
    figure_path = arguments.figure_path or get_visuals_path() / f'temperature_regimes.{arguments.typefileplot}'
    plot_result(result, figure_path, arguments.typeplotback)
    print('Posterior break-count probabilities:', result[3]['break_count_probabilities'])


if __name__ == '__main__':
    main()