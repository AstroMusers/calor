import importlib.util
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest
from tdpy.verbosity import print


def test_synthetic_example_runs_real_variable_dimension_sampler_and_saves_catalogs(tmp_path):
    path = Path(__file__).parents[1] / 'examples' / 'temperature_regimes.py'
    print(f'Reading from {path}...')
    specification = importlib.util.spec_from_file_location('calor_temperature_regime_example', path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)

    result = example.run_example(tmp_path, number_steps=600, number_chains=2)
    output_path = tmp_path / 'temperature_regimes.png'
    example.plot_result(result, output_path)
    payload_path = tmp_path / 'temperature_regime_posterior.json'
    print(f'Reading from {payload_path}...')
    payload = json.loads(payload_path.read_text())

    assert payload['simulated'] is True
    assert len(payload['catalog_samples']) == len(payload['base_samples'])
    assert len({len(catalog) for catalog in payload['catalog_samples']}) > 1
    assert np.isclose(sum(payload['break_count_probabilities']), 1.0)
    assert str(tmp_path) not in json.dumps(payload)
    print(f'Reading from {output_path}...')
    assert mpimg.imread(output_path)[..., :3].min() < 0.8


def test_bokeh_example_plot_contains_measurements_and_posterior_views(tmp_path):
    pytest.importorskip('bokeh')
    path = Path(__file__).parents[1] / 'examples' / 'temperature_regimes.py'
    print(f'Reading from {path}...')
    specification = importlib.util.spec_from_file_location('calor_temperature_bokeh_example', path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    result = example.run_example(tmp_path, number_steps=600, number_chains=2)
    output_path = tmp_path / 'temperature_regimes.html'

    example.plot_result_bokeh(result, output_path)
    print(f'Reading from {output_path}...')
    html = output_path.read_text(encoding='utf-8')

    assert 'Synthetic thermal-regime inference with PCAT' in html
    assert 'Synthetic measurements' in html
    assert 'PCAT posterior median' in html
    assert 'lower_ratio' in html
    assert 'upper_ratio' in html
    assert 'Number of trend breaks' in html
    assert 'Posterior probability' in html


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_regime_plot_themes_preserve_measurements_and_probabilities(tmp_path, monkeypatch, typeplotback, inherit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy.plotting import plot_background_colors, plot_style

    path = Path(__file__).parents[1] / 'examples' / 'temperature_regimes.py'
    print(f'Reading from {path}...')
    specification = importlib.util.spec_from_file_location('calor_temperature_plot', path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    temperature = np.linspace(500.0, 3000.0, 8)  # [K]
    base = np.array([np.log(0.95), 0.15])
    catalog = np.empty((0, 2))
    observations = example.evaluate_temperature_ratio(temperature, base, catalog)
    uncertainty = np.full(temperature.size, 0.025)
    probabilities = np.array([0.6, 0.4])
    posterior = {'base_samples': np.array([base, base]), 'catalog_samples': [catalog, catalog],
                 'break_count_probabilities': probabilities}
    result = (temperature, observations, uncertainty, posterior, base, catalog)
    background, foreground = plot_background_colors(typeplotback)
    save = plt.Figure.savefig
    captured = []

    def capture_and_save(figure, filename, **kwargs):
        captured.append(figure)
        assert figure.get_facecolor() == to_rgba(background)
        axis = figure.axes[0]
        assert axis.xaxis.label.get_color() == foreground
        assert axis.lines[0].get_color() == foreground
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), temperature)
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), observations)
        np.testing.assert_array_equal([bar.get_height() for bar in figure.axes[1].patches], probabilities)
        return save(figure, filename, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    filename = tmp_path / 'theme.png'
    with plot_style(outer_theme):
        original_settings = dict(plt.rcParams)
        assert example.plot_result(result, filename, **options) is None
        assert dict(plt.rcParams) == original_settings
    assert len(captured) == 1
    print(f'Reading from {filename}...')
    with Image.open(filename) as image:
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert image.convert('RGB').getpixel((0, 0)) == expected