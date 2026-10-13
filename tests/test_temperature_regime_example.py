import importlib.util
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


def test_synthetic_example_runs_real_variable_dimension_sampler_and_saves_catalogs(tmp_path):
    path = Path(__file__).parents[1] / 'examples' / 'temperature_regimes.py'
    specification = importlib.util.spec_from_file_location('calor_temperature_regime_example', path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)

    result = example.run_example(tmp_path, number_steps=600, number_chains=2)
    output_path = tmp_path / 'temperature_regimes.png'
    example.plot_result(result, output_path)
    print(f'Reading from {tmp_path / "temperature_regime_posterior.json"}...')
    payload = json.loads((tmp_path / 'temperature_regime_posterior.json').read_text())

    assert payload['simulated'] is True
    assert len(payload['catalog_samples']) == len(payload['base_samples'])
    assert len({len(catalog) for catalog in payload['catalog_samples']}) > 1
    assert np.isclose(sum(payload['break_count_probabilities']), 1.0)
    assert str(tmp_path) not in json.dumps(payload)
    assert mpimg.imread(output_path)[..., :3].min() < 0.8