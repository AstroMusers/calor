import numpy as np
import pytest

from calor import evaluate_temperature_ratio, infer_temperature_regimes, temperature_ratio_log_likelihood


def test_variable_break_model_contains_power_law_and_is_continuous():
    temperature = np.array([500.0, 1000.0, 2000.0])  # [K]
    empty = np.empty((0, 2))
    np.testing.assert_allclose(evaluate_temperature_ratio(temperature, [np.log(0.9), 0.2], empty),
                               0.9 * (temperature / 1000.0)**0.2)
    near_break = np.array([1000.0 - 1e-6, 1000.0 + 1e-6])  # [K]
    broken = evaluate_temperature_ratio(near_break, [np.log(0.9), 0.2], [[0.0, -0.4]])
    np.testing.assert_allclose(broken, 0.9, atol=1e-9)


def test_zero_temperature_error_likelihood_matches_normalized_gaussian():
    ratios = np.array([0.9, 1.0, 1.1])
    errors = np.full(3, 0.1)
    actual = temperature_ratio_log_likelihood([0.0, 0.0], np.empty((0, 2)),
                                               [500.0, 1000.0, 2000.0], ratios, errors)
    expected = np.sum(-0.5 * ((ratios - 1.0) / errors)**2 - np.log(errors * np.sqrt(2.0 * np.pi)))
    assert actual == pytest.approx(expected)


def test_real_pcat_sampler_changes_dimension_and_returns_catalog_samples():
    temperature = np.linspace(500.0, 3000.0, 12)  # [K]
    ratios = np.ones(12)
    result = infer_temperature_regimes(temperature, ratios, np.full(12, 0.2),
                                       number_chains=2, number_steps=1200, burn_steps=200,
                                       thinning=10, maximum_breaks=2, seed=5)

    assert result['base_samples'].shape == (200, 2)
    assert len(np.unique(result['component_counts'])) > 1
    assert result['break_count_probabilities'].sum() == pytest.approx(1.0)
    assert sum(chain['acceptance_counts']['birth'] for chain in result['chains']) > 0
    assert sum(chain['acceptance_counts']['death'] for chain in result['chains']) > 0
    assert all(catalog.shape == (count, 2) for count, catalog in zip(result['component_counts'], result['catalog_samples']))


def test_invalid_thermal_uncertainties_are_rejected():
    with pytest.raises(ValueError, match='positive'):
        infer_temperature_regimes([500.0, 1000.0, 2000.0], [1.0, 1.0, 1.0], [0.0, 0.1, 0.1])


@pytest.mark.parametrize('masked_input', [0, 1, 2, 3])
def test_thermal_likelihood_rejects_masked_observations(masked_input):
    observations = [
        np.array([500.0, 1000.0, 2000.0]),
        np.ones(3), np.full(3, 0.1), np.full(3, 20.0),
    ]
    observations[masked_input] = np.ma.array(
        observations[masked_input], mask=[False, True, False],
    )
    observations[masked_input].data[1] = 99.0
    with pytest.raises(ValueError, match='masked values'):
        temperature_ratio_log_likelihood([0.0, 0.0], np.empty((0, 2)), *observations)


def test_thermal_likelihood_preserves_unmasked_containers():
    observations = [
        np.array([500.0, 1000.0, 2000.0]),
        np.ones(3), np.full(3, 0.1), np.full(3, 20.0),
    ]
    expected = temperature_ratio_log_likelihood([0.0, 0.0], np.empty((0, 2)), *observations)
    actual = temperature_ratio_log_likelihood(
        [0.0, 0.0], np.empty((0, 2)),
        *(np.ma.array(values, mask=False) for values in observations),
    )
    assert actual == expected


def test_temperature_marginalization_matches_linear_gaussian_convolution():
    temperature = np.array([1000.0, 1200.0])  # [K]
    temperature_errors = np.array([50.0, 40.0])  # [K]
    ratios = np.array([1.12, 1.1])
    ratio_errors = np.full(2, 0.1)
    effective_variance = ratio_errors**2 + (temperature_errors / 1000.0)**2
    expected = np.sum(-0.5 * (ratios - temperature / 1000.0)**2 / effective_variance
                      - 0.5 * np.log(2.0 * np.pi * effective_variance))

    actual = temperature_ratio_log_likelihood([0.0, 1.0], np.empty((0, 2)), temperature,
                                               ratios, ratio_errors, temperature_errors)

    assert actual == pytest.approx(expected, abs=1e-6)


def test_inference_rejects_invalid_quadrature_setting():
    with pytest.raises(ValueError, match='quadrature_order'):
        infer_temperature_regimes([500.0, 1000.0, 2000.0], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1],
                                   quadrature_order=False)


def test_configurable_slope_prior_may_exclude_zero():
    result = infer_temperature_regimes(
        [500.0, 1000.0, 2000.0], [0.9, 1.0, 1.1], [0.1, 0.1, 0.1],
        slope_bounds=(0.1, 0.3), maximum_breaks=0, number_chains=1,
        number_steps=60, burn_steps=20, thinning=4,
    )
    assert np.all((result['base_samples'][:, 1] >= 0.1) & (result['base_samples'][:, 1] <= 0.3))