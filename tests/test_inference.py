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


@pytest.mark.parametrize('masked_input', [0, 1, 2])
@pytest.mark.parametrize('fully_masked', [False, True])
def test_temperature_ratio_rejects_masked_inputs(masked_input, fully_masked):
    inputs = [np.array([500.0, 1000.0]), np.array([0.0, 0.2]), np.array([[0.0, -0.4]])]
    masked = np.ma.array(inputs[masked_input], mask=np.full_like(inputs[masked_input], fully_masked, dtype=bool))
    masked.mask.flat[0] = True
    inputs[masked_input] = masked

    with pytest.raises(ValueError, match='masked'):
        evaluate_temperature_ratio(*inputs)


@pytest.mark.parametrize('masked_input', ['base', 'catalog'])
def test_thermal_likelihood_rejects_masked_model_parameters(masked_input):
    base = np.array([0.0, 0.2])
    catalog = np.array([[0.0, -0.4]])
    if masked_input == 'base':
        base = np.ma.array(base, mask=[True, False])
    else:
        catalog = np.ma.array(catalog, mask=[[True, False]])

    with pytest.raises(ValueError, match='masked'):
        temperature_ratio_log_likelihood(base, catalog, [500.0, 1000.0], [1.0, 1.0], [0.1, 0.1])


def test_temperature_ratio_preserves_unmasked_containers():
    inputs = [np.array([500.0, 1000.0]), np.array([0.0, 0.2]), np.array([[0.0, -0.4]])]
    expected = evaluate_temperature_ratio(*inputs)
    actual = evaluate_temperature_ratio(*(np.ma.array(values, mask=False) for values in inputs))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('input_index', [0, 1, 2])
@pytest.mark.parametrize('imaginary_part', [0.0, 1.0])
def test_temperature_ratio_rejects_complex_inputs(input_index, imaginary_part):
    inputs = [np.array([500.0, 1000.0]), np.array([0.0, 0.2]), np.array([[0.0, -0.4]])]
    inputs[input_index] = inputs[input_index].astype(complex) + imaginary_part * 1j
    with pytest.raises(ValueError, match='real'):
        evaluate_temperature_ratio(*inputs)


@pytest.mark.parametrize('input_index', [0, 1, 2, 3])
@pytest.mark.parametrize('imaginary_part', [0.0, 1.0])
def test_thermal_likelihood_rejects_complex_observations(input_index, imaginary_part):
    observations = [np.array([500.0, 1000.0, 2000.0]), np.ones(3), np.full(3, 0.1), np.full(3, 20.0)]
    observations[input_index] = observations[input_index].astype(complex) + imaginary_part * 1j
    with pytest.raises(ValueError, match='real'):
        temperature_ratio_log_likelihood([0.0, 0.0], np.empty((0, 2)), *observations)


@pytest.mark.parametrize('input_index', [0, 1, 2, 3])
def test_inference_rejects_complex_observations_before_sampling(monkeypatch, input_index):
    from unittest.mock import Mock
    import pcat.sampling

    sampler = Mock(side_effect=AssertionError('Invalid data reached the sampler'))
    monkeypatch.setattr(pcat.sampling, 'sample_catalog', sampler)
    observations = [np.array([500.0, 1000.0, 2000.0]), np.ones(3), np.full(3, 0.1), np.full(3, 20.0)]
    observations[input_index] = observations[input_index].astype(complex) + 1j
    with pytest.raises(ValueError, match='real'):
        infer_temperature_regimes(*observations)
    sampler.assert_not_called()


@pytest.mark.parametrize('input_name', ['base', 'catalog'])
def test_thermal_likelihood_rejects_complex_model_parameters(input_name):
    base = np.array([0.0, 0.2])
    catalog = np.array([[0.0, -0.4]])
    if input_name == 'base':
        base = base.astype(complex) + 1j
    else:
        catalog = catalog.astype(complex) + 1j
    with pytest.raises(ValueError, match='real'):
        temperature_ratio_log_likelihood(base, catalog, [500.0, 1000.0], [1.0, 1.0], [0.1, 0.1])


def test_zero_temperature_error_likelihood_matches_normalized_gaussian():
    ratios = np.array([0.9, 1.0, 1.1])
    errors = np.full(3, 0.1)
    actual = temperature_ratio_log_likelihood([0.0, 0.0], np.empty((0, 2)),
                                               [500.0, 1000.0, 2000.0], ratios, errors)
    expected = np.sum(-0.5 * ((ratios - 1.0) / errors)**2 - np.log(errors * np.sqrt(2.0 * np.pi)))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('scale', [0.1, np.finfo(float).smallest_subnormal, 1e-300, 1e307, 1e308])
@pytest.mark.parametrize('quadrature_order', [1, 9])
def test_thermal_likelihood_preserves_finite_gaussian_normalization(scale, quadrature_order):
    from scipy.stats import norm

    with np.errstate(all='raise'):
        actual = temperature_ratio_log_likelihood(
            [0.0, 0.0], np.empty((0, 2)), [1000.0], [1.0], [scale],
            [20.0], quadrature_order=quadrature_order,
        )
        expected = norm.logpdf(1.0, loc=1.0, scale=scale)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('ratio', [1.4e154, 1.6e154])
@pytest.mark.parametrize('quadrature_order', [1, 9])
def test_thermal_likelihood_preserves_representable_gaussian_tails(ratio, quadrature_order):
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 100
        residual = Decimal.from_float(ratio) - Decimal(1)
        expected = float(-Decimal('0.5') * residual * residual)
    with np.errstate(all='raise'):
        actual = temperature_ratio_log_likelihood(
            [0.0, 0.0], np.empty((0, 2)), [1000.0], [ratio], [1.0], quadrature_order=quadrature_order,
        )

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=5e-16)


@pytest.mark.parametrize('quadrature_order', [1, 9])
def test_thermal_likelihood_returns_negative_infinity_for_unrepresentable_tail(quadrature_order):
    with np.errstate(all='raise'):
        actual = temperature_ratio_log_likelihood(
            [0.0, 0.0], np.empty((0, 2)), [1000.0], [np.finfo(float).max], [1.0],
            quadrature_order=quadrature_order,
        )
    assert np.isneginf(actual)


def test_thermal_likelihood_matches_original_formula_for_typical_inputs():
    from scipy.special import logsumexp

    generator = np.random.default_rng(42)
    temperature = np.array([600.0, 1100.0, 2200.0])  # [K]
    temperature_errors = np.array([20.0, 30.0, 40.0])  # [K]
    errors = np.array([0.05, 0.1, 0.2])
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    latent = temperature[:, None] + np.sqrt(2.0) * temperature_errors[:, None] * nodes
    log_temperature = np.log(latent / 1000.0)
    for _ in range(20):
        base = generator.uniform([-0.2, -0.5], [0.2, 0.5])
        catalog = np.array([[np.log(1.7), generator.uniform(-0.5, 0.5)]])
        ratios = generator.uniform(0.7, 1.3, size=temperature.size)
        prediction = np.exp(base[0] + base[1] * log_temperature
                            + np.maximum(log_temperature - catalog[0, 0], 0.0) * catalog[0, 1])
        density = -0.5 * ((ratios[:, None] - prediction) / errors[:, None])**2
        density -= np.log(errors[:, None] * np.sqrt(2.0 * np.pi))
        expected = np.sum(logsumexp(density + np.log(weights / np.sqrt(np.pi)), axis=1))
        actual = temperature_ratio_log_likelihood(base, catalog, temperature, ratios, errors, temperature_errors)
        assert actual == pytest.approx(expected, abs=1e-12)


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