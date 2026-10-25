"""Transdimensional inference of changes in the R versus irradiation-temperature trend."""

import numpy as np
from scipy.special import logsumexp


REFERENCE_TEMPERATURE_K = 1000.0  # [K]


def evaluate_temperature_ratio(temperature_kelvin, base_parameters, break_catalog):
    """Evaluate a continuous broken power law with a variable number of slope changes.

    Base entries are the initial-segment log normalization and slope. Catalog
    rows hold log(T_break / 1000 K) and slope change. Breaks describe empirical
    temperature regimes, not uniquely identified heating or atmospheric processes.
    """

    temperature = np.asarray(temperature_kelvin, dtype=float)
    base = np.asarray(base_parameters, dtype=float)
    catalog = np.asarray(break_catalog, dtype=float)
    if (not np.isfinite(temperature).all() or np.any(temperature <= 0)
            or base.shape != (2,) or not np.isfinite(base).all()
            or catalog.ndim != 2 or catalog.shape[1] != 2 or not np.isfinite(catalog).all()):
        raise ValueError('Provide positive finite temperatures, two base entries, and a (K, 2) break catalog')
    log_temperature = np.log(temperature / REFERENCE_TEMPERATURE_K)
    log_ratio = base[0] + base[1] * log_temperature
    if len(catalog):
        log_ratio += np.sum(np.maximum(log_temperature[..., None] - catalog[:, 0], 0.0) * catalog[:, 1], axis=-1)
    with np.errstate(over='ignore'):
        return np.exp(log_ratio)


def _observations(temperature_kelvin, ratios, ratio_uncertainties, temperature_uncertainties):
    if any(np.ma.is_masked(value) for value in (
        temperature_kelvin, ratios, ratio_uncertainties, temperature_uncertainties,
    )):
        raise ValueError('Observation arrays must not contain masked values')
    temperature, ratios, errors = (np.asarray(value, dtype=float)
                                  for value in (temperature_kelvin, ratios, ratio_uncertainties))
    if any(value.ndim != 1 or value.size == 0 or not np.isfinite(value).all()
           for value in (temperature, ratios, errors)):
        raise ValueError('Observation arrays must be nonempty finite vectors')
    if temperature.shape != ratios.shape or temperature.shape != errors.shape:
        raise ValueError('Observation arrays must have identical lengths')
    if np.any(temperature <= 0) or np.any(ratios <= 0) or np.any(errors <= 0):
        raise ValueError('Temperatures, ratios, and ratio uncertainties must be positive')
    temperature_errors = np.zeros_like(temperature) if temperature_uncertainties is None else np.asarray(temperature_uncertainties, dtype=float)
    if (temperature_errors.shape != temperature.shape or not np.isfinite(temperature_errors).all()
            or np.any(temperature_errors < 0)):
        raise ValueError('Temperature uncertainties must be matching nonnegative finite values')
    return temperature, ratios, errors, temperature_errors


def _temperature_quadrature(temperature, uncertainty, order):
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError('quadrature_order must be a positive integer')
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    latent_temperature = temperature[:, None] + np.sqrt(2.0) * uncertainty[:, None] * nodes
    positive = latent_temperature > 0.0
    log_weights = np.where(positive, np.log(weights / np.sqrt(np.pi)), -np.inf)
    log_weights -= logsumexp(log_weights, axis=1, keepdims=True)
    return np.maximum(latent_temperature, np.finfo(float).tiny), log_weights


def _marginal_likelihood(base, catalog, latent_temperature, log_weights, ratios, errors):
    predicted = evaluate_temperature_ratio(latent_temperature, base, catalog)
    with np.errstate(over='ignore', invalid='ignore'):
        residual = (ratios[:, None] - predicted) / errors[:, None]
        log_density = -0.5 * residual**2 - np.log(errors[:, None] * np.sqrt(2.0 * np.pi))
    return float(np.sum(logsumexp(log_density + log_weights, axis=1)))


def temperature_ratio_log_likelihood(
    base_parameters, break_catalog, temperature_kelvin, ratios, ratio_uncertainties,
    temperature_uncertainties=None, quadrature_order=9,
):
    """Marginalize independent Gaussian temperature errors in a Gaussian ratio likelihood.

    This assumes independent measurements and no supplied temperature-ratio
    covariance. Alternate reductions of the same observation should not be
    passed as independent targets without an explicit correlation model.
    """

    temperature, ratios, errors, temperature_errors = _observations(
        temperature_kelvin, ratios, ratio_uncertainties, temperature_uncertainties,
    )
    latent_temperature, weights = _temperature_quadrature(temperature, temperature_errors, quadrature_order)
    return _marginal_likelihood(base_parameters, break_catalog, latent_temperature, weights, ratios, errors)


def infer_temperature_regimes(
    temperature_kelvin, ratios, ratio_uncertainties, temperature_uncertainties=None,
    *, maximum_breaks=3, count_prior_mean=1.0, number_chains=4,
    number_steps=12000, burn_steps=4000, thinning=10, seed=42,
    normalization_bounds=(0.2, 2.0), slope_bounds=(-2.0, 2.0),
    slope_change_bounds=(-2.0, 2.0), break_temperature_bounds=None,
    quadrature_order=9,
):
    """Sample breakpoint count, temperatures, and slopes using PCAT birth/death moves.

    The normalized count prior is truncated Poisson. Normalization is log-uniform,
    slopes are uniform, and break temperatures are log-uniform over the supplied
    interval (the observed temperature range by default). Every catalog sample
    has its own dimension. Returned draws are not a convergence certification.
    """

    from pcat.sampling import sample_catalog

    temperature, ratios, errors, temperature_errors = _observations(
        temperature_kelvin, ratios, ratio_uncertainties, temperature_uncertainties,
    )
    if (isinstance(number_chains, (bool, np.bool_))
            or not isinstance(number_chains, (int, np.integer)) or number_chains < 1):
        raise ValueError('number_chains must be a positive integer')
    if temperature.size < 3:
        raise ValueError('At least three observations are required for trend inference')
    if not 0 < normalization_bounds[0] < normalization_bounds[1]:
        raise ValueError('Normalization bounds must be positive and increasing')
    if break_temperature_bounds is None:
        break_temperature_bounds = (temperature.min(), temperature.max())  # [K]
    if not 0 < break_temperature_bounds[0] < break_temperature_bounds[1]:
        raise ValueError('Break-temperature bounds must be positive and increasing')
    latent_temperature, weights = _temperature_quadrature(temperature, temperature_errors, quadrature_order)

    def likelihood(base, catalog):
        return _marginal_likelihood(base, catalog, latent_temperature, weights, ratios, errors)

    base_bounds = [np.log(normalization_bounds), slope_bounds]
    component_bounds = [np.log(np.asarray(break_temperature_bounds) / REFERENCE_TEMPERATURE_K), slope_change_bounds]
    child_seeds = np.random.SeedSequence(seed).spawn(number_chains)
    chains = [sample_catalog(
        likelihood, base_bounds, component_bounds, maximum_components=maximum_breaks,
        count_prior_mean=count_prior_mean, number_steps=number_steps, burn_steps=burn_steps,
        thinning=thinning, seed=child_seed,
        initial_base=[np.log(np.clip(np.median(ratios), *normalization_bounds)), np.clip(0.0, *slope_bounds)],
    ) for child_seed in child_seeds]
    counts = np.concatenate([chain['component_counts'] for chain in chains])
    probabilities = np.bincount(counts, minlength=maximum_breaks + 1) / counts.size
    return {
        'chains': chains, 'component_counts': counts,
        'break_count_probabilities': probabilities,
        'base_samples': np.concatenate([chain['base_samples'] for chain in chains]),
        'catalog_samples': [catalog for chain in chains for catalog in chain['catalog_samples']],
        'break_temperature_bounds': tuple(break_temperature_bounds),
        'count_prior_mean': count_prior_mean,
    }