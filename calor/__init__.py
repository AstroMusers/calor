"""Thermal-emission inference for rocky exoplanets."""

from .inference import evaluate_temperature_ratio, infer_temperature_regimes, temperature_ratio_log_likelihood
from .paths import get_data_path, get_repository_path, get_visuals_path

__all__ = [
	"evaluate_temperature_ratio",
	"infer_temperature_regimes",
	"temperature_ratio_log_likelihood",
	"get_repository_path",
	"get_data_path",
	"get_visuals_path",
]