# calor

How hot should a rocky exoplanet be, given the starlight it receives? We use Calor to compare that expectation with dayside emission measurements and explore how residual formation heat, tides, and induction heating could contribute to thermal excesses in planets orbiting M dwarfs, or "M-Earths." A high temperature gets our attention. Establishing which process supplies the heat takes more than a temperature alone. We describe the calculations and science case in [Lin & Daylan (2026)](https://arxiv.org/abs/2601.00412).

The two Jupyter notebooks and their input data reproduce the analysis and figures from our paper.

Set `CALOR_PATH` to the checkout directory when using Calor from another location. If it is unset, the package uses its checkout root. Local inference products belong in the ignored `data/` directory. Example figures are saved directly in the ignored `visuals/` directory.
- `reproduce_figures.ipynb` reproduces figures from our paper, except for Fig. 3, 10, and 11.
- `analyze_R_vs_Tirr` analyzes the observational data of rocky exoplanets with dayside emission temperature measurements and generates Fig. 3, 10, and 11.
- `Data_all_planets.csv` contains exoplanet system data from both the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/index.html) and the [PlanetS catalog](https://dace.unige.ch/exoplanets/?), as well as parameters derived in our study [Lin & Daylan (2026)](https://arxiv.org/abs/2601.00412), such as the thermal emission potential $\Delta R$.
- `Data_observed_planets.csv` contains observed parameters for rocky exoplanets with dayside emission temperature characterizations by JWST or Spitzer.
- `Coy_et_al_2025_Table2.csv` and `Ito_et_al_2015_P_T.csv` contains relevant data from the literature. Check out [Coy et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025ApJ...987...22C/abstract), [Ito et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...801..144I/abstract), and our scripts for their physical meanings and how they are used.
- `Tirr_Roche_boundaries.csv` contains pre-computed $T_{\rm irr}$ boundaries corresponding to the Roche limit distances of exoplanets in our sample.
- The folder `MR_curves` contains exoplanet mass-radius curves computed using the `CORGI` planetary interior composition and structure code. For details about `CORGI`, check out [Lin et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025PSJ.....6...27L/abstract).

## PCAT thermal-regime inference

We also use PCAT to ask how many changes in slope the temperature-ratio data support, instead of fixing that number in advance. `calor.infer_temperature_regimes` samples a continuous broken power law for $\mathcal R=T_{\rm day}/T_{\rm max}$ as a function of irradiation temperature. Each breakpoint contributes its log temperature and a slope change. PCAT samples their number, locations, and slopes using birth, death, and within-dimension moves. With zero breakpoints the trend is a single power law. These empirical regimes do not uniquely identify residual heat, tides, induction heating, or atmospheric redistribution.

The default count prior is Poisson with mean 1, normalized over zero through three breakpoints. The initial-segment normalization is log-uniform over 0.2 to 2. Initial slopes and slope changes have uniform priors from -2 to 2. Break temperatures are log-uniform over the supplied interval, defaulting to the observed temperature range. These priors are configurable and should be checked for the intended comparison.

The Gaussian likelihood marginalizes supplied irradiation-temperature errors with Gaussian quadrature. Ratio errors can use the existing notebook's symmetric approximation $(\sigma_+-\sigma_-)/2$. The calculation treats temperature and ratio errors as independent and does not include covariance from shared stellar parameters. The observational notebook section makes its one-estimate-per-planet selection explicit. Alternate reductions and measurements in different bands should not automatically be treated as independent planets.

Install Calor and its local source dependencies from the Calor repository root:

```bash
python -m pip install -e . -e /path/to/pcat -e /path/to/tdpy -e /path/to/chalcedon
```

Replace each `/path/to/...` value with the location of that package's checkout.

From Python, supply the measured temperatures, ratios, and their uncertainties:

```python
from calor import infer_temperature_regimes

posterior = infer_temperature_regimes(
	temperature_kelvin, measured_ratios, ratio_uncertainties,
	temperature_uncertainties,
	maximum_breaks=3, number_chains=4, seed=42,
)
```

The result retains every variable-sized breakpoint catalog, base parameters, count probabilities, full count traces, log posterior values, and move acceptance counts. Compare independent chains, inspect count transitions, test quadrature convergence, and check prior sensitivity before interpreting the posterior. A short smoke run confirms that the workflow executes. Use longer runs to assess convergence and evidence uncertainty.

The PCAT analysis estimates a variable-break model from the observational data in `analyze_R_vs_Tirr.ipynb`. The paper's published figures retain the original fixed-fit analysis. The standalone synthetic example injects a slope change at 1,700 K with an initial slope of 0.15, a slope change of -0.75, and independent ratio noise of 0.025:

```bash
python examples/temperature_regimes.py --steps 12000 --chains 4
```

The standalone example uses a white plot background with black axes. Add
`--typeplotback dark` for a black background with white axes. Its
`plot_result()` function accepts the same `typeplotback` option from Python
and otherwise inherits an enclosing TDpy `plot_style()` theme.

With the optional Bokeh extra installed, the CLI also writes an interactive HTML companion under `visuals/`. Install it with `python -m pip install -e ".[bokeh]"`.

The example writes posterior catalogs under `data/temperature_regimes/` and a full-temperature-range figure directly under `visuals/`. Pass `--figure-path` to save the figure elsewhere. All values in these outputs come from the synthetic example.
