# calor

`calor` is a pipeline for characterizing and modeling thermal anomalies of rocky exoplanets based on their expected irradiation from their host stars. It simulates planetary internal processes, including residual heat from formation, tidal heating, and induction heating, in an attempt to explain the thermal emission excesses observed for some rocky exoplanets orbiting M dwarfs (or "M-Earths"). Details about the simulation methods and the science case can be found in [Lin & Daylan (2026)](https://arxiv.org/abs/2601.00412).

This repository includes two Jupyter Notebook scripts to reproduce the data analysis processes and figures presented in [Lin & Daylan (2026)](https://arxiv.org/abs/2601.00412), two main data files, and some auxiliary data files.
- `reproduce_figures.ipynb`, as its name suggests, reproduces figures in the paper (except for Fig. 3, 10, and 11).
- `analyze_R_vs_Tirr` analyzes the observational data of rocky exoplanets with dayside emission temperature measurements and generates Fig. 3, 10, and 11.
- `Data_all_planets.csv` contains exoplanet system data from both the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/index.html) and the [PlanetS catalog](https://dace.unige.ch/exoplanets/?), as well as parameters derived in [Lin & Daylan (2026)](https://arxiv.org/abs/2601.00412), such as the thermal emission potential $\Delta R$.
- `Data_observed_planets.csv` contains observed parameters for rocky exoplanets with dayside emission temperature characterizations by JWST or Spitzer.
- `Coy_et_al_2025_Table2.csv` and `Ito_et_al_2015_P_T.csv` contains relevant data from the literature. Check out [Coy et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025ApJ...987...22C/abstract), [Ito et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...801..144I/abstract), and our scripts for their physical meanings and how they are used.
- `Tirr_Roche_boundaries.csv` contains pre-computed $T_{\rm irr}$ boundaries corresponding to the Roche limit distances of exoplanets in our sample.
- The folder `MR_curves` contains exoplanet mass-radius curves computed using the `CORGI` planetary interior composition and structure code. For details about `CORGI`, check out [Lin et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025PSJ.....6...27L/abstract).
