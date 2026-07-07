# Cover letter

> TEMPLATE — fill the bracketed fields before submission.

[Date]

Dear Editors of *Computers & Geosciences*,

We are pleased to submit our manuscript entitled **"Spatially-aware deep
learning for high-resolution snow depth mapping in a Pyrenean catchment:
evaluating pattern fidelity with the SPAEF metric"** for consideration as a
research article in *Computers & Geosciences*.

High-resolution snow-depth mapping is central to water-resource management,
avalanche forecasting and mountain ecology, yet the spatial *pattern* fidelity
of model predictions is rarely evaluated. Our work addresses this gap and makes
the following contributions:

- **Pattern-aware evaluation.** We show that conventional pixel-wise metrics
  (e.g. R²) can hide severe spatial errors and seed-to-seed instability, and we
  adopt the Spatial Efficiency metric (SPAEF) and a multi-scale variant
  (MSPAEF) as complementary criteria for snow mapping.
- **A hybrid spatial loss.** We introduce a loss that blends pixel-wise MSE with
  a spatial-correlation term, and we systematically sweep its weight λ, showing
  that a moderate value (λ = 0.4) improves both accuracy and pattern fidelity
  while roughly halving the variance across random seeds.
- **A reproducible 1 m benchmark.** Using a 27-date airborne LiDAR time series
  over the Izas experimental catchment (Spanish Pyrenees) and a strictly
  temporal train/validation/test split, we benchmark a Random Forest, a U-Net
  and a ResUNet++, and we release the code, configurations and trained weights.
- **Predictor analysis.** A leave-one-group-out ablation identifies meso-scale
  (5 m) topography and recent snow persistence as the dominant predictors,
  whereas instantaneous satellite snow-cover extent and scalar meteorological
  forcing add little.

The study fits the scope of *Computers & Geosciences* through its emphasis on
reproducible geoscientific computing: open code and data, a transparent
multi-seed evaluation protocol, and a metric/loss methodology transferable to
other spatial-prediction problems in the Earth sciences.

This manuscript is original, has not been published previously, and is not under
consideration elsewhere. All authors have approved the submission and declare no
competing interests. The LiDAR and meteorological data were provided through a
collaboration with the Instituto Pirenaico de Ecología (IPE-CSIC).

We suggest the following potential reviewers with expertise in snow remote
sensing and deep learning for the geosciences: [Reviewer 1], [Reviewer 2],
[Reviewer 3].

Thank you for considering our manuscript.

Sincerely,

[Corresponding Author], on behalf of all authors
[Affiliation]
[E-mail]
