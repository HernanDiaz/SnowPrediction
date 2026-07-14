"""Genera las 24 configs del barrido lambda con normalizacion v3.

ResUNet++ HPO (Optuna Trial 17), split temporal, 8 valores de lambda x 3
seeds. Identico al barrido de referencia salvo:
  - `data.norm_version: v3` (normalizacion empirica con exclusion de
    artefactos, ver data/dataset.py::NORM_V3 y
    scripts/compute_norm_v3_stats.py).
  - `training.masked_loss: true` (la loss solo ve pixeles con dato LiDAR;
    antes los nodata entraban como HS=0, contradiciendo el paper).

Salida: configs/norm_v3/resunetpp_v3_sp{tag}_s{seed}.yaml
Todo el experimento queda aislado en results/norm_v3/lambda/.

Uso:
    .venv\\Scripts\\python.exe scripts/generate_norm_v3_configs.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'configs', 'norm_v3')

LAMBDAS = {
    '0.0': '00', '0.1': '01', '0.25': '025', '0.4': '04',
    '0.5': '05', '0.6': '06', '0.75': '075', '1.0': '10',
}
SEEDS = [42, 123, 7]

TEMPLATE = """\
experiment:
  name: {name}

# Barrido lambda con NORMALIZACION V3 (empirica + exclusion de artefactos).
# ResUNet++ hiperparametros Optuna Trial 17, split temporal, 50 epocas.
# Identico a configs/resunetpp_v4_ms_sx200_hpo_sp*.yaml salvo norm_version.
# Aislado en results/norm_v3/lambda/ (no colisiona con otros experimentos).

data:
  root: "dataset_v4_ms_sx200"
  csv_file: "dataset_v4_ms_sx200.csv"
  images_dir: "images"
  masks_dir: "masks"
  source: "lidar"
  split_type: "temporal"
  use_sce: false
  augmentation: false
  norm_version: "v3"

model:
  architecture: "resunetpp"
  in_channels: 22
  out_channels: 1
  features: [64, 128, 256, 512]
  dropout_p: 0.077
  num_groups: 8

training:
  seed: {seed}
  batch_size: 8
  learning_rate: 1.287e-04
  epochs: 50
  loss: "spatial_mse"
  lambda_pearson: {lam}
  masked_loss: true
  weight_decay: 1.239e-04
  optimizer: "adamw"
  grad_clip: 1.0
  early_stopping: false
  num_workers: 0
  device: "auto"

output:
  models_dir: "results/norm_v3/lambda/weights"
  results_dir: "results/norm_v3/lambda/{name}"
  model_name: "{name}"
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for lam, tag in LAMBDAS.items():
        for seed in SEEDS:
            name = f'resunetpp_v3_sp{tag}_s{seed}'
            path = os.path.join(OUT_DIR, f'{name}.yaml')
            with open(path, 'w') as f:
                f.write(TEMPLATE.format(name=name, seed=seed, lam=lam))
            n += 1
    print(f'{n} configs generadas en {OUT_DIR}')


if __name__ == '__main__':
    main()
