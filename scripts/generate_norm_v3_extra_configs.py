"""Genera las configs v3 de los experimentos que faltan por migrar:
comparativa U-Net, ablacion (5 grupos) y split espacial.

Estrategia: LEE las configs de referencia existentes y las parchea, para
reutilizar exactamente sus channel_indices / hiperparametros sin transcribir
nada a mano. A cada una le aplica:
  - data.norm_version: v3
  - training.masked_loss: true
  - training.loss: spatial_mse  (mse puro = spatial_mse con lambda_pearson=0;
    necesario porque la loss enmascarada solo existe en spatial_mse)
  - training.lambda_pearson: se conserva si existe, si no 0.0 (¡critico! el
    default de main.py es 0.5)
  - salida redirigida y aislada bajo results/norm_v3/<exp>/
  - experiment.name con sufijo _v3

NO cubre:
  - RF (invariante a la normalizacion afin -> mismos numeros, se declara)
  - Barrido lambda (ya corriendo en results/norm_v3/lambda/)
  - full de la ablacion (= lambda=0 del barrido, ya hecho)
  - Meteo (se decide aparte)

Uso:
    .venv\\Scripts\\python.exe scripts/generate_norm_v3_extra_configs.py
"""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(ROOT, 'configs')
OUT_DIR = os.path.join(CFG_DIR, 'norm_v3')
SEEDS = [42, 123, 7]
ABLATION_GROUPS = ['sin_sx', 'sin_pers', 'sin_topo5', 'sin_sce', 'sin_topo1']


def patch(cfg, name, results_dir, weights_dir, norm_version='v3'):
    """Aplica norm v3 + loss enmascarada + redireccion de salida.

    norm_version: 'v3' (temporal) o 'v3_spatial' (constantes del train
    espacial, sin fuga hacia la banda de test espacial).
    """
    cfg['data']['norm_version'] = norm_version
    tr = cfg['training']
    tr['masked_loss'] = True
    # mse -> spatial_mse conservando lambda (o 0.0 si no habia)
    if tr.get('loss') != 'spatial_mse':
        tr['loss'] = 'spatial_mse'
    tr['lambda_pearson'] = tr.get('lambda_pearson', 0.0)
    cfg['experiment'] = {'name': name}
    cfg['output'] = {
        'models_dir': weights_dir,
        'results_dir': f'{results_dir}/{name}',
        'model_name': name,
    }
    return cfg


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0

    # --- 1. U-Net GN (comparativa) -----------------------------------------
    for s in SEEDS:
        src = load(os.path.join(CFG_DIR, f'unet_gn_s{s}.yaml'))
        name = f'unet_gn_v3_s{s}'
        cfg = patch(src, name, 'results/norm_v3/comparison',
                    'results/norm_v3/comparison/weights')
        dump(cfg, os.path.join(OUT_DIR, f'{name}.yaml'))
        n += 1

    # --- 2. Ablacion (5 grupos x 3 seeds) ----------------------------------
    for g in ABLATION_GROUPS:
        for s in SEEDS:
            src = load(os.path.join(CFG_DIR,
                                    f'resunetpp_ablation_{g}_s{s}.yaml'))
            name = f'resunetpp_v3_abl_{g}_s{s}'
            cfg = patch(src, name, 'results/norm_v3/ablation',
                        'results/norm_v3/ablation/weights')
            dump(cfg, os.path.join(OUT_DIR, f'{name}.yaml'))
            n += 1

    # --- 3. Split espacial (ResUNet++ lambda=0.4 x 3 seeds) ----------------
    for s in SEEDS:
        src = load(os.path.join(CFG_DIR, f'resunetpp_spatial_s{s}.yaml'))
        name = f'resunetpp_v3_spatial_s{s}'
        cfg = patch(src, name, 'results/norm_v3/spatial',
                    'results/norm_v3/spatial/weights',
                    norm_version='v3_spatial')
        dump(cfg, os.path.join(OUT_DIR, f'{name}.yaml'))
        n += 1

    print(f'{n} configs v3 generadas en {OUT_DIR}')
    print('  U-Net GN     : 3')
    print('  Ablacion     : 15 (5 grupos x 3 seeds)')
    print('  Split espacial: 3')


if __name__ == '__main__':
    main()
