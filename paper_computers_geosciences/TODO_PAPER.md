# TODO — Cerrar el paper (Computers & Geosciences)

Estado tras la sesión autónoma del 14/06/2026: las 8 figuras están hechas y el
manuscrito compila a 26 páginas sin referencias indefinidas. Quedan solo las
acciones que requieren información del usuario (nombres reales, datos externos).

## Figuras (todas hechas)
- [x] **Fig 5 — mapas de predicción** (GT | RF | UNet | ResUNet++).
      `figscripts/fig05_prediction_maps.py`. RF v6 (22ch) reentrenado y cacheado
      en `results/rf_v6_s42/rf_v6_s42.joblib`. Tile representativo
      `20250327_lidar_tile_1024_2048`.
- [x] **Fig 6 — barrido de λ** (`fig06_lambda_sweep.py`). Óptimo λ=0.4.
- [x] **Fig 7 — ablación de canales** (`fig07_ablation.py`).
- [x] **Fig 8 — experimento meteo** (`fig08_meteo.py`). Incluida como resultado
      negativo (22ch ≈ 26ch, meteo con más varianza).

## Texto del paper
- [x] Referencias `\ref{fig:..}` revisadas: todas las figuras se citan en texto.
- [x] Captions corregidos para describir las figuras reales (fig05/06/07/08).
- [x] Highlights añadidos (requisito C&G) + abstract revisado.
- [x] Coherencia numérica texto↔JSON verificada. Único arreglo: UNet SPAEF std
      ±0.216 → ±0.224 (ddof=1) en abstract, resultados y conclusiones.
- [ ] **Rellenar autores reales, afiliación inst1, email del corresponding y
      financiación** (placeholders marcados con `TODO` en `main.tex`).
      *No se rellena automáticamente: requiere datos reales del usuario.*

## Datos / reproducibilidad
- [x] `paper_computers_geosciences/REPRODUCIBILITY.md`: mapea cada figura/tabla a
      su script/config/datos. Puntero añadido al `README.md` raíz (que estaba
      desactualizado, describe experimentos antiguos 5m/17ch).
- [ ] (EXCLUIDO por el usuario) Solicitar datos meteo a Jesús Revuelto.
- [ ] (EXCLUIDO por el usuario) Subir dataset y pesos a Zenodo/HuggingFace y
      añadir el DOI en *Data availability*.

## Envío
- [x] CRediT authorship contribution statement añadido a `main.tex`
      (con nombres placeholder, marcados con `TODO`).
- [x] Cover letter borrador en `paper_computers_geosciences/cover_letter.md`.
- [ ] Revisión final por el usuario (nombres, financiación, revisores sugeridos)
      antes de enviar.
