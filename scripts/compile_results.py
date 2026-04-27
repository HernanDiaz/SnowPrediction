"""
compile_results.py
==================
Compila todos los _metrics.json de la carpeta results/ en una tabla
CSV ordenada por R2 descendente.

Uso:
    python compile_results.py
    python compile_results.py --results_dir results --output results/resultados_todos.csv
"""

import os
import json
import argparse
import datetime
import pandas as pd


def compile_results_table(results_root: str, save_path: str = None) -> pd.DataFrame:
    rows = []
    for dirpath, _, filenames in os.walk(results_root):
        for fname in filenames:
            if fname.endswith('_metrics.json'):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    rows.append(data)
                except Exception as e:
                    print(f"  [WARN] No se pudo leer {fpath}: {e}")

    if not rows:
        print("No se encontraron archivos _metrics.json.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Columnas de interes (en orden)
    cols_order = ['experiment', 'R2', 'MAE', 'RMSE', 'NSE', 'Bias',
                  'n_pixels', 'tta', 'timestamp']
    cols_present = [c for c in cols_order if c in df.columns]
    df = df[cols_present]

    # Ordenar por R2 descendente
    if 'R2' in df.columns:
        df = df.sort_values('R2', ascending=False).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"\nTabla resumen guardada en: {save_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Compilar resultados de experimentos')
    parser.add_argument('--results_dir', default='results',
                        help='Carpeta raiz de resultados (default: results)')
    parser.add_argument('--output', default='results/resultados_todos.csv',
                        help='Ruta CSV de salida (default: results/resultados_todos.csv)')
    args = parser.parse_args()

    print(f"\nBuscando _metrics.json en: {args.results_dir}")
    print("=" * 60)

    df = compile_results_table(args.results_dir, save_path=args.output)

    if not df.empty:
        print("\n--- TABLA DE RESULTADOS (ordenada por R2) ---")
        display_cols = [c for c in ['experiment', 'R2', 'MAE', 'RMSE', 'NSE', 'Bias'] if c in df.columns]
        print(df[display_cols].to_string(index=True))

        best = df.iloc[0]
        print(f"\nMejor modelo: {best.get('experiment','?')}  "
              f"R2={best.get('R2','?')}  MAE={best.get('MAE','?')} m")
    print()


if __name__ == '__main__':
    main()
