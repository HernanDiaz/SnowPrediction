import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SnowDataset(Dataset):
    """
    Dataset de tiles de nieve para entrenamiento y validacion.

    Canales del .npy (dataset v5, 6 canales):
        [0] DEM        - Modelo Digital de Elevacion (metros)
        [1] Slope      - Pendiente (grados)
        [2] Northness  - cos(aspect), orientacion norte [-1, 1]
        [3] Eastness   - sin(aspect), orientacion este  [-1, 1]
        [4] TPI        - Topographic Position Index
        [5] SCE        - Snow Cover Extent satelite (opcional)

    Canales adicionales del .npy (dataset v6, 17 canales):
        [6-13] Sx_DIR_100  - Wind Shelter Index, 8 direcciones, radio 100m
        [14]   Pers_15d    - Fraccion de dias nevados en los ultimos 15 dias
        [15]   Pers_30d    - Fraccion de dias nevados en los ultimos 30 dias
        [16]   Pers_60d    - Fraccion de dias nevados en los ultimos 60 dias
    """

    NORM = {
        'dem_mean':  2100.0,
        'dem_std':   1000.0,
        'slope_max':   90.0,
        'tpi_max':   9200.0,   # Rango real observado: ~[-9155, +9141]
        'sx_max':      90.0,   # Angulo maximo teorico del horizonte
    }

    def __init__(self, dataframe: pd.DataFrame, images_dir: str,
                 masks_dir: str, use_sce: bool = False,
                 augment: bool = False, n_channels: int = None):
        """
        Args:
            dataframe:   DataFrame con metadatos de tiles (tile_id, ...)
            images_dir:  Directorio con los .npy de imagenes
            masks_dir:   Directorio con los .npy de mascaras (snow depth)
            use_sce:     Si True incluye SCE (canal 5). Ignorado si n_channels>5.
            augment:     Si True aplica data augmentation en tiempo real
            n_channels:  Numero de canales a cargar. Si None se deduce de use_sce.
                         Usar 17 para dataset v6 (Sx + persistencia).
        """
        self.df           = dataframe.reset_index(drop=True)
        self.images_dir   = images_dir
        self.masks_dir    = masks_dir
        self.use_sce      = use_sce
        self.augment      = augment
        self.augment_mode = 'hv'   # se sobreescribe desde main.py si es necesario

        if n_channels is not None:
            self.n_channels = n_channels
        else:
            self.n_channels = 6 if use_sce else 5

        # Indices de canales a cargar. Si None, carga los primeros n_channels.
        # Para dataset v6 (33 canales con huecos) usar [0..13, 30..32].
        self.channel_indices = None   # se puede sobreescribir desde main.py

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tile_id  = self.df.iloc[idx]['tile_id']
        img_path = os.path.join(self.images_dir, tile_id)
        msk_path = os.path.join(self.masks_dir,  tile_id)

        try:
            image = np.load(img_path).astype(np.float32)
            mask  = np.load(msk_path).astype(np.float32)
        except Exception as e:
            print(f"Error cargando {tile_id}: {e}")
            return (torch.zeros((self.n_channels, 256, 256)),
                    torch.zeros((1, 256, 256)))

        if self.channel_indices is not None:
            image = image[self.channel_indices, :, :]
        else:
            image = image[:self.n_channels, :, :]
        image, mask = self._clean(image, mask)
        image = self._normalize(image)

        if self.augment:
            image, mask = self._augment(image, mask)

        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy()).unsqueeze(0)

    # ------------------------------------------------------------------
    # Metodos privados
    # ------------------------------------------------------------------
    def _clean(self, image: np.ndarray, mask: np.ndarray):
        image[image == -9999] = 0
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        mask[mask <= -100] = 0
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        return image, mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        # Canales base (presentes en todos los datasets)
        image[0] = (image[0] - self.NORM['dem_mean']) / self.NORM['dem_std']
        image[1] = image[1] / self.NORM['slope_max']
        # Canal 2 (northness) y 3 (eastness) ya estan en [-1, 1]
        # TPI: rango real ~[-9000, +9000] -> normalizamos con su rango real
        image[4] = np.clip(image[4] / self.NORM['tpi_max'], -1.0, 1.0)

        # Canal 5 (SCE): codigos 0/10/11 -> binario [0, 1]
        if self.n_channels >= 6:
            image[5] = (image[5] > 5).astype(np.float32)

        # Canales 6-13 (Sx): angulos de horizonte en grados -> [-1, 1]
        if self.n_channels >= 14:
            image[6:14] = np.clip(image[6:14] / self.NORM['sx_max'], -1.0, 1.0)

        # Canales 14-16 (persistencia nival): ya en [0, 1], no requieren normalizacion

        return image

    def _augment(self, image: np.ndarray,
                 mask: np.ndarray) -> tuple:
        """
        Aplica flips aleatorios en tiempo real.

        Modos controlados por self.augment_mode:
          'hv'  - Flip H y V independientes (prob 0.5 c/u) -> 4 variantes
                  ATENCION: el flip V invierte la asimetria N-S del snowpack
                  (laderas N tienen mas nieve que S en los Pirineos).
                  Puede introducir patrones fisicamente incorrectos.
          'h'   - Solo flip horizontal (prob 0.5) -> 2 variantes
                  Conserva la asimetria N-S. Recomendado para cuencas con
                  gradiente N-S dominante en acumulacion de nieve.

        Correcciones de canal:
          Flip H: negar Eastness (ch 3)  — el eje E-O se invierte
          Flip V: negar Northness (ch 2) — el eje N-S se invierte
        """
        flip_h = random.random() > 0.5
        flip_v = (self.augment_mode == 'hv') and (random.random() > 0.5)

        if flip_h:
            image = np.flip(image, axis=2)   # invertir W
            mask  = np.flip(mask,  axis=1)
            image[3] = -image[3]             # Eastness -> negar

        if flip_v:
            image = np.flip(image, axis=1)   # invertir H
            mask  = np.flip(mask,  axis=0)
            image[2] = -image[2]             # Northness -> negar

        return image, mask


class SnowDatasetEval(SnowDataset):
    """
    Igual que SnowDataset pero devuelve tambien el tile_id,
    necesario para el bucle de evaluacion.
    """

    def __getitem__(self, idx):
        tile_id  = self.df.iloc[idx]['tile_id']
        img_path = os.path.join(self.images_dir, tile_id)
        msk_path = os.path.join(self.masks_dir,  tile_id)

        try:
            image = np.load(img_path).astype(np.float32)
            mask  = np.load(msk_path).astype(np.float32)
        except Exception as e:
            print(f"Error cargando {tile_id}: {e}")
            return (torch.zeros((self.n_channels, 256, 256)),
                    torch.zeros((1, 256, 256)),
                    tile_id)

        if self.channel_indices is not None:
            image = image[self.channel_indices, :, :]
        else:
            image = image[:self.n_channels, :, :]
        image, mask = self._clean(image, mask)
        image = self._normalize(image)

        return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0), tile_id


# ----------------------------------------------------------------------
# Funcion de utilidad para cargar los splits del CSV
# ----------------------------------------------------------------------
def load_splits(csv_path: str,
                source: str = 'lidar',
                split_type: str = 'temporal'):
    """
    Carga el CSV de metadatos y devuelve los tres splits.

    Args:
        csv_path:   Ruta al CSV generado por los notebooks a0x
        source:     'lidar' | 'pleiades' | 'grass' | 'all'
        split_type: 'temporal' (por anyo) | 'spatial' (por posicion)

    Returns:
        Tupla (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)

    if source != 'all':
        df = df[df['source'] == source].reset_index(drop=True)

    if split_type == 'temporal':
        col = 'exp_temporal_split'
    elif split_type == 'spatial':
        col = 'exp_spatial_split'
    else:
        raise ValueError(f"split_type desconocido: '{split_type}'. "
                         f"Usa 'temporal' o 'spatial'.")

    train_df = df[df[col] == 'train'].reset_index(drop=True)
    val_df   = df[df[col] == 'val'].reset_index(drop=True)
    test_df  = df[df[col] == 'test'].reset_index(drop=True)

    print(f"\nSplit '{split_type}' | Fuente: '{source}'")
    print(f"  Train : {len(train_df):>5} tiles  "
          f"({train_df['year'].unique().tolist() if len(train_df) else []})")
    print(f"  Val   : {len(val_df):>5} tiles  "
          f"({val_df['year'].unique().tolist() if len(val_df) else []})")
    print(f"  Test  : {len(test_df):>5} tiles  "
          f"({test_df['year'].unique().tolist() if len(test_df) else []})")

    return train_df, val_df, test_df
