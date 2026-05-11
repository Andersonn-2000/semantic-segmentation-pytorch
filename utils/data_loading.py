import os
import logging
import torch
import numpy as np

from PIL import Image

from functools import lru_cache, partial
from itertools import repeat
from multiprocessing import Pool

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset

def load_image(filename: Path | str):
    ext = os.path.splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def unique_mask_values(idx: int, mask_dir: Path | str, mask_suffix: str):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensisons, found {mask.ndim}')

class BasicDataset(Dataset):
    def __init__(self,
                 image_dir: Path | str,
                 mask_dir: Path | str, scale:
                 float = 1.0,
                 mask_suffix: str = ''):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        assert 0 < scale < 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            os.path.splitext(file)[0] for file in os.listdir(image_dir) if os.path.isfile(
            os.path.join(image_dir, file) and not file.startswith('.')
        )]

        if not self.ids:
            raise RuntimeError(f'No input file found in {image_dir}, make sure your put images there')
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning masks files to to determine unique values.')

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
        
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def preprocess(self, mask_values: list[int], pil_image: Image, scale: float, is_mask: bool):
        w, h = pil_image.size()
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_image = pil_image.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.CUBIC)
        img = np.asarray(pil_image)

        if is_mask:
            mask = np.zeros((newW, newH), dtype='int64')
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            
            return mask
        
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            
            if (img > 1).any():
                img = img / 255.0
            
            return img
    
    def __getitem__(self, index):
        name = self.ids[index]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        image_file = list(self.image_dir.glob(name + '.*'))

        assert len(image_file) == 1, f'Either no image or multiple images found for the ID {name}: {image_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])
        image = load_image(image_file[0])

        assert image.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {image.size} and {mask.size}'
        
        image = self.preprocess(self.mask_values, image, self.scale, is_mask=True)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, image_dir: Path | str, mask_dir: Path | str, scale: float = 1.0):
        super().__init__(image_dir, mask_dir, scale, mask_suffix='_mask')