import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coef

@torch.inference_mode
def evaluate(unet, dataloader, device, amp):
    unet.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            img, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            #predict the mask
            mask_pred = unet(mask_true)
            

