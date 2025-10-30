import os, glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

def to_channels(arr: np.ndarray, num_classes: int, dtype=np.uint8) -> np.ndarray:
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in range(num_classes):
        res[..., c][arr == c] = 1
    return res

def _best_mask_slice(vol):
    if vol.ndim == 2:
        return vol
    k = int(np.argmax((vol > 0).reshape(-1, vol.shape[-1]).sum(axis=0)))
    return vol[:, :, k]

def _central_slice(vol):
    if vol.ndim == 2:
        return vol
    return vol[:, :, vol.shape[-1] // 2]

def _coerce_paths(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if os.path.isdir(x):
        return sorted(glob.glob(os.path.join(x, "*.nii*")))
    if os.path.isfile(x):
        return [x]
    raise FileNotFoundError(str(x))

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False, output_dir="output_images", is_mask=False, num_classes=None, target_id=1):
    os.makedirs(output_dir, exist_ok=True)
    paths = _coerce_paths(imageNames)
    affines = []
    first_vol = nib.load(paths[0]).get_fdata(dtype=np.float32, caching='unchanged')
    first2d = _best_mask_slice(first_vol) if is_mask else _central_slice(first_vol)
    if is_mask and categorical:
        k = int(np.max(first2d)) + 1 if num_classes is None else int(num_classes)
        first2d_oh = to_channels(first2d.astype(np.int32), num_classes=k, dtype=np.uint8)
        rows, cols, ch = first2d_oh.shape
        images = np.zeros((len(paths), rows, cols, ch), dtype=np.uint8)
        vmin_save, vmax_save = 0, k - 1
    elif is_mask and not categorical:
        rows, cols = first2d.shape
        images = np.zeros((len(paths), rows, cols), dtype=np.uint8)
        vmin_save, vmax_save = 0, 255
    else:
        rows, cols = first2d.shape
        images = np.zeros((len(paths), rows, cols), dtype=dtype)
        vmin_save, vmax_save = 0, 255
    if is_mask and categorical and num_classes is None:
        num_classes = int(np.max([np.max(_best_mask_slice(nib.load(p).get_fdata(dtype=np.float32, caching='unchanged'))) for p in paths])) + 1
    for i, p in enumerate(tqdm(paths)):
        ni = nib.load(p)
        vol = ni.get_fdata(dtype=np.float32, caching='unchanged')
        affine = ni.affine
        arr2d = (_best_mask_slice(vol) if is_mask else _central_slice(vol)).astype(dtype)
        if is_mask:
            if categorical:
                onehot = to_channels(arr2d.astype(np.int32), num_classes=int(num_classes), dtype=np.uint8)
                images[i] = onehot
                save2d = np.argmax(onehot, axis=-1).astype(np.uint8)
            else:
                bin2d = (arr2d == target_id).astype(np.uint8)
                images[i] = bin2d
                save2d = (bin2d * 255).astype(np.uint8)
        else:
            if normImage:
                std = arr2d.std()
                arr2d = (arr2d - arr2d.mean()) / (std + 1e-8)
                p1, p99 = np.percentile(arr2d, 1), np.percentile(arr2d, 99)
                disp = np.clip((arr2d - p1) / (p99 - p1 + 1e-8), 0, 1)
                save2d = (disp * 255).astype(np.uint8)
            else:
                p1, p99 = np.percentile(arr2d, 1), np.percentile(arr2d, 99)
                disp = np.clip((arr2d - p1) / (p99 - p1 + 1e-8), 0, 1)
                save2d = (disp * 255).astype(np.uint8)
            images[i] = arr2d.astype(dtype)
        affines.append(affine)
        base = os.path.splitext(os.path.basename(p))[0].replace(".nii", "")
        fp = os.path.join(output_dir, f"{base}.png")
        plt.imsave(fp, save2d, cmap='gray', vmin=vmin_save, vmax=vmax_save)
        if i > 20 and early_stop:
            break
    return (images, affines) if getAffines else images


#data_dir_scans = "Nifti files\semantic_MRs_anon"

data_dir_mask = "Nifti files\semantic_labels_anon"

#image_paths_scans = sorted(glob.glob(os.path.join(data_dir_scans, "*.nii.gz")))
image_paths_masks = sorted(glob.glob(os.path.join(data_dir_mask, "*.nii.gz")))

masks = load_data_2D(image_paths_masks, categorical=False, normImage=False, output_dir="Nifti files/Labels_images", is_mask=True ,target_id=2)

#scans = load_data_2D(image_paths_scans, categorical=False, normImage=True, output_dir="Nifti files/Scan_images", is_mask=False)