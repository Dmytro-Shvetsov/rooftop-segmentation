import numpy as np
import skimage
from skimage.segmentation import watershed
from skimage.measure import label


def watershed_cut(segm:np.ndarray, wngy:np.ndarray, threshold:int=1, object_min_size:int=20, remove_small_holes:bool=True, kernel_size:int=3) -> np.ndarray:
    """
    Builds an instance segmentation mask from a binary one with watershed energies.

    Args:
        segm (np.ndarray): binary semantic segmentation mask of integer values in range [0; 1]
        wngy (np.ndarray): watershed energies mask of integer values in range [0; num_watershed_bins]
        threshold (int, optional): threshold to apply to watershed energies mask (i.e which levels lower than the value to ignore in the energies mask). Defaults to 1.
        object_min_size (int, optional): minimum size to keep the instance object. Defaults to 20.
        remove_small_holes (bool, optional): whether to remove small holes in masks. Defaults to True.
        kernel_size (int, optional): kernel size to apply the filtering. Defaults to 3.

    Returns:
        np.ndarray: resulting instance segmentation mask of integer values in range [0; num_instances]
    """
    segm = segm.astype(bool)

    ccimg = (wngy > threshold) * segm

    # return ccimg 

    if object_min_size is not None:
        ccimg = skimage.morphology.remove_small_objects(ccimg, min_size=object_min_size)

    if remove_small_holes:
        ccimg = skimage.morphology.remove_small_holes(ccimg)

    cclabels = skimage.morphology.label(ccimg)

    ccids = np.unique(cclabels)[1:]

    cclabels_out = np.zeros_like(wngy)
    kernel = np.ones((kernel_size, kernel_size))
    for cid in ccids:
        ccimg_id = (cclabels == cid)
        ccimg_id_dilated = skimage.morphology.binary_dilation(ccimg_id, footprint=kernel)
        cclabels_out[ccimg_id_dilated] = cid

    return cclabels_out


def watershed_energy(energy,
                     msk=None,
                     threshold_energy=1,
                     object_min_size:int=20, 
                     remove_small_holes:bool=True, 
                     kernel_size:int=3,
                     line=True):

    if msk is not None:
        if object_min_size is not None:
            msk = skimage.morphology.remove_small_objects(msk, min_size=object_min_size)

        if remove_small_holes:
            msk = skimage.morphology.remove_small_holes(msk)

    energy_ths = (energy > threshold_energy) * 1

    # Marker labelling
    markers = label(energy_ths)

    if msk is None:
        markers += 1

    labels = watershed(-energy,
                       markers,
                       mask=msk,
                       watershed_line=line)

    ccids = np.unique(labels)[1:]

    cclabels_out = np.zeros_like(labels)
    kernel = np.ones((kernel_size, kernel_size))
    for cid in ccids:
        ccimg_id = (labels == cid)
        ccimg_id_dilated = skimage.morphology.binary_dilation(ccimg_id, footprint=kernel)
        cclabels_out[ccimg_id_dilated] = cid
    return labels


def instance_to_rgb(labels, cmap=None):
    if cmap is None:
        cmap = {iid: np.random.randint(0, 256, 3) for iid in np.unique(labels)}
        # background
        cmap[0] = np.array((127, 127, 127))
    vis = np.zeros(shape=(*labels.shape[:2], 3), dtype=np.uint8)
    for iid in np.unique(labels):
        vis[labels == iid] = cmap[iid][None][None]
    return vis
