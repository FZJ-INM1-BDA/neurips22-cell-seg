import celldetection as cd
import numpy as np
import cv2

__all__ = ['contours2labels']


def contours2labels(contours, size, overlap=False, max_iter=999, verbose=False):
    labels = cd.data.contours2labels(cd.asnumpy(contours), size, initial_depth=3)

    if not overlap:
        kernel = cv2.getStructuringElement(1, (3, 3))
        mask_sm = np.sum(labels > 0, axis=-1)
        mask = mask_sm > 1  # all overlaps
        if mask.any():
            mask_ = mask_sm == 1  # all cores
            lbl = np.zeros(labels.shape[:2], dtype='float64')  # labels.astype('float32')
            lbl[mask_] = labels.max(-1)[mask_]
            for _ in range(max_iter):
                lbl_ = np.copy(lbl)
                m = mask & (lbl <= 0)
                if not np.any(m):
                    break
                lbl[m] = cv2.dilate(lbl, kernel=kernel)[m]
                if np.allclose(lbl_, lbl):
                    break
            if verbose:
                print('Number of propagation runs:', _)
        else:
            lbl = labels.max(-1)
        labels = lbl.astype('int')
    return labels
