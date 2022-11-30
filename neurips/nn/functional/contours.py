import torch

__all__ = ['remove_border_contours', 'exclude_coordinates']


def remove_border_contours(contours, size, padding, top, right, bottom, left):
    # TODO: Maybe a warning if removed object is larger than overlap
    h, w = size[:2]
    x, y = contours[..., 0], contours[..., 1]
    keep = torch.ones(len(contours), dtype=torch.bool, device=contours.device)
    if top:
        keep = keep & (y > padding).all(1)
    if right:
        keep = keep & (x < (w - padding)).all(1)
    if bottom:
        keep = keep & (y < (h - padding)).all(1)
    if left:
        keep = keep & (x > padding).all(1)

    return keep


def exclude_coordinates(coords, overlap, size):  # in xy format; numpy version
    """Exclude coordinates.
    Notes:
    - Exclude all object instances that share pixels with the exclusion zone.
    - Exclude all object instances that share no pixels with the inner tile zone.
    Args:
        coords: Array[groups, points, 2].
        meta: Tiling information.
    Returns:
        Keep indices. (indices of groups to keep)
    """
    ya, xa = overlap[0], overlap[1]
    yb, xb = max(0, size[0] - overlap[0]), max(0, size[1] - overlap[1])
    x, y = coords[:, :, 0], coords[:, :, 1]

    red_mask = ((x >= xb) | (y >= yb)).any(1)  # right, bottom

    outer_mask = (~((x >= xa) & (x < xb) & (y >= ya) & (y < yb))).all(1)  # only on outer ring, none on inside
    print("outer_mask", outer_mask.shape)
    keep = ~(red_mask | outer_mask)
    return keep
