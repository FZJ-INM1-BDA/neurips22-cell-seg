__all__ = ['coords_transpose', 'coords_rot90', 'boxes_transpose', 'boxes_rot90']


def coords_transpose(coords):
    """Transpose coords.

    Transposes coordinates.

    Args:
        coords: Tensor[-1, 2] in xy format.

    Returns:

    """
    assert coords.shape[-1] == 2
    return coords[..., [1, 0]]


def boxes_transpose(coords):
    """Transpose boxes.

    Transposes bounding boxes.

    Args:
        coords: Tensor[-1, 4] in (x0,y0,x1,y1) format.

    Returns:

    """
    assert coords.shape[-1] == 4
    return coords[..., [1, 0, 3, 2]]


def coords_rot90(coords, h, w, k=1):
    assert coords.shape[-1] == 2
    if k <= 0:
        return coords
    coords = coords_transpose(coords)
    coords[..., 1] = w - coords[..., 1]
    k -= 1
    if k:
        coords = coords_rot90(coords, w, h, k=k)
    return coords


def boxes_rot90(coords, h, w, k=1):
    assert coords.shape[-1] == 4
    if k <= 0:
        return coords
    coords = boxes_transpose(coords)
    coords[..., [1, 3]] = w - coords[..., [1, 3]]
    k -= 1
    if k:
        coords = boxes_rot90(coords, w, h, k=k)
    return coords


