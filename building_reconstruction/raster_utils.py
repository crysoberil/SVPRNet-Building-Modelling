import numpy as np
import skimage.draw
import warnings


def load_ndsm_as_mesh(ndsm, return_mask=False):
    height, width = ndsm.shape[: 2]

    # mask = np.isfinite(dsm)
    mask = np.logical_and(np.isfinite(ndsm), ndsm > -7000.0)

    # create triangular mesh
    # For each group of 4 vertices, two triangles are created, ordered as:
    # 0 1 -> [1 0 2] (triangle 1)
    # 2 3    [3 1 2] (triangle 2)
    tri_indices = np.empty((height - 1, width - 1, 2, 3), dtype=np.uint32)
    indices = np.arange(height * width, dtype=np.uint32).reshape(height, width)
    tri_indices[:, :, [0, 1], [0, 1]] = indices[:-1, 1:, np.newaxis] # idx 1
    tri_indices[:, :, 0, 1] = indices[: -1, : -1]  # idx 0
    tri_indices[:, :, 1, 0] = indices[1:, 1:]  # idx 3
    tri_indices[:, :, :, 2] = indices[1:, : -1, np.newaxis] # idx 2

    # mask out and/or update based on whether the voxels are valid
    # valid configurations require at least 3 vertices, i.e.:
    # config = 15 (all 4), 14 (0 1 2), 13 (0 1 3), 11 (0 2 3), or 7 (1 2 3)
    config = mask.view(np.uint8)
    config = ((config[:-1, :-1] << 3) | (config[:-1, 1:] << 2) |
              (config[1:, :-1] << 1) | config[1:, 1:])
    tri_mask = np.tile((config == 15)[:, :, np.newaxis], (1, 1, 2))
    tri_mask[config == 14, 0] = True
    tri_mask[config == 7, 1] = True

    # we'll need to redo the triangles for configs 13 and 11
    config_mask = (config == 13)
    tri_mask[config_mask, 0] = True
    tri_indices[config_mask, 0, 2] = indices[1:, 1:][config_mask]

    config_mask = (config == 11)
    tri_mask[config_mask, 1] = True
    tri_indices[config_mask,1,1] = indices[: -1, : -1][config_mask]

    del config, config_mask

    tri_indices = tri_indices[tri_mask]  # Tx3

    del tri_mask

    # now, we need to re-index the vertices according to which are valid
    vertex_indices = np.empty(len(mask.ravel()), dtype=tri_indices.dtype)
    vertex_indices[mask.ravel()] = np.arange(np.count_nonzero(mask))
    tri_indices = vertex_indices[tri_indices]

    del vertex_indices

    # now, create the actual 3D points in UTM
    c, r = np.meshgrid(np.arange(width), np.arange(height))
    c = c[mask]
    r = r[mask]
    z = ndsm[mask]

    # utm: x, y, z
    # x = dsm_entity.x_left + dsm_entity.pix_w * c
    # y = dsm_entity.y_top - dsm_entity.pix_h * r
    x, y = c, height - r

    result = [np.column_stack((x, y, z)), tri_indices]

    if return_mask:
        result.append(mask)

    return result


def rasterize_polygons(polygons, height, width, rasters_in_different_channels=False):
    rows, cols = [], []
    for poly in polygons:
        r, c = skimage.draw.polygon(poly[:, 1], poly[:, 0], shape=[height, width])
        rows.append(r)
        cols.append(c)
    if rasters_in_different_channels:
        num_channels = len(rows)
        raster = np.zeros([height, width, num_channels], dtype=np.bool)
        for i in range(num_channels):
            r, c = rows[i], cols[i]
            raster[r, c, i] = True
        return raster
    else:
        rows = np.hstack(rows)
        cols = np.hstack(cols)
        raster = np.zeros([height, width], dtype=np.bool)
        raster[rows, cols] = True
        return raster


def resize_raster(raster, shape, interp_order=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized = skimage.transform.resize(raster, output_shape=shape, order=interp_order, preserve_range=True)
    return resized