import numpy as np
from random import Random


MINIMUM_CLIP_PERCENTILE = 1
MAXIMUM_CLIP_PERCENTILE = 99


def __normalize_matrix(mat, target_dtype=np.uint8, clip_minimum_percentile=MINIMUM_CLIP_PERCENTILE, clip_maximum_percentile=MAXIMUM_CLIP_PERCENTILE, scale=255):
    mat = mat.astype(np.float32)
    percentile_clip_min = np.percentile(mat, clip_minimum_percentile)
    percentile_clip_max = np.percentile(mat, clip_maximum_percentile)

    mat = (mat - percentile_clip_min) * (scale / (percentile_clip_max - percentile_clip_min))
    np.clip(mat, 0.0, scale, mat)
    mat = mat.astype(target_dtype)
    return mat


def format_image_for_visualization(image, pad_band=True, target_dtype=np.uint8, clip_minimum_percentile=MINIMUM_CLIP_PERCENTILE, clip_maximum_percentile=MAXIMUM_CLIP_PERCENTILE, min_val=-7000.0):
    assert 2 <= len(image.shape) <= 3
    assert target_dtype == np.uint8 or target_dtype == np.float32
    height, width = image.shape[: 2]
    if len(image.shape) == 2:  # Most likely heigh map of mask
        if image.dtype == np.float32 or image.dtype == np.float64:  # Height map
            # Initial hard clipping first
            image = np.clip(image, min_val, None)
            # Now through away minimum value
            is_valid = image > min_val + 1e-6
            throw_away_min = image[is_valid].min()
            np.clip(image, throw_away_min, None, image)
            soft_clip_min = np.percentile(image, clip_minimum_percentile)
            soft_clip_max = np.percentile(image, clip_maximum_percentile)
            image = (image - soft_clip_min) / (soft_clip_max - soft_clip_min)
            np.clip(image, 0.0, 1.0, image)  # Everything 0 to 1 now
            if not pad_band:
                return image
            cols = np.empty([height, width, 3], dtype=np.float32)
            cols[:, :, 0] = cols[:, :, 1] = cols[:, :, 2] = image
            if target_dtype == np.uint8:
                cols = (cols * 255).astype(dtype=np.uint8)
            return cols
        else:  # Mask
            maxm = np.percentile(image, MAXIMUM_CLIP_PERCENTILE)
            minm = image.min()
            normalized = (image - minm).astype(dtype=np.float32) / (maxm - minm)
            np.clip(normalized, 0.0, 1.0)
            if not pad_band:
                return normalized
            cols = np.empty([height, width, 3], dtype=np.float32)
            cols[:, :, 0] = cols[:, :, 1] = cols[:, :, 2] = normalized
            return cols
    elif len(image.shape) == 3:
        # if image.shape[2] == 4:
        #     return format_image_for_visualization(image[:, :, : 3], pad_band, min_val)

        res = np.empty_like(image, dtype=target_dtype)
        for channel_no in range(image.shape[2]):
            res[:, :, channel_no] = __normalize_matrix(image[:, :, channel_no], clip_minimum_percentile=clip_minimum_percentile, clip_maximum_percentile=clip_maximum_percentile, target_dtype=target_dtype, scale=(255 if target_dtype == np.uint8 else 1.0))
        return res


class ArbitraryColorGenerator:
    def __init__(self, num_of_ids):
        rand_gen = Random(243534)
        self._used_colors = set()
        self._used_color_list = []
        self._min_sum = 30
        self._min_col_diff = 15
        while len(self._used_color_list) < num_of_ids:
            color = rand_gen.randint(1, 255), rand_gen.randint(1, 255), rand_gen.randint(1, 255)
            if self._is_allowable_color(color):
                self._used_colors.add(color)
                self._used_color_list.append(np.array(color, dtype=np.float32) / 255.0)

    def _is_allowable_color(self, col):
        r, g, b = col
        if (r + g + b < self._min_sum) or (abs(r - g) < self._min_col_diff and abs(r - b) < self._min_col_diff) or (col in self._used_colors):
            return False
        return True

    def get_color(self, id):
        assert id < len(self._used_color_list)
        return self._used_color_list[id]


    def get_uint8_color_to_index_map(self):
        uint8_color_to_index = dict()
        for idx, col in enumerate(self._used_color_list):
            col = (col * 255.0 + 1e-5).astype(dtype=np.int32).tolist()
            col = tuple(col)
            uint8_color_to_index[col] = idx
        return uint8_color_to_index

    def get_int32_color_to_index_map(self):
        int32_color_to_index = dict()
        for idx, col in enumerate(self._used_color_list):
            col = (col * 255.0 + 1e-5).astype(dtype=np.int32).tolist()
            col_int = (col[0] << 16) | (col[1] << 8) | col[2]
            int32_color_to_index[col_int] = idx
        return int32_color_to_index


def to_float_colors(image):
    if image.dtype == np.uint8:
        image = image.astype(dtype=np.float32)
        image = np.divide(image, 255.0)
        return image

    if image.dtype == np.uint16:
        image = image.astype(dtype=np.float32)
        image = np.divide(image, 65535.0)
        return image

    assert image.dtype == np.float32 or image.dtype == np.float64
    return image


def display_numpy_image(img_numpy):
    if len(img_numpy.shape) > 2 and img_numpy.shape[2] == 4:
        img_numpy = img_numpy[:, :, : 3]
    if img_numpy.dtype != np.float32 and img_numpy.dtype != np.float64:
        img_numpy = img_numpy / 255.0
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot
    pyplot.imshow(img_numpy, interpolation="nearest")
    pyplot.show()


def visualize_raster(raster, title=None):
    raster = np.array(raster, dtype=np.float32)
    formatted = format_image_for_visualization(raster, clip_minimum_percentile=0.0, clip_maximum_percentile=100.0)

    from matplotlib import pyplot
    pyplot.imshow(formatted, interpolation="nearest")
    if title is not None:
        pyplot.title(title)
    pyplot.show()


def display_labeled_image(img, mask=None, background_label=-1, title=None):
    if mask is not None:
        img = np.copy(img)
        img[np.logical_not(mask)] = background_label
    assigned_colors = {background_label: np.array([0, 0, 0], dtype=np.uint8)}
    labels = np.unique(img).tolist()
    for lab in labels:
        if lab != background_label:
            col1 = (346 + lab * 2362) % 192 + 63
            col2 = (74535 + lab * 34) % 192 + 63
            col3 = (5 + lab * 245) % 192 + 163
            assigned_colors[lab] = np.array([col1, col2, col3], dtype=np.uint8)

    colored_image = np.empty([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    for r in range(colored_image.shape[0]):
        for c in range(colored_image.shape[1]):
            colored_image[r, c] = assigned_colors[img[r, c]]
    display_numpy_image(colored_image, title=title)


def display_numpy_image_grid(images):
    n = len(images)
    assert n >= 1

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot

    if n == 1:
        num_rows, num_cols = 1, 1
    else:
        import math
        num_rows = int(math.sqrt(n))
        num_cols = int(math.ceil(float(n) / num_rows) + 1e-5)
    fig = pyplot.figure(figsize=(16, 12))
    for img_no in range(n):
        fig.add_subplot(num_rows, num_cols, img_no + 1)
        pyplot.imshow(images[img_no])
    pyplot.show()


def display_masked_image(image, mask):
    image = format_image_for_visualization(image, target_dtype=np.float32)
    img_original = np.array(image)
    image[:, :, 0] = np.where(mask, 0.7, image[:, :, 0])
    display_numpy_image_grid([img_original, image])


def display_polygons_over_image(polygons, image, height=None, width=None, revert_y_axis=False):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot

    if height is None:
        height = image.shape[0]
    if width is None:
        width = image.shape[1]

    pyplot.imshow(image)

    for poly in polygons:
        poly = poly.tolist()
        poly.append(poly[0])
        xs, ys = zip(*poly)
        pyplot.plot(xs, ys, color='r')
    ax = pyplot.gca()
    ax.set_xlim(0, width)
    if revert_y_axis:
        ax.set_ylim(height, 0)
    else:
        ax.set_ylim(0, height)
    pyplot.show()

