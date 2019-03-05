from entities import geodetic_bound
import numpy as np


class ImageEntity:
    def __init__(self, img, x_left, y_top, pix_w, pix_h, height_pixel, width_pixel):
        assert pix_w >= 0 and pix_h >= 0
        self.img = img
        self.x_left = x_left
        self.y_top = y_top
        self.pix_w = pix_w
        self.pix_h = pix_h
        self.width_pixel = width_pixel
        self.height_pixel = height_pixel

    def width_meter(self):
        return self.width_pixel * self.pix_w

    def height_meter(self):
        return self.height_pixel * self.pix_h

    @property
    def x_right(self):
        return self.x_left + self.pix_w * self.width_pixel

    @property
    def y_bot(self):
        return self.y_top - self.pix_h * self.height_pixel

    def within_range_x(self, x, exclude_last=False):
        x_right = self.x_left + self.pix_w * self.width_pixel
        if exclude_last:
            x_right -= self.pix_w
        bounded_from_left = np.greater_equal(x, self.x_left)
        bounded_from_right = np.less_equal(x, x_right)
        return np.logical_and(bounded_from_left, bounded_from_right)

    def within_range_y(self, y, exclude_last=False):
        y_bot = self.y_top - self.pix_h * self.height_pixel
        if exclude_last:
            y_bot += self.pix_h
        bounded_from_bottom = np.greater_equal(y, y_bot)
        bounded_from_top = np.less_equal(y, self.y_top)
        return np.logical_and(bounded_from_bottom, bounded_from_top)

    def crop_by_coordinate(self, x_start, x_end, y_start, y_end, return_offsets=False):
        col_start, col_end = int((x_start - self.x_left) / self.pix_w), int(
            (x_end - self.x_left) / self.pix_w + .50000001)  # Inclusive
        row_start, row_end = int((self.y_top - y_end) / self.pix_h), int(
            (self.y_top - y_start) / self.pix_h + .50000001)  # Inclusive
        cropped_x_left = self.x_left + col_start * self.pix_w
        cropped_y_top = self.y_top - row_start * self.pix_h
        cropped_entity = ImageEntity(self.img[row_start: row_end + 1, col_start: col_end + 1], cropped_x_left,
                                     cropped_y_top, self.pix_w, self.pix_h, row_end - row_start + 1,
                                     col_end - col_start + 1)
        if return_offsets:
            return cropped_entity, col_start, row_start
        return cropped_entity

    def crop_by_row_col(self, row_start, row_end, col_start, col_end):
        cropped_x_left = self.x_left + col_start * self.pix_w
        cropped_y_top = self.y_top - row_start * self.pix_h
        return ImageEntity(self.img[row_start: row_end + 1, col_start: col_end + 1], cropped_x_left, cropped_y_top,
                           self.pix_w, self.pix_h, row_end - row_start + 1, col_end - col_start + 1)

    def image_entity_with_new_raster(self, raster):
        return ImageEntity(raster, self.x_left, self.y_top, self.pix_w, self.pix_h, self.height_pixel, self.width_pixel)

    def contains(self, image_entity):
        if not (self.x_left <= image_entity.x_left and self.y_top >= image_entity.y_top):
            return False

        self_x_right, second_x_right = self.x_left + self.pix_w * self.width_pixel, image_entity.x_left + image_entity.pix_w * image_entity.width_pixel
        if self_x_right < second_x_right:
            return False
        self_y_bottom, second_y_bottom = self.y_top - self.pix_h * self.height_pixel, image_entity.y_top - image_entity.pix_h * image_entity.height_pixel

        return self_y_bottom <= second_y_bottom

    def intersects_with(self, image_entity):
        x_intersection = self.x_left <= image_entity.x_left <= self.x_right or self.x_left <= image_entity.x_right <= self.x_right
        if not x_intersection:
            return False
        y_intersection = self.y_bot <= image_entity.y_bot <= self.y_top or self.y_bot <= image_entity.y_top <= self.y_top
        return y_intersection

    def locate_row_col(self, x, y):
        col = (x - self.x_left) / self.pix_w
        row = (self.y_top - y) / self.pix_h
        col = (np.round(col) + 1e-8).astype(dtype=np.int32)
        row = (np.round(row) + 1e-8).astype(dtype=np.int32)
        return row, col

    def row_col_to_geo_location(self, rows, cols):
        y = self.y_top - rows * self.pix_h
        x = self.x_left + cols * self.pix_w
        return x, y

    def get_geodetic_bound(self):
        x_right = self.x_left + self.pix_w * self.width_pixel
        y_bot = self.y_top - self.pix_h * self.height_pixel
        return geodetic_bound.GeodeticBound(self.x_left, x_right, self.y_top, y_bot)

    def get_all_geo_points_values(self):
        assert len(self.img.shape) == 2
        values = self.img.ravel()
        height, width = self.height_pixel, self.width_pixel
        ind = np.indices((height, width))
        rows, cols = ind[0].ravel(), ind[1].ravel()
        x, y = self.row_col_to_geo_location(rows, cols)
        point_values = np.column_stack([x[:, np.newaxis], y[:, np.newaxis], values[:, np.newaxis]]).astype(dtype=np.float64)
        return point_values

    def enrich_using(self, source_entity, reduction_operation=np.maximum):
        assert len(self.img.shape) == 2
        assert abs(self.pix_w - source_entity.pix_w) < 1e-4 and abs(self.pix_h - source_entity.pix_h) < 1e-4
        source_geopoint_values = source_entity.get_all_geo_points_values()
        within_x_range = self.within_range_x(source_geopoint_values[:, 0], exclude_last=True)
        within_y_range = self.within_range_y(source_geopoint_values[:, 1], exclude_last=True)
        point_present_in_target = np.logical_and(within_x_range, within_y_range)
        if point_present_in_target.sum() == 0:
            return  # Nothing to do
        source_geopoint_values = source_geopoint_values[point_present_in_target, :]
        target_rows, target_cols = self.locate_row_col(source_geopoint_values[:, 0], source_geopoint_values[:, 1])
        self.img[target_rows, target_cols] = reduction_operation(self.img[target_rows, target_cols], source_geopoint_values[:, 2])