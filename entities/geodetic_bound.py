from general_utils import gdal_utils
from general_utils import geometric_utils


class GeodeticBound:
    def __init__(self, long_min, long_max, lat_min, lat_max):
        self.long_min = min(long_min, long_max)
        self.long_max = max(long_min, long_max)
        self.lat_min = min(lat_min, lat_max)
        self.lat_max = max(lat_min, lat_max)


    def latitude_in_range(self, latitude):
        return self.lat_min <= latitude <= self.lat_max


    def longitude_in_range(self, longitude):
        return self.long_min <= longitude <= self.long_max

    def contains(self, longitude, latitude):
        return self.lat_min <= latitude <= self.lat_max and self.long_min <= longitude <= self.long_max

    def overlaps_with(self, second):
        if self.long_max < second.long_min or second.long_max < self.long_min:
            return False
        if self.lat_max < second.lat_min or second.lat_max < self.lat_min:
            return False
        return True


    def __str__(self):
        return "Longitude range = [{}, {}]; \nLatitude range = [{}, {}]".format(self.long_min, self.long_max, self.lat_min, self.lat_max)


    def to_shapely_polygon(self):
        rect = [(self.long_min, self.lat_min), (self.long_max, self.lat_min), (self.long_max, self.lat_max), (self.long_min, self.lat_max)]
        return geometric_utils.polygon_to_shapely_polygon(rect)


    def get_pixel_location(self, longitude, latitude, img_width, img_height):
        def _get_spatial_location(pos, l_b, u_b, pixel_range):
            f = (pos - l_b) / (u_b - l_b)
            mapped = f * pixel_range
            return min(int(mapped), pixel_range - 1)

        x = _get_spatial_location(longitude, self.long_min, self.long_max, img_width)
        y = _get_spatial_location(latitude, self.lat_max, self.lat_min, img_height)
        return x, y

    def get_relative_position(self, point_long, point_lat, target_long_span, target_lat_span):
        f_x = (point_long - self.long_min) / (self.long_max - self.long_min) * target_long_span
        f_y = (self.lat_max - point_lat) / (self.lat_max - self.lat_min) * target_lat_span
        return f_x, f_y

    def projection_transform(self, source_proj4, dest_proj4):
        t_left, t_bottom = gdal_utils.transform_projection(self.long_min, self.lat_min, source_proj4, dest_proj4)
        t_right, t_top = gdal_utils.transform_projection(self.long_max, self.lat_max, source_proj4, dest_proj4)
        return GeodeticBound(t_left, t_right, t_bottom, t_top)
