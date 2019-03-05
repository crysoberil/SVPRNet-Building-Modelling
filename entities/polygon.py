import numpy as np
from general_utils import geometric_utils


class Polygon(object):
    def __init__(self, nodes):
        self.nodes = [elm for elm in nodes]
        self.shapely_polygon = None
        self.centroid = None
        self.numpy_poly = None
        self.is_closed = geometric_utils.same_point(nodes[0], nodes[-1])

    def get_numpy_polygon(self):
        if self.numpy_poly is None:
            self.numpy_poly = np.array(self.nodes, dtype=np.float64)
        return self.numpy_poly

    def get_bound(self):
        if self.bound is None:
            self.bound = geometric_utils.get_polygon_bounding_box(self.nodes)
        return self.bound

    def get_shapely_polygon(self):
        if self.shapely_polygon is None:
            self.shapely_polygon = geometric_utils.polygon_to_shapely_polygon(self.nodes)
        return self.shapely_polygon

    def get_centroid(self):
        if self.centroid is None:
            end = len(self.nodes) if not(self.is_closed) else len(self.nodes) - 1
            x_avg = sum([x for x, y in self.nodes[: end]]) / end
            y_avg = sum([y for x, y in self.nodes[: end]]) / end
            self.centroid = x_avg, y_avg
        return self.centroid


    def dropped_last_vertex(self):
        nodes = [elm for elm in self.nodes[: -1]]
        return Polygon(nodes, self.index_no, simplified_polygon=self.simplified_polygon, bound=self.bound, building_height=self.bound)

    # def _get_proper_oriented_polygon(self, base_poly):
    #     # They're closed!
    #     def _index_of_leftish_top(x, y):
    #         # Get the point with maximum y - x value amongst the points that are in the left 5% span of the entire x span
    #         left_span_amount = 0.1
    #         x_min, x_max = np.min(x), np.max(x)
    #         max_accepted_x = x_min + (x_max - x_min) * left_span_amount
    #         indices_considered = [idx for idx in range(x.shape[0]) if x[idx] <= max_accepted_x]
    #         index_score_pairs = [(y[idx] - x[idx], idx) for idx in indices_considered]
    #         _, best_idx = max(index_score_pairs)
    #         return best_idx
    #
    #     # Returns numpy polygon
    #     if base_poly is None:
    #         return None
    #     base_poly = np.array(base_poly, dtype=np.float64)
    #     n = base_poly.shape[0] - 1
    #     base_poly = base_poly[: n, :]
    #     x, y = base_poly[:, 0], base_poly[:, 1]
    #     twice_area = np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    #     if twice_area < 0:
    #         x = np.flip(x, 0)
    #         y = np.flip(y, 0)
    #         poly_before_changed_start = np.column_stack((x, y))
    #     else:
    #         poly_before_changed_start = base_poly
    #     best_start_idx = _index_of_leftish_top(x, y)
    #     poly_after_changed_start = np.row_stack((poly_before_changed_start[best_start_idx:, :], poly_before_changed_start[: best_start_idx, :]))
    #     return poly_after_changed_start

    # def get_bound_shapely(self):
    #     if self.bound_shapely is None:
    #         bound = self.get_bound()
    #         self.bound_shapely = bound.to_shapely_polygon()
    #     return self.bound_shapely

    # def simplified_polygon_relative_to_rectangle(self, rect_center_x, rect_center_y, rect_width, rect_height):
    #     # Instance simplified polygons are not closed. But they end in arbitraty length. Remember to pad these with 0 vectors.
    #     """Returns numpy polygon wrt to the rectangle center"""
    #     original_simplified_poly_len = self.simplified_polygon.shape[0]  # Can be at most 19
    #     relative_x = (self.simplified_polygon[:, 0] - rect_center_x) / float(rect_width)
    #     relative_y = (self.simplified_polygon[:, 1] - rect_center_y) / float(rect_height)
    #     relative_poly = np.column_stack((relative_x, relative_y, np.ones([original_simplified_poly_len], dtype=np.float32)))
    #     filler_len = global_config.MAX_POLYGON_LEN - original_simplified_poly_len - 1
    #     filler = np.zeros([filler_len, 3], dtype=np.float64)
    #     internal_row_for_closing = np.array(relative_poly[0 : 1, :])
    #     internal_row_for_closing[0, 2] = 0.0
    #     res = np.row_stack((relative_poly, internal_row_for_closing, filler))
    #     res = np.array(res, dtype=np.float32)
    #     return res


class Building(Polygon):
    def __init__(self, nodes, ground_height, building_avg_height, building_median_height):
        super(Building, self).__init__(nodes)
        self.ground_height = ground_height
        self.building_avg_height = building_avg_height
        self.building_median_height = building_median_height

    def __as_numpy_array_with_height(self, h):
        nodes_2d = np.array(self.nodes, dtype=np.float32)
        heights = np.ones([nodes_2d.shape[0], 1], dtype=np.float32) * h
        as_3d_points = np.column_stack((nodes_2d, heights))
        return as_3d_points

    def as_numpy_array_with_median_height(self):
        return self.__as_numpy_array_with_height(self.building_median_height)

    def as_numpy_array_with_average_height(self):
        return self.__as_numpy_array_with_height(self.building_avg_height)


def reproject_polygons_to_image_space(polygon_entities, geodetic_bound, target_long_span, target_lat_span):
    res = []
    for poly in polygon_entities:
        reprojected_nodes = [geodetic_bound.get_relative_position(node_long, node_lat, target_long_span, target_lat_span) for node_long, node_lat in poly.nodes]
        reproj_entity = Polygon(reprojected_nodes)
        res.append(reproj_entity)
    return res



def get_mean_shifted_buildings(building_entities):
    centroids = np.array([poly.get_centroid() for poly in building_entities], dtype=np.float64)
    mean_x, mean_y = np.average(centroids[:, 0]), np.average((centroids[:, 1]))
    mean_shifted_entities = []

    for building in building_entities:
        shifted = Polygon([(n[0] - mean_x, n[1] - mean_y) for n in building.nodes])
        mean_shifted_entities.append(shifted)

    return mean_shifted_entities
