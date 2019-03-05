import entities
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np
import scipy.ndimage


def polygon_to_shapely_polygon(polygon):
    return ShapelyPolygon(polygon)


def get_polygon_bounding_box(polygon):
    x_min, x_max, y_min, y_max = polygon[0][0], polygon[0][0], polygon[0][1], polygon[0][1]
    for i in range(1, len(polygon)):
        x, y = polygon[i]
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    return entities.geodetic_bound.GeodeticBound(x_min, x_max, y_min, y_max)


def get_polygon_rectangle_iou(shapely_polygon, rectangle, shapely_polygon_area=None):
    shapely_rectangle = rectangle if isinstance(rectangle, ShapelyPolygon) else polygon_to_shapely_polygon(rectangle)
    if shapely_polygon_area is None:
        shapely_polygon_area = shapely_polygon.area
    intersection = shapely_polygon.intersection(shapely_rectangle)
    intersection_area = float(intersection.area)
    if intersection_area < 1e-5:  # Threshold
        return 0.0
    shapely_rectangle_area = shapely_rectangle.area
    iou = intersection_area / (shapely_polygon_area + shapely_rectangle_area - intersection_area)
    return iou


def get_polygon_bounds_contained_in_boundingbox(polygon_entities, shapely_bounding_rectangle, accept_partially_contained):
    res = []
    bounding_box_area = shapely_bounding_rectangle.area
    for poly_entity in polygon_entities:
        if accept_partially_contained:
            iou = get_polygon_rectangle_iou(shapely_bounding_rectangle, poly_entity.get_bound_shapely(), shapely_polygon_area=bounding_box_area)
            if iou > 1e-5:
                res.append(poly_entity)
        else:
            if shapely_bounding_rectangle.contains(poly_entity.get_bound_shapely()):
                res.append(poly_entity)
    return res


def same_point(p1, p2, dist_threshold=1e-8):
    d1 = abs(p1[0] - p2[0])
    d2 = abs(p1[1] - p2[1])
    return d1 < dist_threshold and d2 < dist_threshold