from general_utils import misc_utils
import numpy as np
import skimage.transform
import warnings
import scipy.ndimage.morphology
import skimage.morphology
import bilateral_solver
# import rasterio
import rasterio.features
import shapely.geometry
from building_reconstruction import raster_utils
from general_utils import file_utilities
from triangulation import TriangleWrapper


class BuildingModellingParams:
    prox_heatmap_alpha = 1.0
    pixel_to_meter = 0.3

    # Proximity heatmap zero level set
    proximity_heatmap_boundary_threshold = 0.5

    # Bilateral solver
    distance_from_boundary_limit = 5.0
    bl_confidence_high = 0.95
    bl_confidence_low = 0.7

    # Polygon simplification
    shapely_simplification_sigma = 5.0
    polygon_extraction_remove_small_objects_size = 36

    # Triangulation
    triangle_library_path = "/Users/crysoberil/PycharmProjects/UNC Assignments/Research/Satellite Building Reconstruction/SVPRNet Building Modeller/triangle/triangle"
    trianglulation_tmp_dir = "/tmp/triangle_temp"


class BuildingModeller:
    def __init__(self, image, cascaded_mask, confidence, predicted_building_heights, ndsm, proximity_heatmap, modelling_params=BuildingModellingParams()):
        self.image = image
        self.cascaded_mask = cascaded_mask
        self.confidence = confidence
        self.predicted_building_heights = predicted_building_heights
        self.ndsm = ndsm
        # TODO Jisan ensure if this is correct. Supposedly, this is to account for per pixel resolution.
        self.ndsm = self.ndsm / modelling_params.pixel_to_meter
        self.proximity_heatmap = proximity_heatmap
        self.modelling_params = modelling_params
        self.n = self.image.shape[0]

        self.image_visualization = misc_utils.format_image_for_visualization(self.image[:, :, [4, 2, 1]], clip_minimum_percentile=3, clip_maximum_percentile=97)
        # self.proximity_heatmap_zero_level_set = np.logical_and(proximity_heatmap > -0.5, proximity_heatmap < 0.5)

    def _get_mask_scores_(self):
        target_dim = self.n
        proximity_heatmap = raster_utils.resize_raster(self.proximity_heatmap, (target_dim, target_dim), interp_order=1)
        proximity_heatmap_zero_level_set = np.logical_and(proximity_heatmap > -self.modelling_params.proximity_heatmap_boundary_threshold, proximity_heatmap < self.modelling_params.proximity_heatmap_boundary_threshold)
        distances_from_proximity_heatmap = scipy.ndimage.morphology.distance_transform_edt(np.logical_not(proximity_heatmap_zero_level_set))

        # misc_utils.display_numpy_image(distances_from_proximity_heatmap)
        # misc_utils.display_numpy_image_grid([proximity_heatmap_zero_level_set, distances_from_proximity_heatmap])

        mask_count = self.cascaded_mask.shape[2]
        mask_scores = np.zeros([mask_count], dtype=np.float32)

        for mask_no in range(mask_count):
            building_mask = self.cascaded_mask[:, :, mask_no]
            # TODO Jisan: improve mask detection
            _d = scipy.ndimage.morphology.distance_transform_edt(np.logical_not(building_mask))
            outline = np.abs(_d - 1) < 1e-8
            prox_heatmap_disaggrements = distances_from_proximity_heatmap[outline]
            mean_disagreement = np.mean(prox_heatmap_disaggrements) / float(self.n)
            prox_h_score = 1.0 - mean_disagreement
            mask_scores[mask_no] = self.confidence[mask_no] + self.modelling_params.prox_heatmap_alpha * prox_h_score

        return mask_scores


    def _apply_ndsm_to_mask_(self, mask):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ndsm = skimage.transform.resize(self.ndsm, output_shape=mask.shape, order=1, preserve_range=True)
        return mask * ndsm


    def _apply_bilateral_solver_(self, mask, pixel_distance_limit=None, confidence_high=None, confidence_low=None):
        if pixel_distance_limit is None:
            pixel_distance_limit = self.modelling_params.distance_from_boundary_limit
        if confidence_high is None:
            confidence_high = self.modelling_params.bl_confidence_high
        if confidence_low is None:
            confidence_low = self.modelling_params.bl_confidence_low

        all_buildings_mask = np.sum(mask, axis=2) > 0
        inverted_mask = np.logical_not(all_buildings_mask)
        d1 = scipy.ndimage.morphology.distance_transform_edt(inverted_mask)
        d2 = scipy.ndimage.morphology.distance_transform_edt(all_buildings_mask)
        inconfident_zone_1 = np.logical_and(all_buildings_mask, d2 < pixel_distance_limit)
        inconfident_zone_2 = np.logical_and(inverted_mask, d1 < pixel_distance_limit)
        inconfident_zone = np.logical_or(inconfident_zone_1, inconfident_zone_2)
        # misc_utils.display_numpy_image_grid([all_buildings_mask, d1, d2, inconfident_zone_1, inconfident_zone_2, inconfident_zone])

        bl_solver_target = all_buildings_mask.astype(dtype=np.float64)
        bl_solver_conf = np.empty_like(bl_solver_target)
        bl_solver_conf[inconfident_zone] = confidence_low
        bl_solver_conf[np.logical_not(inconfident_zone)] = confidence_high

        bilateral_grid = bilateral_solver.BilateralGrid(self.image_visualization)
        bilaterally_solved = bilateral_solver.BilateralSolver(bilateral_grid).solve(bl_solver_target.reshape([-1, 1]), bl_solver_conf.reshape([-1, 1])).reshape(all_buildings_mask.shape)
        # misc_utils.display_numpy_image_grid([self.image_visualization, all_buildings_mask, d1, d2, inconfident_zone_1, inconfident_zone_2, inconfident_zone, bilaterally_solved, bilaterally_solved > 0.5])
        return bilaterally_solved > 0.5

    def _extract_polygons_from_mask_(self, mask, shapely_simplification_sigma=None):
        if shapely_simplification_sigma is None:
            shapely_simplification_sigma = self.modelling_params.shapely_simplification_sigma
        mask = skimage.morphology.remove_small_objects(mask, min_size=self.modelling_params.polygon_extraction_remove_small_objects_size)
        json_polys = ({"properties": {"raster_val": v}, "geometry": s} for i, (s, v) in enumerate(rasterio.features.shapes(np.ones_like(mask, dtype=np.uint8), mask=mask)))
        shapely_polys = [shapely.geometry.shape(elm["geometry"]).simplify(shapely_simplification_sigma) for elm in json_polys]
        numpy_poly_points = [poly.exterior.coords.xy for poly in shapely_polys]
        numpy_polys = [np.column_stack((np.array(x, dtype=np.float32)[:, np.newaxis], np.array(y, dtype=np.float32)[:, np.newaxis])) for (x, y) in numpy_poly_points]
        # misc_utils.display_polygons_over_image(numpy_polys, self.image_visualization)
        assert sum([(np.abs(p[0] - p[-1]) > 1e-8).sum() for p in numpy_polys]) == 0
        return numpy_polys

    def _mesh_polygons_with_heights_(self, polygons_with_heights):
        def _make_3d_point_(point2d, height):
            point3d = np.empty([3], dtype=np.float32)
            point3d[: 2] = point2d
            point3d[2] = height
            return point3d

        def _mesh_polygon_(poly, height):
            def _moded_(n):
                return n % poly_len

            poly_len = poly.shape[0] - 1
            v = np.zeros([2 * poly_len, 3], dtype=np.float32)
            f = []
            for i in range(poly_len):
                p1 = poly[i, :]
                v[i, :] = _make_3d_point_(p1, 0.0)
                v[i + poly_len, :] = _make_3d_point_(p1, height)
                f.append([i, poly_len + _moded_(i + 1), poly_len + i])
                f.append([i, _moded_(i + 1), poly_len + _moded_(i + 1)])

            f = np.array(f, dtype=np.int32)

            # Triangulate the top face now
            top_face_vertices = v[poly_len: 2 * poly_len, : 2]
            top_face_vertices_edges = np.array([(i, (i + 1) % poly_len) for i in range(poly_len)], dtype=np.int32)
            top_v, top_f = TriangleWrapper.triangulate_multipolygon(top_face_vertices, [top_face_vertices_edges], self.modelling_params.triangle_library_path, self.modelling_params.trianglulation_tmp_dir)
            top_f[:, [0, 1, 2]] = top_f[:, [0, 2, 1]]
            top_v = np.column_stack([top_v, np.ones([top_v.shape[0], 1], dtype=np.float32) * height])
            top_f += f.shape[0]
            v = np.row_stack([v, top_v])
            f = np.row_stack([f, top_f])

            return v, f

        img_dim = self.image_visualization.shape[: 2]
        ground_verts = np.array([[0.0, 0.0, 0.0], [img_dim[1], 0.0, 0.0], [img_dim[1], img_dim[0], 0.0], [0.0, img_dim[0], 0.0]], dtype=np.float32)
        ground_faces = np.array([[0, 3, 1], [3, 2, 1]], dtype=np.int32)
        vertices, faces = [ground_verts], [ground_faces]
        num_vertices_so_far = ground_verts.shape[0]
        # TODO remove
        # vertices, faces = [], []
        # num_vertices_so_far = 0

        for polygon, height in polygons_with_heights:
            v, f = _mesh_polygon_(polygon, height)
            vertices.append(v)
            faces.append(f + num_vertices_so_far)
            num_vertices_so_far += v.shape[0]

        vertices = np.row_stack(vertices)
        faces = np.row_stack(faces)
        return vertices, faces

    def get_building_model(self, use_median_height_of_building=True, median_height_from_ndsm=True):
        mask_scores = self._get_mask_scores_()
        sorted_masks_desc = sorted(range(self.cascaded_mask.shape[2]), key=(lambda x: -mask_scores[x]))
        combined_mask = np.zeros([self.n, self.n], dtype=np.bool)
        keep_masks = []

        for mask_no in sorted_masks_desc:
            if np.logical_and(combined_mask, self.cascaded_mask[:, :, mask_no]).sum() == 0:
                keep_masks.append(mask_no)
                combined_mask[self.cascaded_mask[:, :, mask_no]] = True

        final_masks = self.cascaded_mask[:, :, keep_masks]

        bilaterally_solved_mask = self._apply_bilateral_solver_(final_masks)
        building_polygons = self._extract_polygons_from_mask_(bilaterally_solved_mask)

        if use_median_height_of_building:
            if median_height_from_ndsm:
                # for b_p in building_polygons:
                #     b_p[: 1] = self.image_visualization.shape[0] - b_p[: 1]
                rasterized_buildings = raster_utils.rasterize_polygons(building_polygons, self.image_visualization.shape[0], self.image_visualization.shape[1], rasters_in_different_channels=True)
                larger_ndsm = raster_utils.resize_raster(self.ndsm, self.image_visualization.shape[: 2], interp_order=1)
                polygons_with_heights = []
                for poly_no, poly in enumerate(building_polygons):
                    building_ndsm = larger_ndsm[rasterized_buildings[:, :, poly_no]]
                    building_median_height = np.median(building_ndsm)
                    polygons_with_heights.append((poly, building_median_height))
                vertices, faces = self._mesh_polygons_with_heights_(polygons_with_heights)
                vertices[:, 1] = self.image_visualization.shape[0] - vertices[:, 1]
                return vertices, faces
            else:
                print("Not yet implemendted")
                return None, None
        else:
            rasterized_buildings = raster_utils.rasterize_polygons(building_polygons, self.image_visualization.shape[0], self.image_visualization.shape[1], rasters_in_different_channels=False)
            larger_ndsm = raster_utils.resize_raster(self.ndsm, self.image_visualization.shape[: 2], interp_order=1)
            buildings_ndsm = rasterized_buildings * larger_ndsm
            vertices, faces = raster_utils.load_ndsm_as_mesh(buildings_ndsm)
            return vertices, faces


    def model_buildings(self, use_median_height_of_building=True, median_height_from_ndsm=True, model_save_dir="/tmp/saved_model", model_name="model"):
        vertices, faces = self.get_building_model(use_median_height_of_building, median_height_from_ndsm)
        file_utilities.save_texturized_faces(vertices, faces, self.image_visualization, model_save_dir, model_name)
        # file_utilities.save_obj(vertices, faces, "/tmp/saved_model.obj")

