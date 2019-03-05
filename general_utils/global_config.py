import os
import numpy as np
from general_utils import file_utilities
import math
from datasets import DatasetMethods


TRAINABLE_IMAGE_SIZE = 512


class DatasetPath:
    SPACENET_MODEL_SAVE_PATH = "/playpen/jisan/Regular/PycharmProjects/Research/SVPRNet/logs/saved_model/spacenet"
    SVPRNET_MODEL_SAVE_PATH = "/playpen/jisan/Regular/PycharmProjects/Research/SVPRNet/logs/saved_model/svprnet"
    DATA_FUSION_MULTITASK_SAVE_PATH = "/playpen/jisan/Regular/PycharmProjects/Research/SVPRNet/logs/saved_model/datafusionmultitasknet"
    MSCOCO_MRCNN_WEIGHTS_PATH = "/playpen/jisan/Regular/PycharmProjects/Research/SVPRNet/logs/mscoco_mrcnn_weights/mask_rcnn_coco.h5"


class BostonDatasetPath:
    LIDAR_BASE = "/playpen/jisan/Projects/SVPRNet/svpr_net/dataset/single_view_parametric_reconstruction/LIDAR Dataset/LIDAR Images"
    LIDAR_DATASET_CSV = "/playpen/jisan/Projects/SVPRNet/svpr_net/dataset/single_view_parametric_reconstruction/LIDAR Dataset/2013_2014_usgs_post_sandy_ma_nh_ri_minmax.csv"

    OSM_BOSTON_DATA_PATH_WGS84 = "/playpen/jisan/Projects/SVPRNet/svpr_net/dataset/single_view_parametric_reconstruction/OSM Dataset/Boston_AOI_Buildings_wgs84.geojson"
    OSM_BOSTON_DATA_PATH_WGS84_UTM19 = "/playpen/jisan/Projects/SVPRNet/svpr_net/dataset/single_view_parametric_reconstruction/OSM Dataset/Boston_AOI_Buildings_wgs84_utm_19.geojson"
    BOSTON_BUILDING_POINTS_PATH = "/playpen/jisan/Projects/SVPRNet/svpr_net/dataset/single_view_parametric_reconstruction/OSM Dataset/boston_building_points.txt"
    BOSTON_BUILDING_STATS_PATH = "/playpen/jisan/Projects/SVPRNet/svpr_net/dataset/single_view_parametric_reconstruction/OSM Dataset/boston_building_stats.txt"

    SATELLITE_IMAGE_COUNT = 3

    @classmethod
    def satellite_image_paths(cls, sat_image_no):
        def find_tif(dir):
            tiffs = [item for item in os.listdir(dir) if item.lower().endswith("tif")]
            assert len(tiffs) == 1
            return os.path.join(dir, tiffs[0])

        assert 0 <= sat_image_no < cls.SATELLITE_IMAGE_COUNT
        pan_img_path = find_tif(
            "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/capture_{}/pan".format(sat_image_no))
        vnir_img_path = find_tif(
            "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/capture_{}/vnir".format(sat_image_no))
        return pan_img_path, vnir_img_path

    @classmethod
    def northup_satellite_image_paths(cls, sat_image_no):
        assert 0 <= sat_image_no < cls.SATELLITE_IMAGE_COUNT
        pan_img_path = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/capture_{}/north_up_images/pan_north_up.tif".format(sat_image_no)
        vnir_img_path = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/capture_{}/north_up_images/vnir_north_up.tif".format(sat_image_no)
        return pan_img_path, vnir_img_path

    @classmethod
    def pan_sharpened_image_path(cls, sat_image_no):
        assert 0 <= sat_image_no < cls.SATELLITE_IMAGE_COUNT
        img_path = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/capture_{}/pan_sharpened/pan_sharpened.tif".format(sat_image_no)
        return img_path

    @classmethod
    def satellite_mask_path(cls, sat_image_no, registration_level=0):
        assert 0 <= sat_image_no < cls.SATELLITE_IMAGE_COUNT and 0 <= registration_level <= 2
        mask_path = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/capture_{}/pan_aligned_mask/mask{}.tif".format(sat_image_no, "_{}".format(registration_level) if registration_level > 0 else "")
        return mask_path

    @classmethod
    def pan_registration_points_paths(cls, sat_image_no, registration_level=0):
        assert 0 <= sat_image_no < cls.SATELLITE_IMAGE_COUNT and 0 <= registration_level <= 2
        if registration_level == 0:
            return []
        points_path_base = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/capture_{}/pan_aligned_mask/registration_points_{}.txt"
        points_files_path = [points_path_base.format(sat_image_no, i) for i in range(1, registration_level + 1)]
        return points_files_path

    @classmethod
    def final_dataset_path(cls, dataset_type, sample_no):
        assert isinstance(sample_no, int)
        dataset_type = dataset_type.lower()
        assert dataset_type in ["train", "validation"]

        image_path_npz = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/trainable_ground_truths/{}/{}/sample_{}_image.npz".format(dataset_type, sample_no, sample_no)
        mask_path_npz = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/trainable_ground_truths/{}/{}/sample_{}_mask.npz".format(dataset_type, sample_no, sample_no)

        image_png = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/trainable_ground_truths/{}/{}/sample_{}_image.png".format(dataset_type, sample_no, sample_no)
        mask_png = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/trainable_ground_truths/{}/{}/sample_{}_mask.png".format(dataset_type, sample_no, sample_no)

        metadata_dict_path = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/trainable_ground_truths/{}/{}/sample_{}_metadata.pkl".format(dataset_type, sample_no, sample_no)

        if dataset_type == "train":
            return image_path_npz, mask_path_npz, image_png, mask_png, metadata_dict_path
        return image_path_npz, mask_path_npz, image_png, mask_png, metadata_dict_path

    @classmethod
    def get_training_set_size(cls):
        while True:
            try:
                return cls.TRAINING_SET_SIZE
            except:
                base_path = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/trainable_ground_truths/train"
                valid_items = [int(item) for item in os.listdir(base_path) if item.isdigit()]
                assert max(valid_items) == len(valid_items) - 1
                cls.TRAINING_SET_SIZE = len(valid_items)

    @classmethod
    def get_validation_set_size(cls):
        while True:
            try:
                return cls.VALIDATION_SET_SIZE
            except:
                base_path = "/playpen/jisan/Projects/SVPRNet/boston_imagery/cambridge/new/trainable_ground_truths/validation"
                valid_items = [int(item) for item in os.listdir(base_path) if item.isdigit()]
                assert max(valid_items) == len(valid_items) - 1
                cls.VALIDATION_SET_SIZE = len(valid_items)


class DataFusionDataset:
    DATASET_ROOT = "/playpen2/jisan/workspace/Datasets/Data Fusion Challenge"
    _SUB_PATHS = ["Track1", "Track1-MSI-1", "Track1-MSI-2", "Track1-MSI-3", "Track1-RGB", "Track1-Truth", "Trainable-Dataset"]

    _NDSM_LOG_PREDICTION_OFFSET_ = 10.0
    _NDSM_NORMALIZED_SCALING_FACTOR_ = 1.0

    NORMALIZE_PROXIMITY_HEATMAP_PER_BUILDING = True
    PROXIMITY_HEATMAP_TSDF_CUTOFF_PIXELS = 10
    PROXIMITY_HEATMAP_DISTINGUISH_ZERO_LEVEL_SET = True
    PROXIMITY_HEATMAP_USE_TSDF = True

    __ndsm_transforms__ = {"log": {"to_prediction_space": lambda x: np.log(x + DataFusionDataset._NDSM_LOG_PREDICTION_OFFSET_).astype(dtype=np.float32), "from_prediction_space": lambda x: (np.exp(x) - DataFusionDataset._NDSM_LOG_PREDICTION_OFFSET_).astype(dtype=np.float32)},
                           "normalized": {"to_prediction_space": lambda x: (x / DataFusionDataset._NDSM_NORMALIZED_SCALING_FACTOR_).astype(dtype=x.dtype), "from_prediction_space": lambda x: (x * DataFusionDataset._NDSM_NORMALIZED_SCALING_FACTOR_).astype(dtype=x.dtype)}}

    _ndsm_transform_to_use_ = "normalized"

    # Mean colors [128.25891907130307, 127.1716770853364, 125.7489284874508]
    # # Mean multispectral colors [277.5603922755865, 387.3471931696709, 485.3895646971505, 348.80886737913954, 341.36109570535103, 364.9330009459764, 542.3930773346107, 356.0600126165732]
    # Max color values [42294, 6103, 7283, 6627, 3802, 5531, 10138, 28196]
    # _LABEL_FREQUENCIES = {17: 0.01246735760945448, 2: 0.6439544894992556, 5: 0.15332551401722916, 6: 0.13973544049168765, 65: 0.011749266550089446, 9: 0.03876793183228373}

    BUILDING_FINAL_LABEL = 3
    GROUND_FINAL_LABEL = 1
    NO_CLASS_FINAL_LABEL = 4

    @staticmethod
    def get_continuous_semantic_label(semantic_labeled_image):
        mapping = {17: 0, 2: 1, 5: 2, 6: 3, 65: 4, 9: 5}  # 65 is mapped to ground!
        new_img = np.empty_like(semantic_labeled_image, dtype=np.uint8)
        for old_label, new_label in mapping.items():
            old_label_mask = np.equal(semantic_labeled_image, old_label)
            new_img[old_label_mask] = new_label
        return new_img

    @staticmethod
    def get_continuous_labels_frequencies():
        return [0.01246735760945448, 0.6439544894992556, 0.15332551401722916, 0.13973544049168765, 0.011749266550089446, 0.03876793183228373]

    @staticmethod
    def get_continuous_label_normalized_weights(no_weight_unknown_class=True):
        import math
        freq = DataFusionDataset.get_continuous_labels_frequencies()
        unnormalized_weights = [1.0 / item for item in freq]
        if no_weight_unknown_class:
            unnormalized_weights[DataFusionDataset.NO_CLASS_FINAL_LABEL] = 0.0
        unnormalized_weights = [math.sqrt(elm) for elm in unnormalized_weights]
        unnormalized_weights_sum = sum(unnormalized_weights)
        normalized_weights = [item / unnormalized_weights_sum for item in unnormalized_weights]
        return normalized_weights

    @staticmethod
    def get_image_id_from_fname(fname):
        ext_stripped = file_utilities.get_filename_extension(fname)[0]
        assert ext_stripped[-4] == '_'
        return ext_stripped[: -4]


    @staticmethod
    def get_ground_truth_directory():
        gt_dir = file_utilities.path_join(DataFusionDataset.DATASET_ROOT, DataFusionDataset._SUB_PATHS[5])
        return gt_dir


    @staticmethod
    def get_semantic_label_path(img_id):
        gt_dir = DataFusionDataset.get_ground_truth_directory()
        img_fname = img_id + "_CLS.tif"
        abs_path = file_utilities.path_join(gt_dir, img_fname)
        return abs_path

    @staticmethod
    def get_ndsm_path(img_id):
        gt_dir = DataFusionDataset.get_ground_truth_directory()
        img_fname = img_id + "_AGL.tif"
        abs_path = file_utilities.path_join(gt_dir, img_fname)
        return abs_path


    @staticmethod
    def get_building_instances_from_semantic_image(sem_img, building_label=None):
        if building_label is None:
            building_label = DataFusionDataset.BUILDING_FINAL_LABEL
        building_mask = np.equal(sem_img, building_label)
        return DatasetMethods.get_building_instances_from_building_mask(building_mask)

    @classmethod
    def get_dataset_size(cls, dataset_type):
        dataset_type = dataset_type.lower()
        assert dataset_type in ["train", "validation", "validation-all"]
        while True:
            try:
                cls.CACHED_DATASET_SIZE
            except:
                cls.CACHED_DATASET_SIZE = {}
            try:
                return cls.CACHED_DATASET_SIZE[dataset_type]
            except:
                base_path = file_utilities.path_join(file_utilities.path_join(cls.DATASET_ROOT, cls._SUB_PATHS[6]), dataset_type)
                valid_items = [int(item) for item in os.listdir(base_path) if item.isdigit()]
                assert max(valid_items) == len(valid_items) - 1
                cls.CACHED_DATASET_SIZE[dataset_type] = len(valid_items)


    # @staticmethod
    # def get_proximity_heatmap_image(proximity_heatmap, normalized=None):
    #     if normalized is None:
    #         normalized = DataFusionDataset.NORMALIZE_PROXIMITY_HEATMAP_PER_BUILDING
    #     n = proximity_heatmap.shape[0]
    #     assert n == proximity_heatmap.shape[1]
    #     img = np.zeros([n, n, 3], dtype=np.float32)
    #     img[:, :, 0] = np.where(proximity_heatmap >= 0.0, proximity_heatmap, 0.0)
    #     if not normalized:
    #         r_max = img[:, :, 0].max()
    #         if r_max > 0.0:
    #             img[:, :, 0] /= r_max
    #     img[:, :, 2] = np.where(proximity_heatmap < 0, -proximity_heatmap, 0.0)
    #     if not normalized:
    #         b_max = img[:, :, 2].max()
    #         if b_max > 0.0:
    #             img[:, :, 2] /= b_max
    #     return img

    @staticmethod
    def ndsm_to_prediction_space(ndsm):
        return DataFusionDataset.__ndsm_transforms__[DataFusionDataset._ndsm_transform_to_use_]["to_prediction_space"](ndsm)

    @staticmethod
    def ndsm_prediction_space_to_ndsm(ndsm_pred_space):
        return DataFusionDataset.__ndsm_transforms__[DataFusionDataset._ndsm_transform_to_use_]["from_prediction_space"](ndsm_pred_space)

    @staticmethod
    def final_dataset_path(dataset_type, sample_no):
        assert isinstance(sample_no, int)
        dataset_type = dataset_type.lower()
        assert dataset_type in ["train", "validation", "validation-all"]

        final_dataset_root = file_utilities.path_join(DataFusionDataset.DATASET_ROOT, DataFusionDataset._SUB_PATHS[6])
        base_dir = file_utilities.path_join(file_utilities.path_join(final_dataset_root, dataset_type), str(sample_no))

        image_path_tiff = file_utilities.path_join(base_dir, "sample_{}_image.tiff".format(sample_no))
        semantic_label_path_npz = file_utilities.path_join(base_dir, "sample_{}_semantic_label.npz".format(sample_no))
        ndsm_path_npz = file_utilities.path_join(base_dir, "sample_{}_ndsm.npz".format(sample_no))

        image_path_png = file_utilities.path_join(base_dir, "sample_{}_image.png".format(sample_no))
        masked_building_path_png = file_utilities.path_join(base_dir, "sample_{}_masked_building.png".format(sample_no))

        return image_path_tiff, semantic_label_path_npz, ndsm_path_npz, image_path_png, masked_building_path_png


class SpacenetDataset:
    BUILDING_FINAL_LABEL = 3
    GROUND_FINAL_LABEL = 1
    NO_CLASS_FINAL_LABEL = 4

    PROXIMITY_HEATMAP_USE_TSDF = DataFusionDataset.PROXIMITY_HEATMAP_USE_TSDF
    PROXIMITY_HEATMAP_TSDF_CUTOFF_PIXELS = DataFusionDataset.PROXIMITY_HEATMAP_TSDF_CUTOFF_PIXELS
    PROXIMITY_HEATMAP_DISTINGUISH_ZERO_LEVEL_SET = DataFusionDataset.PROXIMITY_HEATMAP_DISTINGUISH_ZERO_LEVEL_SET
    NORMALIZE_PROXIMITY_HEATMAP_PER_BUILDING = DataFusionDataset.NORMALIZE_PROXIMITY_HEATMAP_PER_BUILDING

    NUM_SEMANTIC_LABELS = len(DataFusionDataset.get_continuous_labels_frequencies())

    @staticmethod
    def get_continuous_label_normalized_weights(no_weight_unknown_class=True):
        freq = [0.0, 68.34017094, 0.0, 15.94842613, 15.71140292, 0.0]
        freq = np.array(freq, dtype=np.float64)
        bad_freq_mask = freq < 1e-4
        freq = freq + 1e-4
        unnormalized_weights = 1.0 / freq
        unnormalized_weights[bad_freq_mask] = 0.0
        if no_weight_unknown_class:
            unnormalized_weights[DataFusionDataset.NO_CLASS_FINAL_LABEL] = 0.0
        normalized_weights = unnormalized_weights / unnormalized_weights.sum()
        return normalized_weights.astype(dtype=np.float32)



# if __name__ == "__main__":
#     w1 = np.array(SpacenetDataset.get_continuous_label_normalized_weights(True))
#     w2 = np.array(SpacenetDataset.get_continuous_label_normalized_weights(False))
#
#     print(w1)
#     print(w2)