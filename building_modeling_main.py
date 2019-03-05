from general_utils import file_utilities
from building_reconstruction import building_modelling
from general_utils import misc_utils
from building_reconstruction.building_modeling_utils import CorrectionUtils


def model_main(params):
    image, gt_mask = params["image"], params["gt_mask"]
    predicted_ndsm = params["predicted_ndsm"]
    predicted_semantic_labels = params["predicted_semantic_labels"]
    predicted_proximity_heatmap = params["predicted_proximity_heatmap"]
    predicted_masks = params["masks"]
    building_heights = params["building_heights"]
    confidence = params["scores"]

    predicted_proximity_heatmap = CorrectionUtils.correct_raster_2d(predicted_proximity_heatmap)
    predicted_ndsm = CorrectionUtils.correct_raster_2d(predicted_ndsm)

    building_modelling.BuildingModeller(image, predicted_masks, confidence, building_heights, predicted_ndsm, predicted_proximity_heatmap).model_buildings()

    # print(image[:, :, [4, 2, 1]].min(), image[:, :, [4, 2, 1]].max())

    # misc_utils.display_numpy_image(misc_utils.format_image_for_visualization(image[:, :, [4, 2, 1]], clip_minimum_percentile=5, clip_maximum_percentile=95))
    # misc_utils.display_numpy_image(CorrectionUtils.correct_raster_2d(predicted_proximity_heatmap))


if __name__ == "__main__":
    res = file_utilities.pickle_load_object("/Users/crysoberil/Google Drive/Temp/saved_result.pkl")
    model_main(res)