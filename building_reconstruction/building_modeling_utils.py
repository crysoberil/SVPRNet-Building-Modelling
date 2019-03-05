import numpy as np


class CorrectionUtils:
    @staticmethod
    def correct_raster_2d(raster, min_percentile=1.0, max_percentile=99.0):
        lower_percentile_bound = np.percentile(raster, min_percentile)
        upper_percentile_bound = np.percentile(raster, max_percentile)
        raster = np.clip(raster, lower_percentile_bound, upper_percentile_bound)
        return raster