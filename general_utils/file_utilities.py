import os
import gdal
import osr
from entities import misc_entities
import numpy as np
import csv
import rasterio
import rasterio.windows
import pickle
import tifffile
import scipy.misc


def ensure_folder(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as exc:  # Guard against race condition
            print("Directory could not be created: " + dir)


def try_remove_file(f_path):
    try:
        os.remove(f_path)
    except OSError:
        pass


def ensure_parent_folder(file):
    parent_folder = get_parent_folder_and_file(file)[0]
    ensure_folder(parent_folder)


def get_filename_extension(f_path):
    last_seperator_idx = f_path.rfind(os.sep)
    if last_seperator_idx >= 0:
        f_path = f_path[last_seperator_idx + 1:]
    last_period_idx = f_path.rfind('.')
    assert last_period_idx != -1
    file_name = f_path[: last_period_idx]
    file_ext = f_path[last_period_idx + 1:]
    return file_name, file_ext


def path_join(base_path, file_name):
    return os.path.join(base_path, file_name)


def append_to_file_name(f_path, extension_str):
    fname, ext = get_filename_extension(f_path)
    return fname + extension_str + '.' + ext


def get_csv_reader(file_path, delimiter=','):
    with open(file_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        rows = [row for row in csv_reader]
    return rows


def get_geotiff_affine_transorm(geotiff_path):
    if not os.path.isfile(geotiff_path):
        return None
    try:
        img_src = rasterio.open(geotiff_path)
        affine = img_src.transform
        return affine
    except:
        return None


# def get_geotiff_transorm_matrix(geotiff_path):
#     img_src = gdal.Open(geotiff_path)
#     coeffs = img_src.GetGeoTransform()
#     transform = [[coeffs[1], coeffs[2], coeffs[0]], [coeffs[5], coeffs[4], coeffs[3]], [0, 0, 1]]
#     transform = np.array(transform, dtype=np.float64)
#     return transform


# def get_geotiff_inverse_transofm_matrix(geotiff_path):
#     forward_transform = get_geotiff_transorm_matrix(geotiff_path)
#     r = forward_transform[: 2, : 2]
#     t = forward_transform[: 2, 2][:, np.newaxis]
#     inverse_transform = np.zeros_like(forward_transform)
#     r_inv = np.linalg.pinv(r)
#     inverse_transform[: 2, : 2] = r_inv
#     inverse_transform[: 2, 2] = np.dot(r_inv, t).ravel()
#     inverse_transform[2, 2] = 1.0
#     return inverse_transform


def read_geotiff_gdal(img_path, prefererred_channels=None, supress_3rd_dim_if_possible=True, replace_nans_with=None, minimum_value=None, load_image=True, ignore_skewness=False):
    def _get_invalid_value_maps(dtype):
        if dtype == np.float32 or dtype == np.float64:
            return -9999.0
        if dtype == np.float64 or dtype == np.float64:
            return -9999
        if dtype == np.bool:
            return False
        return None

    img_src = gdal.Open(img_path)
    if load_image:
        if img_src.RasterCount > 1 or not(supress_3rd_dim_if_possible):
            target_rasters = list(range(1, img_src.RasterCount + 1)) if prefererred_channels is None else prefererred_channels
            raster_bands = [np.array(img_src.GetRasterBand(band_no).ReadAsArray()) for band_no in target_rasters]
            img = np.array(raster_bands)  # Band major axis
            img = np.transpose(img, axes=(list(range(1, len(img.shape))) + [0]))
        else:
            img = np.array(img_src.GetRasterBand(1).ReadAsArray())

        invalid_replace_by = _get_invalid_value_maps(img.dtype)
        if invalid_replace_by is not None:
            img = np.where(np.isfinite(img), img, -9999.0)
    else:
        img = None

    x_offset, x_res, x_skew, y_offset, y_skew, y_res = img_src.GetGeoTransform()
    if not ignore_skewness and not (x_skew == 0 and y_skew == 0):
        print("Warning: Skewed satellite imge. Skipping.")
        return None
    img_h, img_w = img_src.RasterYSize, img_src.RasterXSize
    x_left = x_offset if x_res > 0 else x_offset + x_res * img_w
    y_top = y_offset if y_res < 0 else y_offset + y_res * img_h

    if load_image:
        if replace_nans_with is not None:
            img[np.isnan(img)] = replace_nans_with

        if minimum_value is not None:
            img[np.less(img, minimum_value)] = minimum_value

    tiff_img_entity = misc_entities.ImageEntity(img, x_left, y_top, abs(x_res), abs(y_res), img_h, img_w)
    return tiff_img_entity


def get_parent_folder_and_file(abs_path):
    last_seperator = abs_path.rfind(os.sep)
    assert last_seperator != -1
    parent_folder = abs_path[: last_seperator]
    file_name = abs_path[last_seperator + 1:]
    return parent_folder, file_name


def save_geotiff(output_file_path, image, transform, zone_number, zone_letter):
    dtypes = {np.uint8: gdal.GDT_Byte,
              np.uint16: gdal.GDT_UInt16,
              np.uint32: gdal.GDT_UInt32,
              np.int8: gdal.GDT_Byte,
              np.int16: gdal.GDT_Int16,
              np.int32: gdal.GDT_Int32,
              np.float32: gdal.GDT_Float32,
              np.double: gdal.GDT_Float64}

    dtype = dtypes[image.dtype.type]
    driver = gdal.GetDriverByName("GTiff")
    if len(image.shape) == 2:
        dst_ds = driver.Create(
            output_file_path, image.shape[1], image.shape[0], 1, dtype,
            ["COMPRESS=LZW", "INTERLEAVE=BAND"])
        dst_ds.GetRasterBand(1).WriteArray(image)
    else:
        dst_ds = driver.Create(
            output_file_path, image.shape[1], image.shape[0], image.shape[2],
            dtype, ["COMPRESS=LZW", "INTERLEAVE=BAND"])
        for i in range(image.shape[2]):
            dst_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])
    dst_ds.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetUTM(zone_number, zone_letter >= "N")
    dst_ds.SetProjection(srs.ExportToWkt())


def save_geotiff_with_reference(save_path, data, reference_image_path):
    dtypes = {np.uint8: gdal.GDT_Byte,
              np.uint16: gdal.GDT_UInt16,
              np.uint32: gdal.GDT_UInt32,
              np.int8: gdal.GDT_Byte,
              np.int16: gdal.GDT_Int16,
              np.int32: gdal.GDT_Int32,
              np.float32: gdal.GDT_Float32,
              np.double: gdal.GDT_Float64}

    dtype = dtypes[data.dtype.type]

    driver = gdal.GetDriverByName("GTiff")
    if len(data.shape) == 2:
        dst_ds = driver.Create(
            save_path, data.shape[1], data.shape[0], 1, dtype,
            ["COMPRESS=LZW", "INTERLEAVE=BAND"])
        dst_ds.GetRasterBand(1).WriteArray(data)
    else:
        dst_ds = driver.Create(
            save_path, data.shape[1], data.shape[0], data.shape[2],
            dtype, ["COMPRESS=LZW", "INTERLEAVE=BAND"])
        for i in range(data.shape[2]):
            dst_ds.GetRasterBand(i + 1).WriteArray(data[:, :, i])

    ref_img = gdal.Open(reference_image_path)

    ref_transform = list(ref_img.GetGeoTransform())
    x_scaled = data.shape[1] / ref_img.RasterXSize
    y_scaled = data.shape[0] / ref_img.RasterYSize
    assert isinstance(x_scaled, int) and x_scaled == y_scaled
    ref_transform[1] /= x_scaled
    ref_transform[5] /= y_scaled
    dst_ds.SetGeoTransform(tuple(ref_transform))

    ref_wkt = ref_img.GetProjection()
    dst_ds.SetProjection(ref_wkt)


def save_compressed_numpy_array(ndarray, f_path):
    np.savez_compressed(f_path, array=ndarray)


def load_compressed_numpy_array(f_path):
    if not os.path.isfile(f_path):
        return None
    try:
        return np.load(f_path)["array"]
    except:
        return None


def read_geotiff_window(img_path, row_start, row_end_exl, col_start, col_end_exl, preferred_channels=None):
    window = rasterio.windows.Window(col_start, row_start, col_end_exl - col_start, row_end_exl - row_start)
    target_rasters = list(range(1, gdal.Open(img_path).RasterCount + 1)) if preferred_channels is None else preferred_channels
    rasters = []
    with rasterio.open(img_path) as src:
        for band_no in target_rasters:
            raster = src.read(band_no, window=window)
            rasters.append(raster)
    rasters = np.array(rasters)
    rasters = np.transpose(rasters, axes=(list(range(1, len(rasters.shape))) + [0]))
    return rasters


def pickle_save_object(object, f_path):
    with open(f_path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_object(f_path):
    with open(f_path, "rb") as f_in:
        obj = pickle.load(f_in)
    return obj


def read_tiff_with_tifffile(fpath):
    img = tifffile.imread(fpath)
    return img


def write_tiff_with_tifffile(img, fpath):
    tifffile.imwrite(fpath, img, dtype=img.dtype)


def save_obj(vertices, faces, save_fpath):
    with open(save_fpath, 'w') as fout:
        fout.write("# {} vertices\n".format(vertices.shape[0]))
        fout.write("# {} faces\n".format(faces.shape[0]))

        for v in vertices:
            fout.write("v {:.8f} {:.8f} {:.8f}\n".format(*v))

        for f1, f2, f3 in faces:
            f1 += 1
            f2 += 1
            f3 += 1
            fout.write("f {}/{}/{} {}/{}/{} {}/{}/{}\n".format(f1, f1, f1, f2, f2, f2, f3, f3, f3))


def save_texturized_faces(vertices, faces, texture_image, save_dir, base_name):
    obj_file = path_join(save_dir, "{}.obj".format(base_name))
    mtl_file = path_join(save_dir, "{}.mtl".format(base_name))
    texture_file = path_join(save_dir, "{}.png".format(base_name))
    ensure_folder(save_dir)
    scipy.misc.imsave(texture_file, texture_image)

    texture_dim = np.array(texture_image.shape[: 2], dtype=np.float32)

    with open(obj_file, 'w') as fout:
        fout.write("# {} vertices\n".format(vertices.shape[0]))
        fout.write("# {} faces\n".format(faces.shape[0]))
        fout.write("mtllib {}\n".format(get_parent_folder_and_file(mtl_file)[1]))

        for v in vertices:
            fout.write("v {:.8f} {:.8f} {:.8f}\n".format(*v))

        for uv in vertices:
            uv = uv[: 2] / texture_dim
            fout.write("vt {:.8f} {:.8f}\n".format(*uv))
            # fout.write("vt {:.8f} {:.8f}\n".format(10, 10))

        for f1, f2, f3 in faces:
            f1 += 1
            f2 += 1
            f3 += 1
            fout.write("f {}/{}/{} {}/{}/{} {}/{}/{}\n".format(f1, f1, f1, f2, f2, f2, f3, f3, f3))

    with open(mtl_file, 'w') as fout:
        fout.write("newmtl facetextures\n" + "Ka 0.200000 0.200000 0.200000\n" + "Kd 1.000000 1.000000 1.000000\n" + "Ks 1.000000 1.000000 1.000000\n" + "Tr 1.000000\n" + "illum 2\n" + "Ns 0.000000\n" + "map_Ka " + get_parent_folder_and_file(texture_file)[1] + "\nmap_Kd " + get_parent_folder_and_file(texture_file)[1] + '\n')
