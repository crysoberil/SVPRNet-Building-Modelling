import gdal
import general_utils
from general_utils import process_execution_utils
import numpy as np
from general_utils import utm_utils


def __load_rpb(fid):
    rpb_data = dict()
    for line in iter(lambda: fid.readline().strip(), "END;"):
        key, value = line.split("=")
        key = key.rstrip()
        value = value.lstrip()

        if key == "END_GROUP":
            return rpb_data

        if key == "BEGIN_GROUP":
            rpb_data[value] = __load_rpb(fid)
        elif value.startswith('"'):
            rpb_data[key] = value[1:value.rfind('"')]
        elif value.startswith('('):
            values = []
            value = value[1:]
            at_end = False
            while True:
                for entry in value.split(","):
                    entry = entry.strip()
                    if entry:
                        pos = entry.find(")")
                        if pos != -1:
                            entry = entry[:pos]
                            at_end = True

                        values.append(float(entry))

                        if at_end:
                            break
                if at_end:
                    break

                value = fid.readline().strip()

            rpb_data[key] = np.array(values)
        else:
            pos = value.rfind(";")
            if pos == -1:
                pos = len(value)
            rpb_data[key] = float(value[:pos])

    return rpb_data


def get_utm_zone(rpb_file_path):
    with open(rpb_file_path, 'r') as fid:
        rpb_data = __load_rpb(fid)
    proj_data = rpb_data["IMAGE"]
    ref_lat = proj_data["latOffset"]
    ref_lon = proj_data["longOffset"]
    zone_number = utm_utils.latlon_to_zone_number(ref_lat, ref_lon)
    zone_letter = utm_utils.latitude_to_zone_letter(ref_lat)

    return zone_number, zone_letter


def get_gdal_geo_transform(f_path):
    return gdal.Open(f_path).GetGeoTransform()


def warp_raster(inp_path, out_path, proj4):
    general_utils.file_utilities.try_remove_file(out_path)
    command_base = "gdalwarp -t_srs \"{}\" \"{}\" \"{}\""
    command = command_base.format(proj4, inp_path, out_path)
    process_execution_utils.execute_process(command)


def apply_affine_transform(points2d, transformation):
    n = points2d.shape[0]
    points2d = points2d.astype(dtype=np.float64)
    points2d = np.transpose(points2d)
    ones = np.ones([1, n], dtype=np.float64)
    points3d = np.row_stack([points2d, ones])
    transformed = np.dot(transformation, points3d)
    transformed[0, :] /= transformed[2, :]
    transformed[1, :] /= transformed[2, :]
    transformed = transformed[: 2, :]
    transformed = np.transpose(transformed)
    transformed = transformed.astype(dtype=np.float32)
    return transformed


# def rpc_projection_wrt_image(coords, ref_image_with_rpc):
#     tmp_inp_file = "/tmp/rpc_proj_inp"
#     tmp_out_file = "/tmp/rpc_proj_out"
#     general_utils.file_utilities.try_remove_file(tmp_inp_file)
#     general_utils.file_utilities.try_remove_file(tmp_out_file)
#     cpp_utils.save_numpy_2d_array(coords, tmp_inp_file)
#     rpc_proj_command_base = "gdaltransform -i -rpc \"{}\" < \"{}\" > \"{}\""
#     rpc_proj_command = rpc_proj_command_base.format(ref_image_with_rpc, tmp_inp_file, tmp_out_file)
#     process_execution_utils.execute_process(rpc_proj_command)
#     converted = cpp_utils.load_numpy_2d_array(tmp_out_file, coords.shape[0], 3)
#     xs = converted[:, 0]
#     ys = converted[:, 1]
#     os.remove(tmp_inp_file)
#     os.remove(tmp_out_file)
#     return ys, xs