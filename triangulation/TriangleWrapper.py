import os
from triangulation import process_execution_utils
from general_utils import file_utilities
import numpy as np
import shapely.geometry


def triangulate_multipolygon(vertices, polygons, triangle_program_path, temporary_directory):
    def get_poly_area(vertices, poly):
        poly_points = []
        for i in range(poly.shape[0]):
            assert poly[i][0] == poly[i - 1][1]
            poly_points.append(vertices[poly[i][0]].tolist())
        shapely_polygon = shapely.geometry.Polygon(poly_points)
        return shapely_polygon.area

    if len(polygons) == 1:
        all_poly_edges = polygons[0]
        inner_polys = []
    else:
        polys_with_area = [(get_poly_area(vertices, poly), poly) for poly in polygons]
        polys_with_area = sorted(polys_with_area, key=(lambda elm: (elm[0], elm[1].sum())), reverse=True)
        inner_polys = [poly_with_area[1] for poly_with_area in polys_with_area[1:]]
        tuple_of_all_polys = tuple([elm[1] for elm in polys_with_area])
        all_poly_edges = np.row_stack(tuple_of_all_polys)
    triangulation_vertices, triangulation_triangles = triangulate_polygon_with_holes(vertices, all_poly_edges, triangle_program_path, temporary_directory, holes=inner_polys)
    return triangulation_vertices, triangulation_triangles


def triangulate_polygon_with_holes(vertices, all_edges, triangle_program_path, temporary_directory, holes=[]):
    hole_points = __get_hole_random_points(holes, vertices, triangle_program_path, temporary_directory)
    file_utilities.ensure_folder(temporary_directory)
    input_poly_path = os.path.join(temporary_directory, "temppoly.poly")
    # Generate the poly file
    __create_input_polygon_file(input_poly_path, vertices, all_edges, hole_points)

    # Create the triangulations
    execution_command = "exec \"{}\" -p \"{}\"".format(triangle_program_path, input_poly_path)
    process_execution_utils.execute_process(execution_command)

    # Extract the triangulation verices and triangles
    vertices_path = os.path.join(temporary_directory, "temppoly.1.node")
    triangles_path = os.path.join(temporary_directory, "temppoly.1.ele")
    triangle_vertices, id_to_vertex_no = __extract_vertices(vertices_path)
    triangle_edges = __extract_triangles(triangles_path, id_to_vertex_no)

    # Remove temporary files
    os.remove(input_poly_path)
    os.remove(vertices_path)
    os.remove(triangles_path)
    os.remove(os.path.join(temporary_directory, "temppoly.1.poly"))
    return triangle_vertices, triangle_edges


def __get_hole_random_points(hole_polygons, vertices, triangle_program_path, temporary_directory):
    hole_points = []

    for hole_poly in hole_polygons:
        hole_vertices, hole_edges = triangulate_polygon_with_holes(vertices, hole_poly, triangle_program_path, temporary_directory, holes=[])
        random_triangle = hole_edges[0]
        inside_x, inside_y = sum(hole_vertices[random_triangle[i]] for i in range(3)) / 3.0
        hole_points.append([inside_x, inside_y])

    return hole_points


def __extract_triangles(f_path, id_to_vertex_no):
    with open(f_path, 'r') as f_in:
        num_of_triangles = int(f_in.readline().split()[0])
        triangles = np.empty([num_of_triangles, 3], dtype=np.int32)

        i = 0
        while True:
            line = f_in.readline()
            if line is None or len(line) == 0 or line.startswith('#'):
                break
            toks = [int(elm) for elm in line.split()[1: 4]]
            triangles[i] = [id_to_vertex_no[toks[0]], id_to_vertex_no[toks[1]], id_to_vertex_no[toks[2]]]
            i += 1

    return triangles


def __extract_vertices(f_path):
    with open(f_path, 'r') as f_in:
        num_of_verts = int(f_in.readline().split()[0])
        vertices = np.empty([num_of_verts, 2], dtype=np.float32)
        ids = []
        i = 0
        while True:
            line = f_in.readline()
            if line is None or len(line) == 0 or line.startswith('#'):
                break
            toks = line.split()[: 3]
            ids.append(int(toks[0]))
            vertices[i, 0] = float(toks[1])
            vertices[i, 1] = float(toks[2])
            i += 1

    id_to_vertex_no = [None for _ in range(max(ids) + 1)]
    for i in range(num_of_verts):
        id_to_vertex_no[ids[i]] = i

    return vertices, id_to_vertex_no


def __create_input_polygon_file(target_path, vertices, edges, hole_points):
    # TODO handle holes
    edges = edges + 1  # 1 indexed now
    num_of_verts = vertices.shape[0]
    num_of_edges = edges.shape[0]
    with open(target_path, 'w') as poly_file:
        poly_file.write("{} 2 0 1\n".format(num_of_verts))
        for i, (x, y) in enumerate(vertices, 1):
            poly_file.write("{} {} {} 1\n".format(i, x, y))

        poly_file.write("{} 1\n".format(num_of_edges))
        for i, (p1, p2) in enumerate(edges, 1):
            poly_file.write("{} {} {} 1\n".format(i, p1, p2))

        poly_file.write("{}\n".format(len(hole_points)))
        for i, (x, y) in enumerate(hole_points, 1):
            poly_file.write("{} {} {}\n".format(i, x, y))
