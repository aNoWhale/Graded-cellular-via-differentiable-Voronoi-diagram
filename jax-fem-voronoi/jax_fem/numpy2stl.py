import os

import numpy as np
from stl import mesh


def generate_stl_from_matrix(matrix, threshold=0.5, cube_size=1,filename="output"):
    rows, cols = matrix.shape
    vertices = []
    faces = []

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] > threshold:
                base_x = i * cube_size
                base_y = j * cube_size
                base_z = 0
                cube_vertices = [
                    [base_x, base_y, base_z],
                    [base_x + cube_size, base_y, base_z],
                    [base_x + cube_size, base_y + cube_size, base_z],
                    [base_x, base_y + cube_size, base_z],
                    [base_x, base_y, base_z + cube_size],
                    [base_x + cube_size, base_y, base_z + cube_size],
                    [base_x + cube_size, base_y + cube_size, base_z + cube_size],
                    [base_x, base_y + cube_size, base_z + cube_size]
                ]

                vertex_offset = len(vertices)
                vertices.extend(cube_vertices)

                cube_faces = [
                    [0 + vertex_offset, 1 + vertex_offset, 2 + vertex_offset],
                    [0 + vertex_offset, 2 + vertex_offset, 3 + vertex_offset],
                    [4 + vertex_offset, 5 + vertex_offset, 6 + vertex_offset],
                    [4 + vertex_offset, 6 + vertex_offset, 7 + vertex_offset],
                    [0 + vertex_offset, 1 + vertex_offset, 5 + vertex_offset],
                    [0 + vertex_offset, 5 + vertex_offset, 4 + vertex_offset],
                    [1 + vertex_offset, 2 + vertex_offset, 6 + vertex_offset],
                    [1 + vertex_offset, 6 + vertex_offset, 5 + vertex_offset],
                    [2 + vertex_offset, 3 + vertex_offset, 7 + vertex_offset],
                    [2 + vertex_offset, 7 + vertex_offset, 6 + vertex_offset],
                    [3 + vertex_offset, 0 + vertex_offset, 4 + vertex_offset],
                    [3 + vertex_offset, 4 + vertex_offset, 7 + vertex_offset]
                ]

                faces.extend(cube_faces)

    vertices = np.array(vertices)
    faces = np.array(faces)

    cube_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube_mesh.vectors[i][j] = vertices[f[j]]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, '../../src/data/vtk')
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, f'{filename}.stl')
    cube_mesh.save(file_path)
    print(f"STL file saved as '{file_path}'")



