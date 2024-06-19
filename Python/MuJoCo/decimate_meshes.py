import os
import trimesh
import pymeshlab


def decimate_mesh(input_file, target_faces=1000):
    # Load the mesh using trimesh
    mesh = trimesh.load(input_file)

    # Create a new MeshSet
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))

    # Apply decimation
    if 'tibia' in input_file:
        pass
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces,
                                                    preservenormal=True,
                                                    preserveboundary=True,
                                                    # qualitythr=0.5,
                                                    # preserveboundary = True,
                                                    boundaryweight = 1000.0, planarweight = 0.01)
    else:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces,
                                                    preservenormal=True,
                                                    preserveboundary=True,
                                                    # qualitythr=0.5,
                                                    # preserveboundary = True,
                                                    boundaryweight=1000.0, planarweight = 0.01)

    # Get the decimated mesh
    decimated_mesh = ms.current_mesh()

    # Convert decimated mesh back to trimesh format
    decimated_trimesh = trimesh.Trimesh(vertices=decimated_mesh.vertex_matrix(), faces=decimated_mesh.face_matrix())

    return decimated_trimesh


def process_folder(input_folder, output_folder, target_faces=1000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.STL'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            decimated_mesh = decimate_mesh(input_path, target_faces)
            decimated_mesh.export(output_path)
            print(f"Processed {filename} and saved to {output_path}")


# Example usage
input_folder = r'E:\github\Re-inforcement\Spider\Spider_Assembly_coarseMesh\meshes'
output_folder = r'E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\meshes_decimatedMuchMore'
process_folder(input_folder, output_folder, target_faces=100)
