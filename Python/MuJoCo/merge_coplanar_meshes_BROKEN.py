import pymeshlab
import os
import numpy as np

# Load the STL file
def load_stl(file_path):
    return pymesh.load_mesh(file_path)

# Merge coplanar meshes
def merge_coplanar_meshes(mesh):
    tetgen = pymesh.tetgen()
    tetgen.points = mesh.vertices
    tetgen.triangles = mesh.faces
    tetgen.merge_coplanar = True
    tetgen.run()
    return tetgen.mesh

# Save the modified mesh
def save_mesh(mesh, output_path):
    pymesh.save_mesh(output_path, mesh)

# Main function
def main(input_file, output_folder):
    # Load the input STL file
    mesh = load_stl(input_file)

    # Merge coplanar meshes
    merged_mesh = merge_coplanar_meshes(mesh)

    # Prepare the output file path
    output_file_name = os.path.basename(input_file)
    output_file_path = os.path.join(output_folder, output_file_name)

    # Save the modified mesh
    save_mesh(merged_mesh, output_file_path)

    print(f"Merged mesh saved to: {output_file_path}")

if __name__ == "__main__":
    input_file = r"E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\meshes\base.STL"  # Replace with your STL file path
    output_folder = r"E:\github\Re-inforcement\Spider\Spider_Assembly_fineMesh_frictionDamp\meshes_coplanarMerged"  # Replace with your output folder path
    main(input_file, output_folder)
