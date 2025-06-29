import bpy
import bmesh
import mathutils
import os
import csv

def create_polyp_colon_intersection(colon_obj, polyp_obj):
    """
    1. Creates a new object that is the intersection of the polyp and the colon.
    2. Calculates the volume of that intersection.
    3. Finds the bounding box of that intersection in world coordinates.
    4. Creates a bounding box mesh object to visualize that region.
    """

    
    # ----------------------------------------------------
    # 2) DUPLICATE THE POLYP & ADD BOOLEAN INTERSECTION
    # ----------------------------------------------------
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    polyp_obj.select_set(True)
    bpy.context.view_layer.objects.active = polyp_obj
    
    # Duplicate polyp
    bpy.ops.object.duplicate()
    intersection_obj = bpy.context.active_object
    intersection_obj.name = "Polyp_Colon_Intersection"
    
    # Add a boolean modifier for intersection
    bool_mod = intersection_obj.modifiers.new(name="Colon_Intersect", type='BOOLEAN')
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = colon_obj
    
    # Apply the boolean modifier (converts to real mesh)
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)

    # ----------------------------------------------------
    # 3) CALCULATE VOLUME OF INTERSECTION
    # ----------------------------------------------------
    # Ensure we're in object mode before calculating
    bpy.ops.object.mode_set(mode='OBJECT')
    
    mesh = intersection_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = bm.calc_volume()
    bm.free()
    
    print(f"Volume of the polyp portion inside colon: {volume} (Blender units^3)")

    # ----------------------------------------------------
    # 4) GET BOUNDING BOX DIMENSIONS (WORLD SPACE)
    # ----------------------------------------------------
    # The bound_box property is given in local/object coordinates,
    # so we must transform each corner into world space.
    bb_corners_local = intersection_obj.bound_box
    world_matrix = intersection_obj.matrix_world
    
    # Transform each bounding-box corner into world coordinates
    bb_corners_world = [world_matrix @ mathutils.Vector(corner) for corner in bb_corners_local]
    
    xs = [v.x for v in bb_corners_world]
    ys = [v.y for v in bb_corners_world]
    zs = [v.z for v in bb_corners_world]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    dim_z = max_z - min_z
    
    print("Intersection bounding box (world space):")
    print(f"  X range: {min_x:.4f} to {max_x:.4f}, dimension = {dim_x:.4f}")
    print(f"  Y range: {min_y:.4f} to {max_y:.4f}, dimension = {dim_y:.4f}")
    print(f"  Z range: {min_z:.4f} to {max_z:.4f}, dimension = {dim_z:.4f}")

    # ----------------------------------------------------
    # 5) CREATE A BOUNDING BOX MESH OBJECT
    # ----------------------------------------------------
    # The bounding box is just a scaled cube. We'll place its origin
    # at the bounding box center, then scale it to match the dimensions.
    
    # Center of the bounding box
    center_x = (max_x + min_x) / 2.0
    center_y = (max_y + min_y) / 2.0
    center_z = (max_z + min_z) / 2.0
    
    # Create the cube
    bpy.ops.mesh.primitive_cube_add(location=(center_x, center_y, center_z))
    bbox_obj = bpy.context.active_object
    bbox_obj.name = "PolypInsideColon_BBox"
    
    # Scale the cube to match bounding box size
    bbox_obj.scale = (dim_x / 2.0, dim_y / 2.0, dim_z / 2.0)
    
    print("Bounding box object created: 'PolypInsideColon_BBox'")

    # Return the computed values
    return volume, dim_x, dim_y, dim_z

def cleanup_scene():
    """
    Remove objects from the scene given a list of names.
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_obj(filepath):
    """Imports an OBJ file and returns imported objects."""
    bpy.ops.import_scene.obj(filepath=filepath)
    return bpy.context.selected_objects

# ---------------------------
# Main Loop: Process Folder
# ---------------------------

# Get the directory of the currently open Blender file (the .blend file or script)
project_root = os.path.dirname(os.path.dirname(bpy.data.filepath))

# Define input/output folders relative to that
input_folder = os.path.join(project_root, "data", "sample_data", "mesh")
output_csv = os.path.join(project_root, "data", "annotations", "sample_polyp_intersection_results.csv")
results = []

for filename in os.listdir(input_folder):
    if not filename.lower().endswith('.obj'):
        continue
    
    print(f"Processing: {filename}")
    
    # Clear scene for new import
    cleanup_scene()

    # Import OBJ
    filepath = os.path.join(input_folder, filename)
    imported_objects = import_obj(filepath)

    # Identify colon and polyp objects
    colon, polyp = None, None
    for obj in imported_objects:
        if "MyCurveObject_CUMyCurveData" in obj.name:  
            colon = obj
        elif "Sphere" in obj.name:  
            polyp = obj

    if not (colon and polyp):
        print(f"Skipping {filename}: Missing colon/polyp")
        continue

    # Compute intersection
    res = create_polyp_colon_intersection(colon, polyp)
    
    # Unpack the returned values
    volume, x_dim, y_dim, z_dim = res
    
    # Scale the values as specified: dimensions * 10, volume * 1000
    scaled_volume = volume * 1000
    scaled_x = x_dim * 10
    scaled_y = y_dim * 10
    scaled_z = z_dim * 10

    file_name = os.path.basename(filepath)
    results.append([file_name, scaled_volume, scaled_x, scaled_y, scaled_z])

# Save results to CSV
if results:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Volume', 'Dim X', 'Dim Y', 'Dim Z'])
        writer.writerows(results)
    print(f"Saved {len(results)} results to {output_csv}")
else:
    print("No valid results to save.")

