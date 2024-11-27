"""
The goal is to sample camera positions that:

- Are located within the room's layout (room_layout).
- Are not obstructed by objects in the scene (scene_grid).
- Maintain a distance from the objects to provide meaningful views.
"""


# Bedroom
path_in=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps
path_out=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_labels
cd scripts
# python create_camera_positions_trajectory.py --start-idx 0 --end-idx 6000 --path-in $path_in --path-out $path_out --num-samples-scene 40
python create_camera_positions.py --start-idx 0 --end-idx 6000 --path-in $path_in --path-out $path_out --num-samples-scene 40

# Living Rooms
# path_in=/path/to/data/3dfront/processed/living_room_without_lamps/
# path_out=/path/to/data/3dfront/processed/living_room_without_lamps_full_labels
# cd scripts
# python create_camera_positions_living.py --start-idx 0 --end-idx 6000 --path-in $path_in --path-out $path_out --num-samples-scene 40
