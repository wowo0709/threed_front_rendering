# 1. Pre-process dataset following ATISS
# NOTE 1: Pass --dataset_filtering ["threed_front_bedroom", "threed_front_livingroom", "threed_front_diningroom", "threed_front_library"] to preprocess diferent type of scene
# NOTE 2: Change --room_side argument to 6.2m from 3.1m (default) for "threed_front_livingroom", "threed_front_diningroom"
# NOTE 3: After initial run, you can use cached "/YOUR_PATH/threed_front.pkl" pickle file by: PATH_TO_SCENES="/YOUR_PATH/threed_front.pkl" python preprocess_data.py ... (check YOUR_PATH in utils.py)
path_to_output_dir=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps
path_to_3d_front_dataset_dir=/root/data/3D-FRONT/3D-FRONT
path_to_3d_future_dataset_dir=/root/data/3D-FRONT/3D-FUTURE-model
# path_to_3d_future_model_info=/root/dev/threed_front_rendering/demo/model_info.json # model info w/o wall, door, lamp, etc.
path_to_3d_future_model_info=/root/data/3D-FRONT/3D-FUTURE-model/model_info.json     # model info w/ wall,l door, lamp, etc.
path_to_floor_plan_texture_images=/root/dev/threed_front_rendering/demo/floor_plan_texture_images_single_custom
cd scripts
xvfb-run -a python preprocess_data.py $path_to_output_dir $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_future_model_info $path_to_floor_plan_texture_images --dataset_filtering threed_front_bedroom --without_lamps

# Log
# Loading dataset with 4041 rooms
# 197it [10:18,  3.40s/it]libpng warning: iCCP: known incorrect sRGB profile
# libpng warning: iCCP: known incorrect sRGB profile
# 700it [35:08,  2.74s/it]libpng warning: iCCP: known incorrect sRGB profile
# libpng warning: iCCP: known incorrect sRGB profile
# 1752it [1:27:29,  3.05s/it]libpng warning: iCCP: known incorrect sRGB profile
# libpng warning: iCCP: known incorrect sRGB profile
# 3036it [2:30:02,  2.91s/it]libpng warning: iCCP: known incorrect sRGB profile
# libpng warning: iCCP: known incorrect sRGB profile
# 4041it [3:19:26,  2.96s/it]

# 2. Create camera coordinates
# bash create_camera_positions.sh

# 3. Normalize labels for rendering
# bash create_norm_labels.sh

# 4. Render the dataset
# bash render_threed_front.sh
