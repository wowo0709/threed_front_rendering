# Bedrooms diverse views
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate blender


cd BlenderProc-main

# # Download cc textures
# path_cc_textures=/root/data/3D-FRONT/3D-FRONT-processed/blender/cc_textures
# blenderproc run blenderproc/scripts/download_cc_textures.py $path_cc_textures

path_cc_textures=/root/data/3D-FRONT/3D-FRONT-processed/blender/cc_textures
path_to_3d_front_dataset_dir=/root/data/3D-FRONT/3D-FRONT/
path_to_3d_future_dataset_dir=/root/data/3D-FRONT/3D-FUTURE-model/
path_to_3d_future_model_info=/root/data/3D-FRONT/3D-FUTURE-model/model_info.json
path_to_3d_front_texture=/root/data/3D-FRONT/3D-FRONT-texture
path_labels=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_labels_vertices
outdir=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_raw/raw_256
outdir_img=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_images/images_256
img_resolution=256

# for scene_idx in {0..6000}
# do
# echo "Processing scene index: $scene_idx"
# blenderproc run examples/datasets/front_3d_with_improved_mat_traj_same/main.py $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_front_texture $path_cc_textures $path_labels $outdir $scene_idx --img_resolution $img_resolution
# for frame_idx in {0..39}
# do
# blenderproc vis hdf5 $outdir --flip=true --keys colors --save $outdir_img --scene_idx $scene_idx --frame_idx $frame_idx --path_labels $path_labels
# done
# done

outdir=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_raw/raw_256_depth_normal_noflip_vmax20_raw
outdir_img=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_images/images_256_depth_normal_noflip_vmax20_raw
# 
# outdir=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_raw/test/test1
# outdir_img=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_images/test/test1

scene_idx=2
blenderproc run examples/datasets/front_3d_with_improved_mat_traj_same/main.py $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_front_texture $path_cc_textures $path_labels $outdir $scene_idx --img_resolution $img_resolution
for frame_idx in {0..39}
do
blenderproc vis hdf5 $outdir --keys colors normals depth --rgb_keys colors normals --depth_keys depth --save $outdir_img --scene_idx $scene_idx --frame_idx $frame_idx --path_labels $path_labels
done



# # Living room diverse views
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate blender
# cd BlenderProc-main
# # Download cc textures
# path_cc_textures=/path/to/data/3dfront/processed/blender/cc_textures
# # blenderproc run blenderproc/scripts/download_cc_textures.py $path_cc_textures

# path_to_3d_front_dataset_dir=/path/to/data/3D-FRONT/
# path_to_3d_future_dataset_dir=/path/to/3D-FUTURE-v2/3D-FUTURE/
# path_to_3d_future_model_info=/path/to/code/ATISS/demo/model_info.json
# path_to_3d_front_texture=/path/to/3D-FRONT-texture
# outdir=/path/to/data/3dfront/processed/living_room_without_lamps_full_raw/raw
# path_labels=/path/to/data/3dfront/processed/living_room_without_lamps_full_labels_vertices
# outdir_img=/path/to/data/3dfront/processed/living_room_without_lamps/images
# img_resolution=256
# for scene_idx in {0..6000}
# do
# blenderproc run examples/datasets/front_3d_with_improved_mat_traj_same_living/main.py $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_front_texture $path_cc_textures $path_labels $outdir $scene_idx --img_resolution $img_resolution
# for frame_idx in {0..39}
# do
# blenderproc vis hdf5 $outdir --flip=true --keys colors --save $outdir_img --scene_idx $scene_idx --frame_idx $frame_idx --path_labels $path_labels
# done
# done



# After creating hdf5 files, create .png files with visHdf5Files.py
# Ex. blenderproc vis hdf5 /root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_raw/raw --scene_idx 0 --frame_idx 0 --save /root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_images/images_512