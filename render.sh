cd scripts

cam_path_in=/root/desktop/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps
cam_path_out=/root/desktop/tmp/3D-FRONT-processed/bedrooms_without_lamps_full_labels_zuniform-h1.7-r4
start_idx=0
end_idx=5

vert_path_out=/root/desktop/tmp/3D-FRONT-processed/bedrooms_without_lamps_full_labels_vertices

norm_path_out=/root/desktop/tmp/3D-FRONT-processed/bedrooms_without_lamps_full/labels

path_to_3d_front_dataset_dir=/root/desktop/3D-FRONT/3D-FRONT
path_to_3d_future_dataset_dir=/root/desktop/3D-FRONT/3D-FUTURE-model
path_to_3d_front_texture=/root/desktop/3D-FRONT/3D-FRONT-texture
path_cc_textures=/root/desktop/3D-FRONT/3D-FRONT-processed/blender/cc_textures
main_out_dir=/root/desktop/tmp/3D-FRONT-processed/bedrooms_without_lamps_full_raw/raw_256_depth_normal_noflip_vmax20_raw_zuniform-h1.7-r4
img_resolution=256
# img_out_dir=/root/desktop/tmp/3D-FRONT-processed/bedrooms_without_lamps_full_images/images_256
# depth_out_dir=/root/desktop/tmp/3D-FRONT-processed/bedrooms_without_lamps_full_images/depths_256
out_dir=/root/desktop/tmp/3D-FRONT-processed/bedrooms_without_lamps_full_images/images_256_depth_normal_noflip_vmax20_raw_zuniform-h1.7-r4

python create_camera_positions.py --start-idx $start_idx --end-idx $end_idx --path-in $cam_path_in --path-out $cam_path_out --num-samples-scene 40

python add_vertices_calc.py --in-dir $cam_path_out --out-dir $vert_path_out

python normalize_dataset.py --in-dir $vert_path_out --out-dir $norm_path_out

cd ../BlenderProc-main

for scene_idx in $(seq $start_idx $((end_idx - 1)))
do
echo "Processing scene index: $scene_idx"
blenderproc run examples/datasets/front_3d_with_improved_mat_traj_same/main.py $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_front_texture $path_cc_textures $vert_path_out $main_out_dir $scene_idx --img_resolution $img_resolution
# blenderproc debug examples/datasets/front_3d_with_improved_mat_traj_same/main.py $path_to_3d_front_dataset_dir $path_to_3d_future_dataset_dir $path_to_3d_front_texture $path_cc_textures $vert_path_out $main_out_dir $scene_idx --img_resolution $img_resolution
for frame_idx in $(seq 0 39)
do
blenderproc vis hdf5 $main_out_dir --keys colors normals depth --rgb_keys colors normals --depth_keys depth --save $out_dir --scene_idx $scene_idx --frame_idx $frame_idx --path_labels $vert_path_out
# blenderproc vis hdf5 $main_out_dir --flip=true --keys depth --save $depth_out_dir --scene_idx $scene_idx --frame_idx $frame_idx --path_labels $vert_path_out
done
done
