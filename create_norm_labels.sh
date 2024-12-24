cd scripts

in_dir=/root/data/3D-FRONT/3D-FRONT-processed/cameras/bedrooms_without_lamps_full_labels_zuniform-h1.7-r4
out_dir=/root/data/3D-FRONT/3D-FRONT-processed/vertices/bedrooms_without_lamps_full_labels_vertices_zuniform-h1.7-r4
out_dir_norm=/root/data/3D-FRONT/3D-FRONT-processed/norm/bedrooms_without_lamps_full/labels_zuniform-h1.7-r4
python add_vertices_calc.py --in-dir $in_dir --out-dir $out_dir
python normalize_dataset.py --in-dir $out_dir --out-dir $out_dir_norm