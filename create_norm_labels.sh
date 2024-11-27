cd scripts

in_dir=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_labels
out_dir=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full_labels_vertices
out_dir_norm=/root/data/3D-FRONT/3D-FRONT-processed/bedrooms_without_lamps_full/labels
python add_vertices_calc.py --in-dir $in_dir --out-dir $out_dir
python normalize_dataset.py --in-dir $out_dir --out-dir $out_dir_norm