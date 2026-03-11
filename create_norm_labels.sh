cd scripts

in_dir=/home/youngwoo/Desktop/2026-1/Research/dev/threed_front_rendering/output/Library/library_without_lamps_full_labels_orbit
out_dir=/home/youngwoo/Desktop/2026-1/Research/dev/threed_front_rendering/output/Library/library_without_lamps_full_labels_vertices_orbit
out_dir_norm=/home/youngwoo/Desktop/2026-1/Research/dev/threed_front_rendering/output/Library/library_without_lamps_full_labels_norm_orbit
python add_vertices_calc.py --in-dir $in_dir --out-dir $out_dir
python normalize_dataset.py --in-dir $out_dir --out-dir $out_dir_norm --max-coords 8.0,4.0,8.0
