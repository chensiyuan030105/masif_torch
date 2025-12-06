masif_opts = {}
exp = "full"

# Default directories
masif_opts["raw_pdb_dir"] = f"./data/masif_ppi_search/{exp}/raw_pdbs/"
masif_opts["pdb_chain_dir"] = f"./data/masif_ppi_search/{exp}/chain_pdbs/"
masif_opts["ply_chain_dir"] = f"./data/masif_ppi_search/{exp}/chain_surfaces/"
masif_opts["tmp_dir"] = "./tmp/"
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_{}.ply"

# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
masif_opts["compute_iface"] = True

# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True

# Coords params
masif_opts["radius"] = 12.0

# Neural network patch application specific parameters.
masif_opts["ppi_search"] = {}
masif_opts["ppi_search"]["training_list"] = "./data/masif_ppi_search/lists/n_10_training.txt"
masif_opts["ppi_search"]["testing_list"] = "./data/masif_ppi_search/lists/n_10_testing.txt"
masif_opts["ppi_search"]["max_shape_size"] = 200
masif_opts["ppi_search"]["max_distance"] = 12.0  # Radius for the neural network.
masif_opts["ppi_search"][
    "masif_precomputation_dir"
] = f"./data/masif_ppi_search/{exp}/precomputation_12A/"
masif_opts["ppi_search"]["feat_mask"] = [1.0] * 5
masif_opts["ppi_search"]["max_sc_filt"] = 1.0
masif_opts["ppi_search"]["min_sc_filt"] = 0.5
masif_opts["ppi_search"]["pos_surf_accept_probability"] = 1.0
masif_opts["ppi_search"]["pos_interface_cutoff"] = 1.0
masif_opts["ppi_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["ppi_search"]["cache_dir"] = f"./data/masif_ppi_search/{exp}/nn_models/sc05/cache/"
masif_opts["ppi_search"]["model_dir"] = f"./data/masif_ppi_search/{exp}/nn_models/sc05/all_feat/model_data/"
masif_opts["ppi_search"]["desc_dir"] = f"./data/masif_ppi_search/{exp}/descriptors/sc05/all_feat/"
masif_opts["ppi_search"]["gif_descriptors_out"] = f"./data/masif_ppi_search/{exp}/gif_descriptors/"

# Parameters for shape complementarity calculations.
masif_opts["ppi_search"]["sc_radius"] = 12.0
masif_opts["ppi_search"]["sc_interaction_cutoff"] = 1.5
masif_opts["ppi_search"]["sc_w"] = 0.25

# Parameters for training
masif_opts["ppi_search"]["learning_rate"] = 1e-3
masif_opts["ppi_search"]["epoch"] = 10000
masif_opts["ppi_search"]["save_epoch"] = 100
masif_opts["ppi_search"]["batch_size"] = 8