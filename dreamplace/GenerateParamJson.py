import os.path as osp
import json

def create_param_json(netlist_dir,param_dir,ourmodel=False):
    param = {}
    param["lef_input"] = [osp.join(netlist_dir,"ispd19_test1.input.lef")]
    param["def_input"] = osp.join(netlist_dir,"ispd19_test1.input.def")
    # param["verilog_input"] = osp.join(netlist_dir,"top.v")
    param["gpu"] = 1
    param["num_bins_x"] = 1024
    param["num_bins_y"] = 1024
    param["global_place_stages"] = [{"num_bins_x" : 1024, "num_bins_y" : 1024, "iteration" : 1000, "learning_rate" : 0.01, "wirelength" : "weighted_average", "optimizer" : "nesterov"}]
    if ourmodel:
        param["global_place_stages"] = [{"num_bins_x" : 1024, "num_bins_y" : 1024, "iteration" : 400, "learning_rate" : 0.01, "learning_rate_decay":0.97,"wirelength" : "weighted_average", "optimizer" : "adam"}]
    param["target_density"] = 0.65
    param["density_weight"] = 8e-5
    param["gamma"] = 4.0
    param["random_seed"] = 1000
    param["ignore_net_degree"] = 100
    param["enable_fillers"] = 1
    param["gp_noise_ratio"] = 0.025
    param["global_place_flag"] = 1
    param["legalize_flag"] = 1
    param["detailed_place_flag"] = 1
    param["detailed_place_engine"] = "thirdparty/ntuplace_4dr"
    param["detailed_place_command"] = "-nolegal -nodetail"
    param["stop_overflow"] = 0.07
    param["dtype"] = "float32"
    param["plot_flag"] = 0
    param["random_center_init_flag"] = 1
    param["sort_nets_by_degree"] = 0
    param["num_threads"] = 8
    param["sol_file_format"] = "DEF"
    json_data = json.dumps(param).replace(", ",",\r    ")
    with open(param_dir,"w") as f:
        f.write(json_data)

def creat_param():
    param = {}
    param["gpu"] = 1
    param["num_bins_x"] = 512
    param["num_bins_y"] = 512
    param["global_place_stages"] = [{"num_bins_x" : 512, "num_bins_y" : 512, "iteration" : 1000, "learning_rate" : 0.01, "learning_rate_decay":0.97,"wirelength" : "weighted_average", "optimizer" : "nesterov"}]
    param["target_density"] = 0.0
    param["density_weight"] = 8e-5
    param["gamma"] = 4.0
    param["random_seed"] = 1000
    param["ignore_net_degree"] = 100
    param["enable_fillers"] = 1
    param["gp_noise_ratio"] = 0.025
    param["global_place_flag"] = 0
    param["legalize_flag"] = 0
    param["detailed_place_flag"] = 0
    param["detailed_place_engine"] = "thirdparty/ntuplace_4dr"
    param["detailed_place_command"] = "-nolegal -nodetail"
    param["stop_overflow"] = 0.07
    param["dtype"] = "float32"
    param["plot_flag"] = 0
    param["random_center_init_flag"] = 1
    param["sort_nets_by_degree"] = 0
    param["num_threads"] = 8
    param["sol_file_format"] = "DEF"
    return param