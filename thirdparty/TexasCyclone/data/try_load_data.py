import torch
import sys, os
sys.path.append(os.path.abspath('.'))
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_dis_deflect
from data.pretrain import load_pretrain_data


if __name__ == '__main__':
    # netlist = netlist_from_numpy_directory('test/dataset1/large-noclu', save_type=2)
    # netlist = netlist_from_numpy_directory('../../Placement-datasets/dac2012/superblue2', save_type=2)
    # print(netlist.original_netlist.graph)
    # print(netlist.graph)
    # print(netlist.graph.nodes['cell'].data['ref_pos'])
    # print(netlist.cell_flow.flow_edge_indices)
    #
    # movable_edge_dis = torch.tensor([
    #     650,
    #     500,
    #     850,
    #     500,
    #     450,
    # ], dtype=torch.float32)
    # movable_edge_deflect = torch.tensor([
    #     0.1 * torch.pi,
    #     -0.25 * torch.pi,
    #     0.5 * torch.pi,
    #     0.25 * torch.pi,
    #     0.0 * torch.pi,
    # ], dtype=torch.float32)
    # layout, discrep = layout_from_netlist_dis_deflect(
    #     netlist,
    #     movable_edge_dis,
    #     movable_edge_deflect
    # )
    # print(discrep.data)
    # print(layout.cell_pos.numpy())
    # print(layout.netlist.graph.nodes['cell'].data['ref_pos'].cpu().numpy())
    # print(layout.cell_pos.numpy() - layout.netlist.graph.nodes['cell'].data['ref_pos'].cpu().numpy())
    netlist_, dict_nid_dis_deflect = load_pretrain_data('test/dataset1/large', save_type=2)
    print(netlist_.graph)
    print(netlist_.dict_sub_netlist[4].graph)
    print(netlist_.graph.nodes['cell'].data)
    print(netlist_.dict_sub_netlist[4].graph.nodes['cell'].data)
    print(dict_nid_dis_deflect[-1][0])
    print(dict_nid_dis_deflect[-1][1] / torch.pi % 2)
    print(dict_nid_dis_deflect[4][0])
    print(dict_nid_dis_deflect[4][1] / torch.pi % 2)
