import torch
import numpy as np

###### Method copied + updated from Nie nielang@bjtu.edu.cn
# @article{nie2020view,
#   title={A view-free image stitching network based on global homography},
#   author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Meiqin and Zhao, Yao},
#   journal={Journal of Visual Communication and Image Representation},
#   volume={73},
#   pages={102950},
#   year={2020},
#   publisher={Elsevier}
# }
#
# updated and adapted for PyTorch
#

#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

Aux_M2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)

Aux_M3 = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1]], dtype=np.float64)

Aux_M4 = np.array([
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

Aux_M5 = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

Aux_M6 = np.array([
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0]], dtype=np.float64)

Aux_M71 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

Aux_M72 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, -1, 0]], dtype=np.float64)

Aux_M8 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, -1]], dtype=np.float64)

Aux_Mb = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)


########################################################
def device():
    the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return the_device

def prepare_torch_tensor(matrix, batch_size):
    starting_tensor = torch.tensor(matrix, dtype=torch.float32)
    expanded_tensor = starting_tensor.unsqueeze(0)
    prepared_tile_tensor = torch.tile(expanded_tensor, (batch_size, 1, 1))
    return prepared_tile_tensor

def solve_DLT(pre_4pt_shift, patch_size=128.):
    pre_4pt_shift.to(device())
    batch_size = pre_4pt_shift.shape[0]
    pts_1_tile = torch.tensor([[0., 0., patch_size, 0., 0., patch_size, patch_size, patch_size],], dtype=torch.float32)
    pts_1_tile = torch.tile(pts_1_tile.unsqueeze(2), (batch_size, 1, 1))
    pts_1_tile.to(device())
    pred_pts_2_tile = torch.add(pre_4pt_shift, pts_1_tile)

    # change the oder of original pt4 and predicted pt4 (so that we can get the inverse matrix of H simply)
    orig_pt4 = pred_pts_2_tile
    pred_pt4 = pts_1_tile

    # Auxiliary tensors used to create Ax = b equation
    M1_tile = prepare_torch_tensor(Aux_M1, batch_size)
    M2_tile = prepare_torch_tensor(Aux_M2, batch_size)
    M3_tile = prepare_torch_tensor(Aux_M3, batch_size)
    M4_tile = prepare_torch_tensor(Aux_M4, batch_size)
    M5_tile = prepare_torch_tensor(Aux_M5, batch_size)
    M6_tile = prepare_torch_tensor(Aux_M6, batch_size)
    M71_tile = prepare_torch_tensor(Aux_M71, batch_size)
    M72_tile = prepare_torch_tensor(Aux_M72, batch_size)
    M8_tile = prepare_torch_tensor(Aux_M8, batch_size)
    Mb_tile = prepare_torch_tensor(Aux_Mb, batch_size)

    A1 = torch.matmul(M1_tile, orig_pt4)
    A2 = torch.matmul(M2_tile, orig_pt4)
    A3 = M3_tile
    A4 = torch.matmul(M4_tile, orig_pt4)
    A5 = torch.matmul(M5_tile, orig_pt4)
    A6 = M6_tile
    A7 = torch.matmul(M71_tile, pred_pt4) * torch.matmul(M72_tile, orig_pt4)
    A8 = torch.matmul(M71_tile, pred_pt4) * torch.matmul(M8_tile, orig_pt4)

    # A_mat = torch.transpose(torch.stack([torch.reshape(A1, [-1, 8]),
    #                                        torch.reshape(A2, [-1, 8]),
    #                                        torch.reshape(A3, [-1, 8]),
    #                                        torch.reshape(A4, [-1, 8]),
    #                                        torch.reshape(A5, [-1, 8]),
    #                                        torch.reshape(A6, [-1, 8]),
    #                                        torch.reshape(A7, [-1, 8]),
    #                                        torch.reshape(A8, [-1, 8])], dim=1), 1, 2)
    A_mat = torch.transpose(torch.stack([torch.mean(torch.permute(A1, [0, 2, 1]), 1),
                                         torch.mean(torch.permute(A2, [0, 2, 1]), 1),
                                         torch.mean(torch.permute(A3, [0, 2, 1]), 1),
                                         torch.mean(torch.permute(A4, [0, 2, 1]), 1),
                                         torch.mean(torch.permute(A5, [0, 2, 1]), 1),
                                         torch.mean(torch.permute(A6, [0, 2, 1]), 1),
                                         torch.mean(torch.permute(A7, [0, 2, 1]), 1),
                                         torch.mean(torch.permute(A8, [0, 2, 1]), 1)], dim=1), 1, 2)

    b_mat = torch.matmul(Mb_tile, pred_pt4)

    # Solve the Ax = b
    homography_solved = torch.linalg.solve(A_mat, b_mat)
    homography_ones = torch.ones([batch_size, 1, 1])
    homography_ninth = torch.concat([homography_solved, homography_ones], 1)
    homography_flat = torch.reshape(homography_ninth, [-1, 9])
    homography_matrix = torch.reshape(homography_flat, [-1, 3, 3])
    return homography_matrix

