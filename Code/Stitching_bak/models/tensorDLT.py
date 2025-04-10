import torch


def solve_DLT(shift, patch_size=128.0):
    """
    Solve the Direct Linear Transform (DLT) to compute the homography matrix.

    Args:
        shift (torch.Tensor): Shift tensor of shape (batch_size, 8), representing the 4 corner points.
        patch_size (float): Size of the patch for normalization.

    Returns:
        torch.Tensor: Homography matrix of shape (batch_size, 3, 3).
    """
    batch_size = shift.shape[0]

    # Original corner points of the patch
    pts_1_tile = torch.tensor(
        [0.0, 0.0, patch_size, 0.0, 0.0, patch_size, patch_size, patch_size],
        dtype=torch.float32,
        device=shift.device,
    ).view(1, 8, 1).repeat(batch_size, 1, 1)

    # Predicted corner points (shifted points)
    pred_h4p_tile = shift.view(batch_size, 8, 1)
    pred_pts_2_tile = pred_h4p_tile + pts_1_tile

    # Reorder points for inverse matrix computation
    orig_pt4 = pred_pts_2_tile
    pred_pt4 = pts_1_tile

    # Construct the A matrix for DLT
    A = []
    for i in range(batch_size):
        x1, y1, x2, y2, x3, y3, x4, y4 = orig_pt4[i].view(-1)
        u1, v1, u2, v2, u3, v3, u4, v4 = pred_pt4[i].view(-1)

        A.append([
            [-x1, -y1, -1, 0, 0, 0, u1 * x1, u1 * y1, u1],
            [0, 0, 0, -x1, -y1, -1, v1 * x1, v1 * y1, v1],
            [-x2, -y2, -1, 0, 0, 0, u2 * x2, u2 * y2, u2],
            [0, 0, 0, -x2, -y2, -1, v2 * x2, v2 * y2, v2],
            [-x3, -y3, -1, 0, 0, 0, u3 * x3, u3 * y3, u3],
            [0, 0, 0, -x3, -y3, -1, v3 * x3, v3 * y3, v3],
            [-x4, -y4, -1, 0, 0, 0, u4 * x4, u4 * y4, u4],
            [0, 0, 0, -x4, -y4, -1, v4 * x4, v4 * y4, v4],
        ])
    A = torch.tensor(A, dtype=torch.float32, device=shift.device)

    # Solve for the homography matrix using SVD
    U, S, V = torch.svd(A)
    H = V[:, -1].view(batch_size, 3, 3)

    return H