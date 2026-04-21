import torch

try:
    import simulon_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class GPUKDTree:
    def __init__(self, points):
        self.points = points
        self.device = points.device

    def query_pairs(self, r):
        n = self.points.shape[0]
        i_indices = torch.arange(n, device=self.device)
        j_indices = torch.arange(n, device=self.device)
        mask = i_indices.unsqueeze(1) < j_indices.unsqueeze(0)
        i_pairs = i_indices.unsqueeze(1).expand(n, n)[mask]
        j_pairs = j_indices.unsqueeze(0).expand(n, n)[mask]
        points_i = self.points[i_pairs]
        points_j = self.points[j_pairs]
        diff = points_i - points_j
        dist_sq = torch.sum(diff * diff, dim=1)
        valid_mask = dist_sq < r ** 2
        valid_i = i_pairs[valid_mask]
        valid_j = j_pairs[valid_mask]
        result = [(i.item(), j.item()) for i, j in zip(valid_i, valid_j)]
        return result

    def batch_query_pairs(self, r, batch_size=1000):
        n = self.points.shape[0]
        result = []
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            i_points = self.points[i_start:i_end]
            for j_start in range(i_start, n, batch_size):
                j_end = min(j_start + batch_size, n)
                j_points = self.points[j_start:j_end]
                i_expanded = i_points.unsqueeze(1)
                j_expanded = j_points.unsqueeze(0)
                diff = i_expanded - j_expanded
                dist_sq = torch.sum(diff * diff, dim=2)
                i_idx, j_idx = torch.where(dist_sq < r ** 2)
                i_idx += i_start
                j_idx += j_start
                valid_mask = i_idx < j_idx
                i_idx = i_idx[valid_mask]
                j_idx = j_idx[valid_mask]
                for i, j in zip(i_idx, j_idx):
                    result.append((i.item(), j.item()))
        return result


def find_neighbors_gpu_pbc(positions, cutoff, box, batch_size=2000):
    """
    box: Box 对象（支持三斜）或 标量/张量 box_length（向后兼容）。
    """
    device = positions.device

    # CUDA kernel 路径：仅支持正交盒子（scalar box_length）
    _is_box_obj = hasattr(box, 'minimum_image')
    _orthogonal  = (not _is_box_obj) or box.is_orthogonal

    if CUDA_AVAILABLE and device.type == 'cuda' and _orthogonal:
        bl = float(box.diag[0]) if _is_box_obj else float(box)
        try:
            edge_index, edge_attr = simulon_cuda.neighbor_search_cuda(
                positions.contiguous(), float(cutoff), bl
            )
            return edge_index, edge_attr
        except Exception as e:
            print(f"CUDA kernel failed, falling back to PyTorch: {e}")

    return find_neighbors_gpu_pbc_pytorch(positions, cutoff, box, batch_size)


def find_neighbors_gpu_pbc_pytorch(positions, cutoff, box, batch_size=None):
    """
    优化版邻居搜索：单层i-batch循环，向量化对比所有N个j原子。
    支持正交和三斜盒子（通过 Box 对象）。

    box: Box 对象 或 标量 box_length（向后兼容）。
    """
    device = positions.device
    n = positions.shape[0]
    rc2 = cutoff * cutoff

    # 构造最小镜像函数
    _is_box_obj = hasattr(box, 'minimum_image')
    if _is_box_obj:
        def _min_image(r):          # r: [B, N, 3]
            shape = r.shape
            return box.minimum_image(r.reshape(-1, 3)).reshape(shape)
    else:
        bl = float(box)
        def _min_image(r): return r - bl * torch.round(r / bl)

    # 根据可用显存自动计算 batch_size（目标单批次 ≤ 512 MB）
    if batch_size is None:
        if device.type == 'cuda':
            target_bytes = 512 * 1024 * 1024
            batch_size = max(64, target_bytes // (n * 3 * 4))
            batch_size = min(batch_size, n)
        else:
            batch_size = min(2000, n)

    edge_i_list = []
    edge_j_list = []

    j_global = torch.arange(n, device=device)

    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        i_pos = positions[i_start:i_end]          # [B, 3]

        rij = i_pos.unsqueeze(1) - positions.unsqueeze(0)  # [B, N, 3]
        rij = _min_image(rij)
        dist_sq = (rij * rij).sum(-1)                       # [B, N]

        # upper-triangle过滤：全局 i < 全局 j（消除重复边和自身对）
        i_global = torch.arange(i_start, i_end, device=device).unsqueeze(1)  # [B, 1]
        upper_mask = i_global < j_global.unsqueeze(0)                         # [B, N]

        mask = upper_mask & (dist_sq < rc2)
        bi, bj = torch.where(mask)          # bi: batch内局部索引, bj: 全局j索引

        if bi.numel() > 0:
            edge_i_list.append(bi + i_start)
            edge_j_list.append(bj)

    if edge_i_list:
        ei = torch.cat(edge_i_list)
        ej = torch.cat(edge_j_list)
        # 只对筛选后的边重新计算精确距离（使用已定义的 _min_image 闭包）
        rij_valid = _min_image(positions[ei] - positions[ej])
        dist = torch.norm(rij_valid, dim=1)
        return torch.stack([ei, ej]), dist
    else:
        return (
            torch.zeros((2, 0), device=device, dtype=torch.long),
            torch.zeros(0, device=device),
        )
