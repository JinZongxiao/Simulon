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

                i_expanded = i_points.unsqueeze(1)  # [batch_i, 1, 3]
                j_expanded = j_points.unsqueeze(0)  # [1, batch_j, 3]

                diff = i_expanded - j_expanded  # [batch_i, batch_j, 3]
                dist_sq = torch.sum(diff * diff, dim=2)  # [batch_i, batch_j]

                i_idx, j_idx = torch.where(dist_sq < r ** 2)

                i_idx += i_start
                j_idx += j_start

                valid_mask = i_idx < j_idx
                i_idx = i_idx[valid_mask]
                j_idx = j_idx[valid_mask]

                for i, j in zip(i_idx, j_idx):
                    result.append((i.item(), j.item()))

        return result


def find_neighbors_gpu_pbc(positions, cutoff, box_length, batch_size=2000):

    device = positions.device
    
    # Try to use CUDA kernel if available
    if CUDA_AVAILABLE and device.type == 'cuda':
        try:
            edge_index, edge_attr = simulon_cuda.neighbor_search_cuda(
                positions.contiguous(), 
                float(cutoff), 
                float(box_length)
            )
            return edge_index, edge_attr
        except Exception as e:
            print(f"CUDA kernel failed, falling back to PyTorch: {e}")
    
    # Fallback to PyTorch implementation
    return find_neighbors_gpu_pbc_pytorch(positions, cutoff, box_length, batch_size)


def find_neighbors_gpu_pbc_pytorch(positions, cutoff, box_length, batch_size=2000):
    """
    PyTorch-based GPU neighbor search (original implementation)
    """
    device = positions.device
    n = positions.shape[0]

    edge_indices = []
    edge_attrs = []

    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        i_indices = torch.arange(i_start, i_end, device=device)
        i_pos = positions[i_start:i_end]

        for j_start in range(0, n, batch_size):
            j_end = min(j_start + batch_size, n)
            j_indices = torch.arange(j_start, j_end, device=device)
            j_pos = positions[j_start:j_end]

            rij = i_pos.unsqueeze(1) - j_pos.unsqueeze(0)
            rij = rij - box_length * torch.round(rij / box_length)

            dist_sq = torch.sum(rij * rij, dim=2)

            mask = (dist_sq < cutoff ** 2)
            if i_start == j_start:
                diag_mask = torch.eye(i_end - i_start, j_end - j_start, device=device, dtype=torch.bool)
                mask = mask & ~diag_mask

            if i_start > j_start:
                i_expanded = i_indices.unsqueeze(1).expand(i_end - i_start, j_end - j_start)
                j_expanded = j_indices.unsqueeze(0).expand(i_end - i_start, j_end - j_start)
                upper_mask = i_expanded < j_expanded
                mask = mask & upper_mask

            i_idx, j_idx = torch.where(mask)

            i_idx += i_start
            j_idx += j_start

            if i_idx.numel() > 0:
                pos_i = positions[i_idx]
                pos_j = positions[j_idx]

                r_ij = pos_j - pos_i
                r_ij = r_ij - box_length * torch.round(r_ij / box_length)
                dist = torch.norm(r_ij, dim=1)

                edge_indices.append(torch.stack([i_idx, j_idx], dim=0))
                edge_attrs.append(dist)

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs)
        return edge_index, edge_attr
    else:
        return torch.zeros((2, 0), device=device, dtype=torch.long), torch.zeros(0, device=device)