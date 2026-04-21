"""
core/box.py — 周期性边界条件盒子管理

支持三种盒子类型：
  - 立方体   (cubic):       h = scalar a   → H = diag(a, a, a)
  - 正交体   (orthorhombic): h = [Lx,Ly,Lz] → H = diag(Lx, Ly, Lz)
  - 三斜体   (triclinic):    h = [[a1,a2,a3],
                                   [b1,b2,b3],
                                   [c1,c2,c3]]  (行 = 格矢量)

所有力场、邻居搜索、积分器通过 Box.minimum_image() 做最小镜像约定，
避免在各处硬编码 `rij -= L * round(rij/L)`。
"""
from __future__ import annotations
import torch
from typing import Union


class Box:
    """周期性盒子，统一正交和三斜情形。"""

    def __init__(
        self,
        h: Union[float, list, torch.Tensor],
        device: torch.device = None,
    ):
        """
        Parameters
        ----------
        h : scalar | list[3] | list[3][3] | Tensor
            盒子参数（单位 Å）。
        device : torch.device, optional
            张量所在设备，默认自动检测。
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        h_t = torch.as_tensor(h, dtype=torch.float64)

        if h_t.dim() == 0 or h_t.numel() == 1:
            L = float(h_t)
            self._H = torch.diag(torch.tensor([L, L, L], dtype=torch.float64, device=device))
        elif h_t.dim() == 1 and h_t.numel() == 3:
            self._H = torch.diag(h_t.double().to(device))
        elif h_t.shape == (3, 3):
            self._H = h_t.double().to(device)
        else:
            raise ValueError(f"h 须为标量、[3] 或 [3,3] 张量，得到 shape={tuple(h_t.shape)}")

        self._refresh()

    # ─── 内部派生量更新 ─────────────────────────────────────────────────────
    def _refresh(self):
        self._H_inv = torch.linalg.inv(self._H)
        off = self._H - torch.diag(torch.diag(self._H))
        self._is_orthogonal = bool(torch.allclose(off, torch.zeros_like(off), atol=1e-8))

    # ─── 属性 ───────────────────────────────────────────────────────────────
    @property
    def H(self) -> torch.Tensor:
        """格矢量矩阵 [3, 3]（行 = 格矢）。"""
        return self._H

    @H.setter
    def H(self, value: torch.Tensor):
        self._H = value.double().to(self.device)
        self._refresh()

    @property
    def H_inv(self) -> torch.Tensor:
        return self._H_inv

    @property
    def is_orthogonal(self) -> bool:
        return self._is_orthogonal

    @property
    def volume(self) -> float:
        return float(torch.det(self._H).abs())

    @property
    def lengths(self) -> torch.Tensor:
        """每个格矢的长度 [3]。"""
        return torch.norm(self._H, dim=1)

    @property
    def diag(self) -> torch.Tensor:
        """正交盒子的对角元素 [Lx, Ly, Lz]。"""
        return torch.diag(self._H)

    @property
    def box_length(self) -> torch.Tensor:
        """立方体盒子的棱长（标量张量）。非立方体时引发 ValueError。"""
        d = torch.diag(self._H)
        if not self._is_orthogonal or not torch.allclose(d, d[0].expand_as(d), rtol=1e-5):
            raise ValueError(
                "box_length 仅适用于立方体盒子；"
                "正交/三斜体系请使用 box.diag 或 box.H。"
            )
        return d[0].to(torch.float32)

    @property
    def box_length_cpu(self) -> float:
        """立方体棱长（Python float，兼容旧代码）。"""
        return float(self.box_length)

    # ─── 核心操作 ────────────────────────────────────────────────────────────
    def minimum_image(self, rij: torch.Tensor) -> torch.Tensor:
        """
        对位移向量施加最小镜像约定（PBC）。

        Parameters
        ----------
        rij : Tensor [..., 3]
            任意批次形状的笛卡尔位移向量。

        Returns
        -------
        Tensor [..., 3]  与输入同形状的折叠后位移向量。
        """
        if self._is_orthogonal:
            d = torch.diag(self._H).to(rij)        # [3]
            return rij - d * torch.round(rij / d)
        else:
            H_inv = self._H_inv.to(rij)             # [3, 3]
            H     = self._H.to(rij)
            s = rij @ H_inv.T                       # 分数坐标
            s = s - torch.round(s)
            return s @ H.T                          # 回到笛卡尔

    def wrap_positions(self, pos: torch.Tensor) -> torch.Tensor:
        """将坐标折回主盒子 [0, L)。"""
        if self._is_orthogonal:
            d = torch.diag(self._H).to(pos)
            return pos - torch.floor(pos / d) * d
        else:
            H_inv = self._H_inv.to(pos)
            H     = self._H.to(pos)
            s = pos @ H_inv.T
            s = s - torch.floor(s)
            return s @ H.T

    # ─── NPT 缩放 ────────────────────────────────────────────────────────────
    def scale(self, factor: float):
        """各向同性等比缩放（用于 Berendsen NPT）。"""
        self._H = self._H * factor
        self._refresh()

    def scale_anisotropic(self, mu: torch.Tensor):
        """各向异性缩放：H_new = mu @ H。"""
        self._H = mu.double().to(self.device) @ self._H
        self._refresh()

    # ─── 设备迁移 ────────────────────────────────────────────────────────────
    def to(self, device) -> "Box":
        self.device = device
        self._H     = self._H.to(device)
        self._H_inv = self._H_inv.to(device)
        return self

    # ─── 序列化（用于 restart） ───────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {"H": self._H.cpu()}

    @classmethod
    def from_state_dict(cls, d: dict, device=None) -> "Box":
        box = cls.__new__(cls)
        box.device = device or torch.device("cpu")
        box._H = d["H"].to(box.device)
        box._refresh()
        return box

    def __repr__(self) -> str:
        kind = "cubic" if self._is_orthogonal and torch.allclose(
            torch.diag(self._H), torch.diag(self._H)[0].expand(3), rtol=1e-4
        ) else ("orthogonal" if self._is_orthogonal else "triclinic")
        return f"Box({kind}, H=\n{self._H.cpu().numpy()})"
