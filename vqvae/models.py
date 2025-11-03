import torch
import torch.nn as nn
import torch.nn.functional as F

def encoder3d_resnet(
        video: torch.Tensor,        # (B, T, 3, H, W), float32
        d_model: int = 512,         # 输出通道 D
        return_stride: bool = True  # 是否返回时间总stride
) -> tuple[torch.Tensor, int]:
    """
    
    """
    x = video.permute(0, 2, 1, 3, 4).contiguous()
    from torchvision.models.video import r3d_18
    
