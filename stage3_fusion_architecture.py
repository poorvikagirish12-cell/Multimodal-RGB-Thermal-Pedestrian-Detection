import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    """
    Memory-Efficient Bi-directional Cross-Modal Attention Fusion (CMAF) Module
    Uses spatially pooled Keys/Values (Linear Cross-Attention) for linear O(HW) complexity.
    """
    def __init__(self, in_channels, pool_size=8):
        super(CrossModalAttentionFusion, self).__init__()
        self.in_channels = in_channels
        self.pool_size = pool_size
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        # Spatial Attention Projection
        self.query_rgb = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_ir    = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_ir  = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.query_ir  = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_rgb   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Channel Squeeze-and-Excitation Gate
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output Fusion Conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_rgb, feat_ir):
        B, C, H, W = feat_rgb.size()
        
        # 1. Pool Keys & Values to pool_size x pool_size (K=pool_size^2=64)
        ir_pooled = self.spatial_pool(feat_ir)   # B x C x P x P
        rgb_pooled = self.spatial_pool(feat_rgb) # B x C x P x P
        
        # 2. RGB Queries Thermal
        q_rgb = self.query_rgb(feat_rgb).view(B, -1, H * W).permute(0, 2, 1) # B x HW x C'
        k_ir  = self.key_ir(ir_pooled).view(B, -1, self.pool_size * self.pool_size) # B x C' x K
        v_ir  = self.value_ir(ir_pooled).view(B, -1, self.pool_size * self.pool_size) # B x C x K
        
        attn_rgb_ir = F.softmax(torch.bmm(q_rgb, k_ir) / (C ** 0.5), dim=-1) # B x HW x K
        enhanced_rgb = torch.bmm(v_ir, attn_rgb_ir.permute(0, 2, 1)).view(B, C, H, W) + feat_rgb
        
        # 3. Thermal Queries RGB
        q_ir  = self.query_ir(feat_ir).view(B, -1, H * W).permute(0, 2, 1)
        k_rgb = self.key_rgb(rgb_pooled).view(B, -1, self.pool_size * self.pool_size)
        v_rgb = self.value_rgb(rgb_pooled).view(B, -1, self.pool_size * self.pool_size)
        
        attn_ir_rgb = F.softmax(torch.bmm(q_ir, k_rgb) / (C ** 0.5), dim=-1)
        enhanced_ir = torch.bmm(v_rgb, attn_ir_rgb.permute(0, 2, 1)).view(B, C, H, W) + feat_ir
        
        # 4. Dynamic Channel Gating & Concatenation
        concat_feat = torch.cat([enhanced_rgb, enhanced_ir], dim=1) # B x 2C x H x W
        weights = self.channel_gate(concat_feat)
        gated_feat = concat_feat * weights
        
        # 5. Final Fusion Projection
        fused_out = self.fuse_conv(gated_feat)
        return fused_out


class SmallObjectEnhancementModule(nn.Module):
    """
    Small-Object Enhancement Module (SOEM)
    Constructs a high-resolution P2 feature level (stride 4, 480x270)
    with Dense Multi-Scale Feature Aggregation (MSFA).
    """
    def __init__(self, feature_channels=256):
        super(SmallObjectEnhancementModule, self).__init__()
        self.channels = feature_channels
        
        self.p2_lateral = nn.Conv2d(64, feature_channels, kernel_size=1)
        
        self.dilated_conv1 = nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1, dilation=1)
        self.dilated_conv2 = nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=2, dilation=2)
        
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, c2_fused, p3_feat):
        p3_upsampled = F.interpolate(p3_feat, scale_factor=2, mode='nearest')
        p2_lat = self.p2_lateral(c2_fused)
        p2_fused = p2_lat + p3_upsampled
        
        d1 = self.dilated_conv1(p2_fused)
        d2 = self.dilated_conv2(p2_fused)
        p2_enhanced = self.smooth_p2(torch.cat([d1, d2], dim=1))
        
        return p2_enhanced


if __name__ == '__main__':
    print("Testing Memory-Efficient CMAF and SOEM PyTorch modules...")
    cmaf = CrossModalAttentionFusion(in_channels=256, pool_size=8)
    soem = SmallObjectEnhancementModule(feature_channels=256)
    
    rgb_c3 = torch.randn(2, 256, 135, 240)
    ir_c3  = torch.randn(2, 256, 135, 240)
    fused_c3 = cmaf(rgb_c3, ir_c3)
    print("CMAF output shape:", fused_c3.shape)
    
    c2_fused = torch.randn(2, 64, 270, 480)
    p3_feat  = torch.randn(2, 256, 135, 240)
    p2_out = soem(c2_fused, p3_feat)
    print("SOEM P2 output shape:", p2_out.shape)
    print("Memory-efficient PyTorch modules tested successfully!")
