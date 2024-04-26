import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.models.archs.fftformer_arch import (
    OverlapPatchEmbed,
    TransformerBlock,
    Upsample,
    Fuse,
)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


##########################################################################
##---------- FFTformer -----------------------
class fftformer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12, 8],
        num_refinement_blocks=4,
        ffn_expansion_factor=3,
        bias=False,
    ):
        super(fftformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
