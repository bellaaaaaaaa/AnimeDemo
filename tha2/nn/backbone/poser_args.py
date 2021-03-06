from typing import Optional

from torch.nn import Sigmoid, Sequential, Tanh

from tha2.nn.base.conv import create_conv3, create_conv3_from_block_args
from tha2.nn.base.nonlinearity_factory import ReLUFactory
from tha2.nn.base.normalization import InstanceNorm2dFactory
from tha2.nn.base.util import BlockArgs

STRIDE = 1
PADDING = 1
KERNEL = 3
OUTCHANNELS = 1


class PoserArgs00:
    def __init__(self,
                 image_size: int,
                 input_image_channels: int,
                 output_image_channels: int,
                 start_channels: int,
                 num_pose_params: int,
                 block_args: Optional[BlockArgs] = None):
        self.num_pose_params = num_pose_params
        self.start_channels = start_channels
        self.output_image_channels = output_image_channels
        self.input_image_channels = input_image_channels
        self.image_size = image_size
        if block_args is None:
            self.block_args = BlockArgs(
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))
        else:
            self.block_args = block_args

    def create_alpha_block(self):
        from torch.nn import Sequential
        from torch.nn import Conv2d
        from tha2.nn.base.util import wrap_conv_or_linear_module
        
        bias=True
        initialization_method=self.block_args.initialization_method
        use_spectral_norm=False

        con2d = Conv2d(self.start_channels,
                        OUTCHANNELS,
                        kernel_size = KERNEL,
                        stride=STRIDE,
                        padding=PADDING,
                        bias=bias)
                        
        ret_conv_3 = wrap_conv_or_linear_module(con2d, initialization_method, use_spectral_norm)
        
        sequence = Sequential(ret_conv_3, Sigmoid())
        return sequence

    # def create_all_channel_alpha_block(self):
    #     from torch.nn import Sequential
    #     return Sequential(
    #         create_conv3(
    #             in_channels=self.start_channels,
    #             out_channels=self.output_image_channels,
    #             bias=True,
    #             initialization_method=self.block_args.initialization_method,
    #             use_spectral_norm=False),
    #         Sigmoid())

    def create_all_channel_alpha_block(self):
        from torch.nn import Sequential
        from torch.nn import Conv2d
        from tha2.nn.base.util import wrap_conv_or_linear_module

        in_channels=self.start_channels
        out_channels=self.output_image_channels
        bias=True
        initialization_method=self.block_args.initialization_method
        use_spectral_norm=False
        ks = 3
        con2d = Conv2d(in_channels, out_channels, kernel_size = ks, stride=STRIDE, padding=PADDING, bias=bias)

        cc3 = wrap_conv_or_linear_module(con2d, initialization_method, use_spectral_norm)

        return Sequential(cc3, Sigmoid())

    def create_color_change_block(self):
        return Sequential(
            create_conv3_from_block_args(
                in_channels=self.start_channels,
                out_channels=self.output_image_channels,
                bias=True,
                block_args=self.block_args),
            Tanh())

    def create_grid_change_block(self):
        return create_conv3(
            in_channels=self.start_channels,
            out_channels=2,
            bias=False,
            initialization_method='zero',
            use_spectral_norm=False)