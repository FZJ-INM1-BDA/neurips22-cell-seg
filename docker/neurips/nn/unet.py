import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter as ILG
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.segmentation.deeplabv3 import ASPP
from collections import OrderedDict
from typing import List, Dict, Type, Union
from celldetection.models.commons import TwoConvNormRelu, ResBlock
from celldetection.models.resnet import *
# from celldetection.models.mobilenetv3 import *
from celldetection.util.util import lookup_nn, get_nd_max_pool, get_nd_conv, get_nd_linear
# from celldetection.models.convnext import ConvNeXtSmall, ConvNeXtTiny, ConvNeXtBase, ConvNeXtLarge
from functools import partial

__all__ = [
    'UNetEncoder', 'UNet', 'U12', 'U17', 'U22', 'SlimU22', 'WideU22',
    # 'MobileNetV3SmallUNet', 'MobileNetV3LargeUNet',
    'ResNet18UNet', 'ResNet34UNet', 'ResNet50UNet', 'ResNet101UNet', 'ResNet152UNet', 'ResNeXt50UNet', 'ResNeXt101UNet',
    'ResNeXt152UNet', 'WideResNet50UNet', 'WideResNet101UNet', 'ResUNet'
]


class IntermediateLayerGetter(ILG):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str], unpool=False) -> None:
        super().__init__(model, return_layers)
        self.unpool = unpool

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            indices = None
            if self.unpool and isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.modules.pooling._MaxPoolNd) and self.unpool:
                        m.return_indices = True
                        size = x.shape[-2:]
                        kernel_size = m.kernel_size
                        padding = m.padding
                        stride = m.stride
                        x, _ = x_, indices = m(x)
                    else:
                        x = x_ = m(x)
            else:
                x = x_ = module(x)
            if indices is not None:
                x_ = F.max_unpool2d(x, indices.repeat([1, x.shape[1] // indices.shape[1], 1, 1]), kernel_size, stride,
                                    output_size=size, padding=padding)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x_
        return out


class Ppm(nn.Module):
    def __init__(self, in_channels, out_channels=64, scales=(1, 2, 3, 6), kernel_size=1, bias=True,
                 norm='BatchNorm2d', activation='relu'):
        super().__init__()
        self.mods = nn.ModuleList()
        norm = lookup_nn(norm, call=False)
        activation = lookup_nn(activation, call=False)

        for scale in scales:
            self.mods.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=scale),
                nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
                norm(out_channels),
                activation(),
            ))

    def forward(self, x):
        return torch.cat([
            F.interpolate(
                m(x), x.shape[-2:], mode='bilinear', align_corners=False
            ) for m in self.mods
        ], 1)


def append_pyr_pool_(encoder, out_channels=None, scales=(1, 2, 3, 6), method='aspp'):
    if out_channels is None:
        out_channels = encoder.out_channels[-1]
    if method == 'aspp':
        scales = sorted(tuple(set(scales) - {1}))
        p = ASPP(encoder.out_channels[-1], scales, out_channels)
    elif method == 'ppm':
        assert (out_channels % len(scales)) == 0
        p = Ppm(encoder.out_channels[-1], out_channels // len(scales), scales=scales)
    else:
        raise ValueError
    encoder.append(p)
    encoder.out_channels += (out_channels,)


class UNetEncoder(nn.Sequential):
    def __init__(self, in_channels, depth=5, base_channels=64, factor=2, pool=True, block_cls: Type[nn.Module] = None,
                 spatial_dims=2):
        """U-Net Encoder.

        Args:
            in_channels: Input channels.
            depth: Model depth.
            base_channels: Base channels.
            factor: Growth factor of base_channels.
            pool: Whether to use max pooling or stride 2 for downsampling.
            block_cls: Block class. Callable as `block_cls(in_channels, out_channels, stride=stride)`.
        """
        block_cls = block_cls or partial(TwoConvNormRelu, spatial_dims=spatial_dims)
        MaxPool = get_nd_max_pool(spatial_dims)
        layers = []
        self.out_channels = []
        for i in range(depth):
            in_c = base_channels * int(factor ** (i - 1)) * int(i > 0) + int(i <= 0) * in_channels
            out_c = base_channels * (factor ** i)
            self.out_channels.append(out_c)
            block = block_cls(in_c, out_c, stride=int((not pool and i > 0) + 1))
            if i > 0 and pool:
                block = nn.Sequential(MaxPool(2, stride=2), block)
            layers.append(block)
        super().__init__(*layers)


class GeneralizedUNet(FeaturePyramidNetwork):
    def __init__(
            self,
            in_channels_list,
            out_channels: int,
            block_cls: nn.Module,
            block_kwargs: dict = None,
            final_activation=None,
            interpolate='nearest',
            final_interpolate=None,
            initialize=True,
            spatial_dims=2
    ):
        super().__init__([], 0)
        self.encoder_out_channels = icl = in_channels_list
        block_kwargs = {} if block_kwargs is None else block_kwargs
        Conv = get_nd_conv(spatial_dims)
        self.interpolate = interpolate
        self.out_channels = out_channels
        self.out_layer = Conv(icl[0], out_channels, 1) if out_channels > 0 else None
        self.spatial_dims = spatial_dims
        self.final_interpolate = final_interpolate
        if self.final_interpolate is None:
            self.final_interpolate = get_nd_linear(spatial_dims)

        for j, in_channels in enumerate(icl):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            if j > 0:
                inner = Conv(in_channels, icl[j - 1], 1) if icl[j - 1] < in_channels else nn.Identity()
                self.inner_blocks.append(inner)
            if j < len(icl) - 1:
                layer_block_module = block_cls(in_channels + min(icl[j:j + 2]), in_channels, **block_kwargs)
                self.layer_blocks.append(layer_block_module)

        if initialize:
            for m in self.children():
                if isinstance(m, Conv):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)
        self.final_activation = None if final_activation is None else lookup_nn(final_activation)

    def forward(self, x: Dict[str, Tensor], size: List[int]) -> Union[Dict[str, Tensor], Tensor]:
        """

        Args:
            x: Input dictionary. E.g. {
                    0: Tensor[1, 64, 128, 128]
                    1: Tensor[1, 128, 64, 64]
                    2: Tensor[1, 256, 32, 32]
                    3: Tensor[1, 512, 16, 16]
                }
            size: Desired final output size. If set to None output remains as it is.

        Returns:
            Output dictionary. For each key in `x` a corresponding output is returned; the final output
            has the key `'out'`.
            E.g. {
                out: Tensor[1, 2, 128, 128]
                0: Tensor[1, 64, 128, 128]
                1: Tensor[1, 128, 64, 64]
                2: Tensor[1, 256, 32, 32]
                3: Tensor[1, 512, 16, 16]
            }
        """
        names = list(x.keys())
        x = list(x.values())
        last_inner = x[-1]
        results = [last_inner]
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = x[idx]
            feat_shape = inner_lateral.shape[-self.spatial_dims:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode=self.interpolate,
                                           align_corners=None if self.interpolate == 'nearest' else False)
            inner_top_down = self.get_result_from_inner_blocks(inner_top_down, idx)  # reduce channels
            last_inner = torch.cat((inner_lateral, inner_top_down), 1)  # concat
            last_inner = self.get_result_from_layer_blocks(last_inner, idx)  # apply layer
            results.insert(0, last_inner)

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        if size is None:
            final = results[0]
        else:
            final = F.interpolate(last_inner, size=size, mode=self.final_interpolate, align_corners=False)
        if self.out_layer is not None:
            final = self.out_layer(final)
        if self.final_activation is not None:
            final = self.final_activation(final)
        if self.out_channels:
            return final
        results.insert(0, final)
        names.insert(0, 'out')
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class BackboneAsUNet(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, block, block_kwargs: dict = None,
                 final_activation=None, interpolate='nearest', spatial_dims=2, unpool=False, **kwargs):
        super(BackboneAsUNet, self).__init__()
        block = block or partial(TwoConvNormRelu, spatial_dims=spatial_dims)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers, unpool=unpool)
        self.unet = GeneralizedUNet(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            block_cls=block,
            block_kwargs=block_kwargs,
            # extra_blocks=LastLevelMaxPool(),
            final_activation=final_activation,
            interpolate=interpolate,
            spatial_dims=spatial_dims,
            **kwargs
        )
        self.out_channels = list(in_channels_list)
        self.spatial_dims = spatial_dims

    def forward(self, inputs):
        features = self.body(inputs)
        x = self.unet(features, size=inputs.shape[-self.spatial_dims:])
        return x, features


class UNet(BackboneAsUNet):
    def __init__(self, backbone, out_channels: int, return_layers: dict = None,
                 block: Type[nn.Module] = None, block_kwargs: dict = None, final_activation=None,
                 interpolate='nearest', spatial_dims=2, **kwargs):
        """U-Net.

        Examples:
            >>> model = UNet(UNetEncoder(in_channels=3), out_channels=2)

            >>> model = UNet(UNetEncoder(in_channels=3, base_channels=16), out_channels=2)
            >>> o = model(torch.rand(1, 3, 256, 256))
            >>> for k, v in o.items():
            ...     print(k, "\t", v.shape)
            out 	 torch.Size([1, 2, 256, 256])
            0 	 torch.Size([1, 16, 256, 256])
            1 	 torch.Size([1, 32, 128, 128])
            2 	 torch.Size([1, 64, 64, 64])
            3 	 torch.Size([1, 128, 32, 32])
            4 	 torch.Size([1, 256, 16, 16])

        Args:
            backbone: Backbone instance.
            out_channels: Output channels.
            return_layers: Dictionary like `{backbone_layer_name: out_name}`.
                Note that this influences how outputs are computed, as the input for the upsampling
                is gathered by `IntermediateLayerGetter` based on given dict keys.
            block: Module class. Default is `block=TwoConvNormRelu`. Must be callable: block(in_channels, out_channels).
            final_activation: Final activation function.
        """
        block = partial(TwoConvNormRelu, spatial_dims=spatial_dims) if block is None else block
        names = [name for name, _ in backbone.named_children()]  # assuming ordered
        if return_layers is None:
            return_layers = {n: str(i) for i, n in enumerate(names)}
        layers = {str(k): (str(names[v]) if isinstance(v, int) else str(v)) for k, v in return_layers.items()}
        super(UNet, self).__init__(
            backbone=backbone,
            return_layers=layers,
            in_channels_list=list(backbone.out_channels),
            out_channels=out_channels,
            block=block,
            block_kwargs=block_kwargs,
            final_activation=final_activation if out_channels else None,
            interpolate=interpolate,
            spatial_dims=spatial_dims,
            **kwargs
        )


def _ni_pretrained(pretrained):
    if pretrained:
        raise NotImplemented('The `pretrained` option is not yet available for this model.')


def _default_unet_kwargs(backbone_kwargs, pretrained=False):
    _ni_pretrained(pretrained)
    kw = dict()
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


class U22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        """U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.

        References:
            https://arxiv.org/abs/1505.04597

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        super().__init__(
            UNetEncoder(in_channels=in_channels, block_cls=block_cls,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, **kwargs
        )


class ResUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        """Residual U-Net.

        U-Net with residual blocks.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``ResBlock``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        block_cls = block_cls or ResBlock
        super().__init__(
            UNetEncoder(in_channels=in_channels, block_cls=block_cls,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, **kwargs
        )


class SlimU22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        """Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels reduce by half.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        super().__init__(
            UNetEncoder(in_channels=in_channels, base_channels=32, block_cls=block_cls,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, **kwargs
        )


class WideU22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        """Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels doubled.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        super().__init__(
            UNetEncoder(in_channels=in_channels, base_channels=128, block_cls=block_cls,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, **kwargs
        )


class U17(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        """U-Net 17.

        U-Net with 17 convolutions on 4 feature resolutions (1, 1/2, 1/4, 1/8) and one final output layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        super().__init__(
            UNetEncoder(in_channels=in_channels, depth=4, block_cls=block_cls,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, **kwargs
        )


class U12(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        """U-Net 12.

        U-Net with 12 convolutions on 3 feature resolutions (1, 1/2, 1/4) and one final output layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        super().__init__(
            UNetEncoder(in_channels=in_channels, depth=3, block_cls=block_cls,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, **kwargs
        )


def _default_res_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(fused_initial=False, pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


class ResNet18UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, pyr_pool=False, **kwargs):
        """ResNet 18 U-Net.

        A U-Net with ResNet 18 encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        encoder = ResNet18(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained))
        if pyr_pool:
            append_pyr_pool_(encoder, **(pyr_pool if isinstance(pyr_pool, dict) else {}))
        super().__init__(encoder,
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)


class ResNet34UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        super().__init__(ResNet34(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 34')


class ResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, pyr_pool=False, **kwargs):
        encoder = ResNet50(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained))
        if pyr_pool:
            append_pyr_pool_(encoder, **(pyr_pool if isinstance(pyr_pool, dict) else {}))
        super().__init__(encoder,
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 50')


class ResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, pyr_pool=False, **kwargs):
        encoder = ResNet101(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained))
        if pyr_pool:
            append_pyr_pool_(encoder, **(pyr_pool if isinstance(pyr_pool, dict) else {}))
        super().__init__(encoder,
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 101')


class ResNet152UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        super().__init__(ResNet152(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 152')


class ResNeXt50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, pyr_pool=False, **kwargs):
        encoder = ResNeXt50_32x4d(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained))
        if pyr_pool:
            append_pyr_pool_(encoder, **(pyr_pool if isinstance(pyr_pool, dict) else {}))
        super().__init__(
            encoder,
            out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 50')


class ResNeXt101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, pyr_pool=False, **kwargs):
        encoder = ResNeXt101_32x8d(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained))
        if pyr_pool:
            append_pyr_pool_(encoder, **(pyr_pool if isinstance(pyr_pool, dict) else {}))
        super().__init__(
            encoder,
            out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 101')


class ResNeXt152UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        super().__init__(
            ResNeXt152_32x8d(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 152')


class WideResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        super().__init__(
            WideResNet50_2(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Wide ResNet 50')


class WideResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, **kwargs):
        super().__init__(
            WideResNet101_2(in_channels, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Wide ResNet 101')


# class MobileNetV3SmallUNet(UNet):
#     def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
#                  block_cls=None, **kwargs):
#         _ni_pretrained(pretrained)
#         super().__init__(MobileNetV3Small(in_channels, **(backbone_kwargs or {})), out_channels,
#                          final_activation=final_activation, block=block_cls, **kwargs)
#
#     __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Small MobileNet V3')
#
#
# class MobileNetV3LargeUNet(UNet):
#     def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
#                  block_cls=None, **kwargs):
#         _ni_pretrained(pretrained)
#         super().__init__(MobileNetV3Large(in_channels, **(backbone_kwargs or {})), out_channels,
#                          final_activation=final_activation, block=block_cls, **kwargs)
#
#     __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'Large MobileNet V3')
