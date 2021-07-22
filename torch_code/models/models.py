import torch.nn as nn
from .resnet import ResNet, BasicBlock, Bottleneck, SimpleResNet
from .simplecnn import SimpleCNN


class Models:
    """
    This is a wrapper class that defines several model functions.
    The model functions are called by passing the function name as a string.
    All model functions take the number of outputs (num_outputs) as an argument (int).

    Example:
        model_chooser = Models()
        model = model_chooser('SimpleResNet_8blocks')(num_outputs=1)

    """
    def __init__(self):
        pass

    def __call__(self, func):
        """
        Args:
            func: function name as string

        Returns: corresponding model function
        """
        return getattr(self, func)

    # SIMPLE CNN

    def SimpleCNN_8layers(self, num_outputs):
        return SimpleCNN(out_features=(16, 32, 64, 128, 128, 128, 128, 128),
                         num_outputs=num_outputs,
                         global_pool=nn.AdaptiveAvgPool1d)

    # ---- WIDER VERSIONS ----

    # double the number of features in each layer w.r.t. SimpleCNN_8layers
    def SimpleCNN_8layers_width2(self, num_outputs):
        return SimpleCNN(out_features=(32, 64, 128, 256, 256, 256, 256, 256),
                         num_outputs=num_outputs,
                         global_pool=nn.AdaptiveAvgPool1d)

    # The number of features in each layer w.r.t. SimpleCNN_8layers multiplied by factor 4
    def SimpleCNN_8layers_width4(self, num_outputs):
        return SimpleCNN(out_features=(64, 128, 256, 512, 512, 512, 512, 512),
                         num_outputs=num_outputs,
                         global_pool=nn.AdaptiveAvgPool1d)

    # The number of features in each layer w.r.t. SimpleCNN_8layers multiplied by factor 8
    def SimpleCNN_8layers_width8(self, num_outputs):
        return SimpleCNN(out_features=(128, 256, 512, 1024, 1024, 1024, 1024, 1024),
                         num_outputs=num_outputs,
                         global_pool=nn.AdaptiveAvgPool1d)

    # ---- DEEPER VERSIONS ----

    def SimpleCNN_12layers(self, num_outputs):
        return SimpleCNN(out_features=(16, 32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128),
                         num_outputs=num_outputs,
                         global_pool=nn.AdaptiveAvgPool1d)

    def SimpleCNN_16layers(self, num_outputs):
        return SimpleCNN(out_features=(16, 32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
                         num_outputs=num_outputs,
                         global_pool=nn.AdaptiveAvgPool1d)

    # SIMPLE RESNET

    def SimpleResNet_8blocks(self, num_outputs):
        return SimpleResNet(block=BasicBlock,
                            in_features=1,
                            out_features=(16, 32, 64, 128, 128, 128, 128, 128),
                            num_outputs=num_outputs,
                            global_pool=nn.AdaptiveAvgPool1d,
                            max_pool=True)

    def SimpleResNet_8blocks_width2(self, num_outputs):
        return SimpleResNet(block=BasicBlock,
                            in_features=1,
                            out_features=(32, 64, 128, 256, 256, 256, 256, 256),
                            num_outputs=num_outputs,
                            global_pool=nn.AdaptiveAvgPool1d,
                            max_pool=True)

    def SimpleResNet_8blocks_width4(self, num_outputs):
        return SimpleResNet(block=BasicBlock,
                            in_features=1,
                            out_features=(64, 128, 256, 512, 512, 512, 512, 512),
                            num_outputs=num_outputs,
                            global_pool=nn.AdaptiveAvgPool1d,
                            max_pool=True)

    # ---- DEEPER VERSIONS ----

    def SimpleResNet_4blocks(self, num_outputs):
        return SimpleResNet(block=BasicBlock,
                            in_features=1,
                            out_features=(16, 32, 64, 128),
                            num_outputs=num_outputs,
                            global_pool=nn.AdaptiveAvgPool1d,
                            max_pool=True)

    def SimpleResNet_6blocks(self, num_outputs):
        return SimpleResNet(block=BasicBlock,
                            in_features=1,
                            out_features=(16, 32, 64, 128, 128, 128),
                            num_outputs=num_outputs,
                            global_pool=nn.AdaptiveAvgPool1d,
                            max_pool=True)

    def SimpleResNet_12blocks(self, num_outputs):
        return SimpleResNet(block=BasicBlock,
                            in_features=1,
                            out_features=(16, 32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128),
                            num_outputs=num_outputs,
                            global_pool=nn.AdaptiveAvgPool1d,
                            max_pool=True)

    def SimpleResNet_16blocks(self, num_outputs):
        return SimpleResNet(block=BasicBlock,
                            in_features=1,
                            out_features=(16, 32, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
                            num_outputs=num_outputs,
                            global_pool=nn.AdaptiveAvgPool1d,
                            max_pool=True)

    # 1D-RESNET
    def _resnet(self, block, layers, **kwargs):
        model = ResNet(block, layers, **kwargs)
        return model

    def resnet10(self, **kwargs):
        r"""ResNet-10 model, a smaller adaptation of the ResNet-18 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return self._resnet(BasicBlock, [1, 1, 1, 1], **kwargs)

    def resnet18(self, **kwargs):
        r"""ResNet-18 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return self._resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

    def resnet34(self, **kwargs):
        r"""ResNet-34 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return self._resnet(BasicBlock, [3, 4, 6, 3], **kwargs)

    def resnet50(self, **kwargs):
        r"""ResNet-50 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return self._resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

    def resnet101(self, **kwargs):
        r"""ResNet-101 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return self._resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

    def resnet152(self, **kwargs):
        r"""ResNet-152 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return self._resnet(Bottleneck, [3, 8, 36, 3], **kwargs)

    def resnext50_32x4d(self, **kwargs):
        r"""ResNeXt-50 32x4d model from
        `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
        """
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        return self._resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

    def resnext101_32x8d(self, **kwargs):
        r"""ResNeXt-101 32x8d model from
        `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
        """
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        return self._resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

    def wide_resnet50_2(self, **kwargs):
        r"""Wide ResNet-50-2 model from
        `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
        The model is the same as ResNet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
        channels, and in Wide ResNet-50-2 has 2048-1024-2048.
        """
        kwargs['width_per_group'] = 64 * 2
        return self._resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

    def wide_resnet101_2(self, **kwargs):
        r"""Wide ResNet-101-2 model from
        `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
        The model is the same as ResNet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
        channels, and in Wide ResNet-50-2 has 2048-1024-2048.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs['width_per_group'] = 64 * 2
        return self._resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
