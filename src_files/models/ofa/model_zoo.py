# Adopted from https://github.com/mit-han-lab/once-for-all:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
# ------------------------------------------------------------------
# Main change from original model - hard-sigmoid layers converted to regular sigmoids (faster, and give better scores)
# Hence we renamed model from ofa_flops_595m to ofa_flops_595m_s.
# ------------------------------------------------------------------

import json
import os

import src_files.models.ofa.utils
from src_files.models.ofa.layers import *
from src_files.models.ofa.utils import MyNetwork, MyModule


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier):
        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = self.pool(x)
        # x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = self.classifier(x)
        x = x.view(int(x.size(0)), int(x.size(1)))

        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut,
                                                                                          IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()


def ofa_specialized(net_id, num_classes=None):
    import src_files.models.ofa.layers as layers

    layers.build_activation = src_files.models.ofa.utils.build_activation
    layers.SEModule = src_files.models.ofa.utils.SEModule

    net_config = json.load(
        open(os.path.join('./src_files/models/ofa/specialized_models_configs/' + net_id, 'net.config')))
    net_config['classifier']['out_features'] = num_classes
    net = MobileNetV3.build_from_config(net_config)

    # model init
    net.init_model('he_fout')

    image_size = \
        json.load(open(os.path.join('./src_files/models/ofa/specialized_models_configs/' + net_id, 'run.config')))[
            'image_size']
    return net, image_size



def ofa_flops_595m_s(model_params):
    net, image_size = ofa_specialized("flops@595M_top1@80.0_finetune@75",
                                      num_classes=model_params['num_classes'])
    return net

