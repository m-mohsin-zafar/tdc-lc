from collections import OrderedDict
import torch
import torch.nn.functional as F
from detectron2.engine import DefaultPredictor
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout, init, AvgPool2d, LeakyReLU, Dropout2d, ModuleDict, ZeroPad2d, ReflectionPad2d, AdaptiveMaxPool2d
from models.configs import Configurations


class PCNet3P2(Module):

    def __init__(self, in_channels, num_classes=3):
        super(PCNet3P2, self).__init__()

        self.model_name = 'PCNetP2'
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Entry block to capture image semantics with a high receptive field
        self.entry_block = Sequential(
            OrderedDict(
                [
                    (
                    'input_conv_layer', Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=5, stride=1)),
                    ('batch_norm_1', BatchNorm2d(32)),
                    ('relu_1', ReLU(inplace=True)),
                    ('conv_layer_ex2',
                     Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=2)),
                    ('batch_norm_2', BatchNorm2d(64)),
                    ('relu_2', ReLU(inplace=True)),
                    ('conv_layer_ex3',
                     Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0, dilation=2)),
                    ('batch_norm_3', BatchNorm2d(96)),
                    ('relu_3', ReLU(inplace=True)),
                    ('conv_layer_ex4',
                     Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1)),
                    ('batch_norm_4', BatchNorm2d(128)),
                    ('relu_4', ReLU(inplace=True)),
                ]
            )
        )

        # Introducing channel dropout to force learning of independent features in channels
        # self.dropout2d_1 = Dropout2d(p=0.3)

        # 2 Paths into two res2incept blocks. Path A and Path B
        block_params_1xA = {
            'A': {
                'krnl_size': (5, 1),
                'activ': 'relu'
            },
            'B': {
                'krnl_size': (3, 1),
                'activ': 'relu'
            },
            'C': {
                'krnl_size': (3, 1),
                'activ': 'relu'
            },
            'D': {
                'krnl_size': (2, 1)
            }
        }
        block_params_1xB = {
            'A': {
                'krnl_size': (1, 5),
                'activ': 'relu'
            },
            'B': {
                'krnl_size': (1, 3),
                'activ': 'relu'
            },
            'C': {
                'krnl_size': (1, 3),
                'activ': 'relu'
            },
            'D': {
                'krnl_size': (1, 2)
            }
        }

        self.res2incept_1xA = Res2InceptBlock(in_channels=128, out_channels=192, num_sub_blocks=4,
                                              use_sub_blocks_type=('A', 'B', 'C', 'D'),
                                              sub_blocks_params=block_params_1xA)
        self.res2incept_1xB = Res2InceptBlock(in_channels=128, out_channels=192, num_sub_blocks=4,
                                              use_sub_blocks_type=('A', 'B', 'C', 'D'),
                                              sub_blocks_params=block_params_1xB)

        self.middle_block = Sequential(
            OrderedDict(
                [
                    ('relu_m0', ReLU(inplace=True)),
                    ('conv_layer_mx1',
                     Conv2d(in_channels=384, out_channels=416, kernel_size=3, stride=2, padding=0, dilation=1)),
                    ('relu_5', ReLU(inplace=True)),
                    ('conv_layer_mx2',
                     Conv2d(in_channels=416, out_channels=448, kernel_size=3, stride=1, padding=0, dilation=2)),
                    ('batch_norm_3', BatchNorm2d(448)),
                    ('relu_6', ReLU(inplace=True)),
                    ('conv_layer_mx3',
                     Conv2d(in_channels=448, out_channels=480, kernel_size=3, stride=1, padding=0, dilation=4)),
                    ('relu_7', ReLU(inplace=True)),
                    ('conv_layer_mx4',
                     Conv2d(in_channels=480, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1)),
                    ('batch_norm_4', BatchNorm2d(512)),
                    ('relu_8', ReLU(inplace=True)),
                ]
            )
        )

        # Introducing channel dropout to force learning of independent features in channels
        # self.dropout2d_2 = Dropout2d(p=0.3)

        # 2 Paths into two res2incept blocks. Path A and Path B
        block_params_2xA = {
            'A': {
                'krnl_size': (5, 1),
                'activ': 'relu'
            },
            'B': {
                'krnl_size': (3, 1),
                'activ': 'relu'
            },
            'C': {
                'krnl_size': (3, 1),
                'activ': 'relu'
            },
            'D': {
                'krnl_size': (2, 1)
            }
        }
        block_params_2xB = {
            'A': {
                'krnl_size': (1, 5),
                'activ': 'relu'
            },
            'B': {
                'krnl_size': (1, 3),
                'activ': 'relu'
            },
            'C': {
                'krnl_size': (1, 3),
                'activ': 'relu'
            },
            'D': {
                'krnl_size': (1, 2)
            }
        }

        self.res2incept_2xA = Res2InceptBlock(in_channels=512, out_channels=640, num_sub_blocks=4,
                                              use_sub_blocks_type=('A', 'B', 'C', 'D'),
                                              sub_blocks_params=block_params_2xA)
        self.res2incept_2xB = Res2InceptBlock(in_channels=512, out_channels=640, num_sub_blocks=4,
                                              use_sub_blocks_type=('A', 'B', 'C', 'D'),
                                              sub_blocks_params=block_params_2xB)

        self.bottleneck_block = Sequential(
            OrderedDict(
                [
                    ('relu_b0', ReLU(inplace=True)),
                    ('conv_layer_bx1',
                     Conv2d(in_channels=1280, out_channels=1536, kernel_size=3, stride=2, padding=1, dilation=1)),
                    ('batch_norm_5', BatchNorm2d(1536)),
                    ('relu_9', ReLU(inplace=True)),
                    ('conv_layer_bx2',
                     Conv2d(in_channels=1536, out_channels=1792, kernel_size=3, stride=1, padding=0, dilation=2)),
                    ('batch_norm_6', BatchNorm2d(1792)),
                    ('relu_10', ReLU(inplace=True)),
                    ('conv_layer_bx3',
                     Conv2d(in_channels=1792, out_channels=2048, kernel_size=3, stride=2, padding=1, dilation=1)),
                    ('batch_norm_7', BatchNorm2d(2048)),
                    ('relu_11', ReLU(inplace=True)),
                ]
            )
        )
        self.conv_features = Sequential(OrderedDict(
            [
                ('max_pool_layer', AdaptiveMaxPool2d(1))
            ]
        ))

        self.classification_layers = Sequential(OrderedDict(
            [
                ('linear_1', Linear(2048 * 1 * 1, out_features=self.num_classes))
            ]
        ))

    def forward(self, x):
        # Shape = (Batch_size, 3, 256, 256)
        x = self.entry_block(x)
        # x = self.dropout2d_1(x)
        # Shape = (Batch_size, 128, 122, 122)
        x_a = self.res2incept_1xA(x)
        # Shape = (Batch_size, 192, 122, 122)
        x_b = self.res2incept_1xB(x)
        # Shape = (Batch_size, 192, 122, 122)
        x = torch.cat([x_a, x_b], dim=1)
        # x = x_a + x_b
        # Shape = (Batch_size, 384, 122, 122)
        x = self.middle_block(x)
        # x = self.dropout2d_2(x)
        # Shape = (Batch_size, 512, 24, 24)
        x_a = self.res2incept_2xA(x)
        # Shape = (Batch_size, 640, 24, 24)
        x_b = self.res2incept_2xB(x)
        # Shape = (Batch_size, 640, 24, 24)
        x = torch.cat([x_a, x_b], dim=1)
        # x = x_a + x_b
        # Shape = (Batch_size, 1280, 24, 24)
        x = self.bottleneck_block(x)
        # Shape = (Batch_size, 2048, 4, 4)
        x = self.conv_features(x)
        # Shape = (Batch_size, 2048, 1, 1)
        x = x.reshape(x.size(0), -1)
        # Shape = (Batch_size, 2048)
        x = self.classification_layers(x)
        # Shape = (Batch_size, num_classes)

        return x


class Res2InceptBlock(Module):

    def __init__(self, in_channels, out_channels, num_sub_blocks=4, use_sub_blocks_type=('A', 'B', 'C', 'D'),
                 sub_blocks_params=None):
        super(Res2InceptBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sub_blocks = num_sub_blocks
        self.use_blocks_type = use_sub_blocks_type
        self.sub_blocks_params = sub_blocks_params

        assert self.num_sub_blocks == len(
            self.use_blocks_type), 'Number of Sub Blocks must be equal to Number of types of Blocks to be used!'

        self.conv_1x1 = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, 1))
        # Here, we shall initialize sub blocks in accordance with their paramters
        self.sub_blocks = ModuleDict()
        sub_blocks_dict = {}
        for k in use_sub_blocks_type:
            if k == 'A':
                sub_blocks_dict['block_A'] = SubBlockTypeA(
                    in_chs=self.out_channels,
                    out_chs=self.out_channels // self.num_sub_blocks,
                    krnl_size=self.sub_blocks_params[k]['krnl_size'],
                    activ=self.sub_blocks_params[k]['activ']
                )
            elif k == 'B':
                sub_blocks_dict['block_B'] = SubBlockTypeB(
                    in_chs=self.out_channels,
                    out_chs=self.out_channels // self.num_sub_blocks,
                    krnl_size=self.sub_blocks_params[k]['krnl_size'],
                    activ=self.sub_blocks_params[k]['activ']
                )
            elif k == 'C':
                sub_blocks_dict['block_C'] = SubBlockTypeC(
                    in_chs=self.out_channels,
                    out_chs=self.out_channels // self.num_sub_blocks,
                    krnl_size=self.sub_blocks_params[k]['krnl_size'],
                    activ=self.sub_blocks_params[k]['activ']
                )
            elif k == 'D':
                sub_blocks_dict['block_D'] = SubBlockTypeD(
                    in_chs=self.out_channels,
                    out_chs=self.out_channels // self.num_sub_blocks,
                    krnl_size=self.sub_blocks_params[k]['krnl_size'],
                )

        self.sub_blocks.update(sub_blocks_dict)

    def forward(self, x):
        x = self.conv_1x1(x)

        # Forward pass will be according to the initialized sub blocks
        sub_block_outputs = []
        for k in self.sub_blocks.keys():
            sub_block_outputs.append(self.sub_blocks[k](x))

        # Dimension = 1 represents channel in this case
        x = torch.cat(sub_block_outputs, dim=1)

        return x


class SubBlockTypeA(Module):

    def __init__(self, in_chs, out_chs, krnl_size, activ='relu'):
        super(SubBlockTypeA, self).__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.krnl_size = krnl_size

        if activ == 'relu':
            self.activ_func = ReLU(inplace=True)
        elif activ == 'leaky_relu':
            self.activ_func = LeakyReLU(negative_slope=0.2, inplace=True)

        # Due to asymmetric filter we need to calculate padding which may be done using the following formula
        # kernel_sizes = (1, 2)
        # conv_padding = [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]]
        padding = None
        if self.krnl_size == (2, 1):
            padding = (0, 0, 0, 1)
        elif self.krnl_size == (1, 2):
            padding = (0, 1, 0, 0)
        elif self.krnl_size == (1, 5):
            padding = (2, 2, 0, 0)
        elif self.krnl_size == (5, 1):
            padding = (0, 0, 2, 2)

        self.pad = ReflectionPad2d(padding=padding)

        self.conv_1x1 = Conv2d(in_channels=self.in_chs, out_channels=self.out_chs, kernel_size=(1, 1))
        self.main_path = Sequential(OrderedDict(
            [
                ('padding_1', self.pad),
                ('conv_1', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                                  padding=0, stride=1)),
                ('padding_2', self.pad),
                ('conv_2', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                                  padding=0, stride=1)),
                ('bn_1', BatchNorm2d(self.out_chs)),
                ('activation_1', self.activ_func),
                ('padding_3', self.pad),
                ('conv_3', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                                  padding=0, stride=1)),
                ('padding_4', self.pad),
                ('conv_4', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                                  padding=0, stride=1)),
                ('bn_2', BatchNorm2d(self.out_chs)),
            ]
        ))

    def forward(self, x):

        x = self.conv_1x1(x)
        identity = x
        x = self.main_path(x)
        aggr_x = x + identity

        return aggr_x


class SubBlockTypeB(Module):

    def __init__(self, in_chs, out_chs, krnl_size, activ='relu'):
        super(SubBlockTypeB, self).__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.krnl_size = krnl_size
        padding = None
        if self.krnl_size == (2, 1):
            padding = (0, 0, 0, 1)
        elif self.krnl_size == (1, 2):
            padding = (0, 1, 0, 0)
        elif self.krnl_size == (1, 3):
            padding = (1, 1, 0, 0)
        elif self.krnl_size == (3, 1):
            padding = (0, 0, 1, 1)
        self.pad = ReflectionPad2d(padding=padding)

        if activ == 'relu':
            self.activ_func = ReLU(inplace=True)
        elif activ == 'leaky_relu':
            self.activ_func = LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_1x1 = Conv2d(in_channels=self.in_chs, out_channels=self.out_chs, kernel_size=(1, 1))
        self.main_path = Sequential(OrderedDict(
            [
                ('padding_1', self.pad),
                ('conv_1', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                                     padding=0, stride=1)),
                ('bn_1', BatchNorm2d(self.out_chs)),
                ('activation_1', self.activ_func),
                ('padding_2', self.pad),
                ('conv_2', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                             padding=0, stride=1)),
                ('padding_3', self.pad),
                ('conv_3', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                             padding=0, stride=1)),
                ('bn_2', BatchNorm2d(self.out_chs)),
            ]
        ))

    def forward(self, x):

        x = self.conv_1x1(x)
        identity = x
        x = self.main_path(x)
        aggr_x = x + identity

        return aggr_x


class SubBlockTypeC(Module):

    def __init__(self, in_chs, out_chs, krnl_size, activ='relu'):
        super(SubBlockTypeC, self).__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.krnl_size = krnl_size
        padding = None
        if self.krnl_size == (2, 1):
            padding = (0, 0, 0, 1)
        elif self.krnl_size == (1, 2):
            padding = (0, 1, 0, 0)
        elif self.krnl_size == (1, 3):
            padding = (1, 1, 0, 0)
        elif self.krnl_size == (3, 1):
            padding = (0, 0, 1, 1)
        self.pad = ReflectionPad2d(padding=padding)

        if activ == 'relu':
            self.activ_func = ReLU(inplace=True)
        elif activ == 'leaky_relu':
            self.activ_func = LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_1x1 = Conv2d(in_channels=self.in_chs, out_channels=self.out_chs, kernel_size=(1, 1))
        self.main_path = Sequential(OrderedDict(
            [
                ('padding_1', self.pad),
                ('conv_1', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                                 padding=0, stride=1)),
                ('bn_1', BatchNorm2d(self.out_chs)),
                ('activation_1', self.activ_func),
                ('padding_2', self.pad),
                ('conv_2', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                                 padding=0, stride=1)),
                ('bn_2', BatchNorm2d(self.out_chs)),
            ]
        ))

    def forward(self, x):

        x = self.conv_1x1(x)
        identity = x
        x = self.main_path(x)
        aggr_x = x + identity

        return aggr_x


class SubBlockTypeD(Module):

    def __init__(self, in_chs, out_chs, krnl_size):
        super(SubBlockTypeD, self).__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.krnl_size = krnl_size
        padding = None
        if self.krnl_size == (2, 1):
            padding = (0, 0, 0, 1)
        elif self.krnl_size == (1, 2):
            padding = (0, 1, 0, 0)
        self.pad = ReflectionPad2d(padding=padding)

        self.conv_1x1 = Conv2d(in_channels=self.in_chs, out_channels=self.out_chs, kernel_size=(1, 1))
        self.main_path = Sequential(OrderedDict(
            [
                ('padding_1', self.pad),
                ('conv_1', Conv2d(in_channels=self.out_chs, out_channels=self.out_chs, kernel_size=self.krnl_size,
                             padding=0, stride=1)),
                ('bn_1', BatchNorm2d(self.out_chs))
            ]
        ))

    def forward(self, x):
        x = self.conv_1x1(x)
        identity = x
        x = self.main_path(x)
        aggr_x = x + identity

        return aggr_x


class BuildModels:

    def __init__(self):
        _ckpts_root = './checkpoints/'
        self._clf_ckpt = _ckpts_root + 'model_multi_01_PCNetP2_best_weights.pth'
        self._det_ckpt = _ckpts_root + 'model_final.pth'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # empty cuda cache
        torch.cuda.empty_cache()

        self._clf_model = self._build_model(model_type='clf')
        self._det_model = self._build_model(model_type='det')

    def _build_model(self, model_type):
        model = None
        if model_type == 'clf':
            model = PCNet3P2(in_channels=3, num_classes=3)
            model.to(device=self.device)
            model.load_state_dict(torch.load(self._clf_ckpt, map_location=self.device))
            model.eval()
        elif model_type == 'det':
            mc = Configurations(model_name='mrcnn', param_key='mrcnn_circular_dab', device=self.device)
            cfg = mc.get_configurations()
            model = DefaultPredictor(cfg)

        return model

    def get_classifier(self):
        return self._clf_model

    def get_detector(self):
        return self._det_model


if __name__ == '__main__':
    models = BuildModels()
    _clf = models.get_classifier()
    _det = models.get_detector()
