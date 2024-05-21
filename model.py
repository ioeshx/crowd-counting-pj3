import torch.nn as nn
import torch
from torchvision import models

# 采用CSRnet-A 
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        # 修改前端层的输入通道数为6
        self.frontend = make_layers(self.frontend_feat, in_channels=6)
        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # 从预训练模型加载权重
            pretrained_state_dict = mod.features.state_dict()
            frontend_state_dict = self.frontend.state_dict()

            # 处理预训练权重，使其适应前6个通道
            for key in frontend_state_dict:
                if 'weight' in key and frontend_state_dict[key].shape != pretrained_state_dict.get(key, frontend_state_dict[key]).shape:
                    if key == '0.weight':  # 只调整第一个卷积层的权重
                        pretrained_weight = pretrained_state_dict[key]
                        # 构建新的权重张量，增加通道数
                        new_weight = torch.zeros_like(frontend_state_dict[key])
                        new_weight[:, :3, :, :] = pretrained_weight  # 前3个通道使用预训练权重
                        new_weight[:, 3:, :, :] = pretrained_weight  # 后3个通道复制前3个通道的权重
                        frontend_state_dict[key] = new_weight
                else:
                    if key in pretrained_state_dict:
                        frontend_state_dict[key] = pretrained_state_dict[key]

            self.frontend.load_state_dict(frontend_state_dict)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
