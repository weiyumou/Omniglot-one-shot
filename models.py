import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TripletNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64,
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # (N, 64, 6, 6)
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (N, 64, 1, 1)
        # self.fc = nn.Linear(in_features=64*6*6, out_features=128)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        # conv4_out = self.conv4(conv3_out)
        avg_pool_out = self.avg_pool(conv3_out)
        net_out = avg_pool_out.reshape(avg_pool_out.size(0), -1)
        return net_out


class TripletNetWithFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.PReLU(num_parameters=128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.fc = nn.Linear(in_features=128 * 3 * 3, out_features=256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv4_out = conv4_out.reshape(conv4_out.size(0), -1)
        # fc_out = self.fc(conv3_out)
        return conv4_out, conv3_out, conv2_out, conv1_out


class MetricNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=128 * 2, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(F.relu(fc1_out))
        return fc2_out

# class MetricNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=64,
#                       kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64,
#                       kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128,
#                       kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_features=128 * 3 * 3, out_features=512),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(in_features=512, out_features=1),
#             # nn.Softplus(beta=0.1)
#             losses.SlideFunc()
#         )
#
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv2_out = self.conv2(conv1_out)
#         conv3_out = self.conv3(conv2_out)
#         conv3_out = conv3_out.reshape(conv3_out.size(0), -1)
#         fc1_out = self.fc1(conv3_out)
#         fc2_out = self.fc2(fc1_out)
#         return fc2_out


class BrendenNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=120, kernel_size=5, bias=False),  # 24
            nn.BatchNorm2d(num_features=120),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 12
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=300, kernel_size=5, bias=False),  # 8
            nn.BatchNorm2d(num_features=300),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 4
        )
        self.fc = nn.Linear(in_features=300 * 4 * 4, out_features=3000)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv2_out = conv2_out.reshape(conv2_out.size(0), -1)
        fc_out = self.fc(conv2_out)
        return fc_out


class ResnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,
                                      stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(in_features=256 * 2 * 2, out_features=128)

        self.conv = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

    def forward(self, x):
        conv_out = self.conv(x)
        layer1_out = self.resnet.layer1[0](conv_out)
        layer2_out = self.resnet.layer2[0](layer1_out)
        layer3_out = self.resnet.layer3[0](layer2_out)
        layer3_out = layer3_out.reshape(layer3_out.size(0), -1)
        fc_out = self.resnet.fc(layer3_out)
        return fc_out


class KochNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv4_out = conv4_out.reshape(conv4_out.size(0), -1)
        fc_out = self.fc(conv4_out)
        return fc_out


class ResiduleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=128),
        )
        self.downsample256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=256),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.conv2 = torchvision.models.resnet.BasicBlock(64, 128, 2, self.downsample128)
        self.conv3 = torchvision.models.resnet.BasicBlock(128, 256, 2, self.downsample256)
        self.fc = nn.Linear(in_features=256*4*4, out_features=1024)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv3_out = conv3_out.reshape(conv3_out.size(0), -1)
        fc_out = self.fc(conv3_out)
        return fc_out
