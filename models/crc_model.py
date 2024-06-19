import timm
import torch.nn as nn

from models.config import IMG_SIZE, GRID_SIZE


class PatchClsHead(nn.Module):

    def __init__(self, in_channels, out_channels=2):
        super(PatchClsHead, self).__init__()

        # conv1
        self.conv1 = nn.Conv2d(in_channels, 1024, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=True)

        # conv2
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)

        # classification layer
        self.conv3 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)

        return x


class CRCModel(nn.Module):

    def __init__(self, model_name='convnextv2_large.fcmae_ft_in22k_in1k', num_classes=2, p_cls=False, test_cam=False):
        super().__init__()

        self.p_cls = p_cls  # patch classification
        self.test_cam = test_cam  # test cam

        # convnextv2
        self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=num_classes, in_chans=1)

        # image-level classification
        self.classifier = nn.Linear(self.model.num_features, num_classes)

        # patch-level classification
        self.patch_cls = PatchClsHead(in_channels=self.model.num_features)

    def forward(self, x1, x2=None, loopback=False):

        # x1:image (B, C, H, W) or (1, D, C, H, W)
        # x2:volume (B, 1) or (B, C, H, W, 2*slice+1)/ or (1, D, C, H, W, 2*slice+1)

        if self.test_cam:
            if self.p_cls:
                out1 = self.model.forward_features(x1)  # features
                out1_cls = self.model.forward_head(out1)
                out1_p_cls = self.patch_cls(out1)
                return out1_cls, out1_p_cls
            else:
                out1 = self.model.forward_features(x1)  # features
                out1_cls = self.model.forward_head(out1)
                return out1_cls
        else:
            x2_dim = x2.dim()
            if x1.dim() == 5:
                x1, x2 = x1[0], x2[0]

            out1 = self.model.forward_features(x1)  # patch features
            out1_cls = self.model.forward_head(out1)  # [B, 2]

            # loopback or classification 
            if loopback or x2_dim == 2:
                return out1_cls

            # patch cls
            if self.p_cls:
                out1_p_cls = self.patch_cls(out1)  # patch features
                x2 = x2.permute(0, 4, 1, 2, 3).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
                out2_patch = self.model.forward_features(x2)
                outnei_p_cls = self.patch_cls(out2_patch).reshape(x1.shape[0], -1, 2, GRID_SIZE, GRID_SIZE)

                return out1_cls, out1_p_cls, outnei_p_cls
