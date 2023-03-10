from abc import *
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, simclr_dim=128, pred_layer=None):
        super(BaseModel, self).__init__()
        
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )
        if pred_layer is None:
            self.shift_cls_layer = nn.Linear(last_dim, 2)
            self.linear = nn.Linear(last_dim, num_classes)
            self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)
            self.layer_type = nn.Linear
        else:
            self.shift_cls_layer = pred_layer(last_dim, 2)
            self.linear = pred_layer(last_dim, num_classes)
            self.joint_distribution_layer = pred_layer(last_dim, 4 * num_classes)
            self.layer_type = pred_layer


    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def set_shift_cls_layer(self, k):
        self.shift_cls_layer = self.layer_type(self.last_dim, k)
        return self

    def forward(self, inputs, penultimate=False, simclr=False, shift=False, joint=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)

        output = self.linear(features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)

        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features)

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer(features)

        if _return_aux:
            return output, _aux

        return output
