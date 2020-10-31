import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Tanh
from torch.nn.functional import softmax
from .base_nets import BaseRegressor, BaseClassifier, BaseNet
from .mi_nets import MainNet


class WeightsDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, w):
        if self.p == 0:
            return w
        d0 = [[i] for i in range(len(w))]
        d1 = w.argsort(dim=2)[:, :, :int(w.shape[2] * self.p)]
        d1 = [i.reshape(1, -1)[0].tolist() for i in d1]
        #
        w_new = w.clone()
        w_new[d0, :, d1] = 0
        #
        d1 = [i[0].nonzero().flatten().tolist() for i in w_new]
        w_new[d0, :, d1] = Softmax(dim=1)(w_new[d0, :, d1])
        return w_new


class SelfAttention(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.w_query = nn.Linear(inp_dim, out_dim)
        self.w_key = nn.Linear(inp_dim, out_dim)
        self.w_value = nn.Linear(inp_dim, out_dim)

    def forward(self, x):
        keys = self.w_key(x)
        querys = self.w_query(x)
        values = self.w_value(x)

        att_weights = softmax(querys @ torch.transpose(keys, 2, 1), dim=-1)
        weighted_values = values[:, :, None] * torch.transpose(att_weights, 2, 1)[:, :, :, None]
        outputs = weighted_values.sum(dim=1)

        return outputs


class AttentionNet(BaseNet):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.estimator = Linear(ndim[-1], 1)
        #
        input_dim = ndim[-1]
        attention = []
        for dim in det_ndim:
            attention.append(Linear(input_dim, dim))
            attention.append(Sigmoid())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        self.detector = Sequential(*attention)

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()


    def forward(self, x, m):
        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det)
        w = WeightsDropout(p=self.dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class GlobalTempAttentionNet(BaseNet):

    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.estimator = Linear(ndim[-1], 1)
        #
        input_dim = ndim[-1]
        attention = []
        for dim in det_ndim:
            attention.append(Linear(input_dim, dim))
            attention.append(Sigmoid())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        self.detector = Sequential(*attention)

        self.temp = torch.nn.Parameter(torch.Tensor([0.1]))
        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()


    def forward(self, x, m):
        temp = self.temp.to(x.device)

        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det / temp)
        w = WeightsDropout(p=self.dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class TempAttentionNet(BaseNet):

    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.estimator = Linear(ndim[-1], 1)
        #
        input_dim = ndim[-1]
        attention = []
        for dim in det_ndim:
            attention.append(Linear(input_dim, dim))
            attention.append(Sigmoid())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        self.detector = Sequential(*attention)

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()


    def forward(self, x, m):

        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det / 0.1)
        w = WeightsDropout(p=self.dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class SelfAttentionNet(BaseNet):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.self_attention = SelfAttention(ndim[-1], ndim[-1])
        self.detector = Sequential(Linear(ndim[-1], det_ndim[0]), Tanh(), Linear(det_ndim[0], 1))
        self.estimator = Linear(ndim[-1], 1)

        if init_cuda:
            self.main_net.cuda()
            self.self_attention.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)
        x = self.self_attention(x)

        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det)
        w = WeightsDropout(p=self.dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class GatedAttentionNet(AttentionNet, BaseNet):
    def __init__(self, ndim=None, det_ndim=None,  init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim,  init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.attention_V = Sequential(Linear(ndim[-1], det_ndim[0]), Tanh())
        self.attention_U = Sequential(Linear(ndim[-1], det_ndim[0]), Sigmoid())
        self.detector = Linear(det_ndim[0], 1)
        self.estimator = Linear(ndim[-1], 1)

        if init_cuda:
            self.main_net.cuda()
            self.attention_V.cuda()
            self.attention_U.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x, m):
        x = self.main_net(x)
        w_v = self.attention_V(x)
        w_u = self.attention_U(x)

        x_det = torch.transpose(m * self.detector(w_v * w_u), 2, 1)
        w = Softmax(dim=2)(x_det)
        w = WeightsDropout(p=self.dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class AttentionNetClassifier(AttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class AttentionNetRegressor(AttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class SelfAttentionNetClassifier(SelfAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class SelfAttentionNetRegressor(SelfAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GatedAttentionNetClassifier(GatedAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class GatedAttentionNetRegressor(GatedAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)

class TempAttentionNetRegressor(TempAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)

class TempAttentionNetClassifier(TempAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)

class GlobalTempAttentionNetRegressor(GlobalTempAttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)

class GlobalTempAttentionNetClassifier(GlobalTempAttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


