from __future__ import absolute_import
from __future__ import division
from .meta_model import MetaModel
from ...resnet_drop import resnet12
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from ...dconv.layers import DeformConv
from torchdiffeq import odeint as odeint
from core.utils import accuracy

# from core.model.metric.metric_model import MetricModel
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        input_ = inputs
        input_ = input_.view(input_.size(0), input_.size(1), -1)

        log_probs = self.logsoftmax(input_)
        targets_ = torch.zeros(input_.size(0), input_.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets_ = targets_.unsqueeze(-1)
        targets_ = targets_.cuda()
        loss = (- targets_ * log_probs).mean(0).sum() 
        return loss / input_.size(2)

def one_hot(indices, depth, use_cuda=True):
    if use_cuda:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    else:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies

def shuffle(images, targets, global_targets):
    """
    A trick for CAN training
    """
    sample_num = images.shape[1]
    for i in range(4):
        indices = torch.randperm(sample_num).to(images.device)
        images = images.index_select(1, indices)
        targets = targets.index_select(1, indices)
        global_targets = global_targets.index_select(1, indices)
    return images, targets, global_targets

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class DynamicWeights_(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1):
        super(DynamicWeights_, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

        padding = 1 if kernel == 3 else 0
        offset_groups = 1
        self.off_conv = nn.Conv2d(channels * 2, 3 * 3 * 2, 5,
                                  padding=2, dilation=dilation, bias=False)
        self.kernel_conv = DeformConv(channels, groups * kernel * kernel,
                                      kernel_size=3, padding=dilation, dilation=dilation, bias=False)

        self.K = kernel * kernel
        self.group = groups

    def forward(self, support, query):
        N, C, H, W = support.size()
        R = C // self.group
        offset = self.off_conv(torch.cat([query, support], 1))
        dynamic_filter = self.kernel_conv(support, offset)
        dynamic_filter = F.sigmoid(dynamic_filter)
        return dynamic_filter


class DynamicWeights(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1, nFeat=640):
        super(DynamicWeights, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        padding = 1 if kernel == 3 else 0
        offset_groups = 1
        self.unfold = nn.Unfold(kernel_size=(kernel, kernel),
                                padding=padding, dilation=1)

        self.K = kernel * kernel
        self.group = groups
        self.nFeat = nFeat

    def forward(self, t=None, x=None):
        query, dynamic_filter = x
        N, C, H, W = query.size()
        N_, C_, H_, W_ = dynamic_filter.size()
        R = C // self.group
        dynamic_filter = dynamic_filter.reshape(-1, self.K)

        xd_unfold = self.unfold(query)

        xd_unfold = xd_unfold.view(N, C, self.K, H * W)
        xd_unfold = xd_unfold.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2,
                                                                                                             4).contiguous().view(
            N * self.group * H * W, R, self.K)
        out1 = torch.bmm(xd_unfold, dynamic_filter.unsqueeze(2))
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(
            N, self.group * R, H, W)

        out1 = F.relu(out1)
        return (out1, torch.zeros([N_, C_, H_, W_]).cuda())


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x[0])
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-2, atol=1e-2, method='rk4')
        return out[0][1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class Model(nn.Module):
    def __init__(self, num_classes=64, kernel=3, groups=1):
        super(Model, self).__init__()
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.global_clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

        self.dw_gen = DynamicWeights_(self.nFeat, 1, kernel, groups)
        self.dw = self.dw = ODEBlock(DynamicWeights(self.nFeat, 1, kernel, groups, self.nFeat))

    def reshape(self, ftrain, ftest):
        b, n1, c, h, w = ftrain.shape
        n2 = ftest.shape[1]
        ftrain = ftrain.unsqueeze(2).repeat(1, 1, n2, 1, 1, 1)
        ftest = ftest.unsqueeze(1).repeat(1, n1, 1, 1, 1, 1)
        return ftrain, ftest

    def process_feature(self, f, ytrain, num_train, num_test, batch_size):
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest = self.reshape(ftrain, ftest)

        # b, n2, n1, c, h, w
        ftrain = ftrain.transpose(1, 2)
        ftest = ftest.transpose(1, 2)
        return ftrain, ftest

    def get_score(self, ftrain, ftest, num_train, num_test, batch_size):
        b, n2, n1, c, h, w = ftrain.shape

        ftrain_ = ftrain.clone()
        ftest_ = ftest.clone()
        ftrain_ = ftrain_.reshape(-1, *ftrain.size()[3:])
        ftest_ = ftest_.reshape(-1, *ftest.size()[3:])

        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.reshape(-1, *ftrain_norm.size()[3:])
        conv_weight = ftrain_norm.mean(-1, keepdim=True).mean(-2, keepdim=True)

        filter_weight = self.dw_gen(ftrain_, ftest_)
        cls_scores = self.dw(x=(ftest_, filter_weight))
        cls_scores = cls_scores.view(b * n2, n1, *cls_scores.size()[1:])
        cls_scores = cls_scores.view(1, -1, *cls_scores.size()[3:])
        cls_scores = F.conv2d(cls_scores, conv_weight, groups=b * n1 * n2, padding=1)
        cls_scores = cls_scores.view(b * n2, n1, *cls_scores.size()[2:])
        return cls_scores

    def get_global_pred(self, ftest, ytest, num_test, batch_size, K):
        h = ftest.shape[-1]
        ftest_ = ftest.view(batch_size, num_test, K, -1)
        ftest_ = ftest_.transpose(2, 3)
        ytest_ = ytest.unsqueeze(3)
        ftest_ = torch.matmul(ftest_, ytest_)
        ftest_ = ftest_.view(batch_size * num_test, -1, h, h)
        global_pred = self.global_clasifier(ftest_)
        return global_pred

    def get_test_score(self, score_list):
        return score_list.mean(-1).mean(-1)

    def forward(self, xtrain, xtest, ytrain, ytest, global_labels=None):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        ftrain, ftest = self.process_feature(f, ytrain, num_train,
                                                num_test, batch_size)
        cls_scores = self.get_score(ftrain, ftest,
                                             num_train, num_test, batch_size)

        if not self.training:
            return self.get_test_score(cls_scores)

        global_pred = self.get_global_pred(ftest, ytest, num_test, batch_size, K)
        return global_pred, cls_scores

class DynamicWeightsModel(MetaModel):
    def __init__(self,test_way,test_shot,test_query, way_num, query_num, shot_num, num_classes=64, kernel=3, groups=1, **kwargs):
        super(DynamicWeightsModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_func = CrossEntropyLoss()
        self._init_network()
        self.way_num = way_num
        self.query_num = query_num
        self.shot_num = shot_num
        self.test_way = test_way
        self.test_shot = test_shot
        self.test_query = test_query
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.global_clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
        # self.emb_func = emb_func

        self.dw_gen = DynamicWeights_(self.nFeat, 1, kernel, groups)
        self.dw = self.dw = ODEBlock(DynamicWeights(self.nFeat, 1, kernel, groups, self.nFeat))

    def reshape(self, ftrain, ftest):
        b, n1, c, h, w = ftrain.shape
        n2 = ftest.shape[1]
        ftrain = ftrain.unsqueeze(2).repeat(1, 1, n2, 1, 1, 1)
        ftest = ftest.unsqueeze(1).repeat(1, n1, 1, 1, 1, 1)
        return ftrain, ftest

    def process_feature(self, f, ytrain, num_train, num_test, batch_size):
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest = self.reshape(ftrain, ftest)

        # b, n2, n1, c, h, w
        ftrain = ftrain.transpose(1, 2)
        ftest = ftest.transpose(1, 2)
        return ftrain, ftest

    def get_score(self, ftrain, ftest, num_train, num_test, batch_size):
        b, n2, n1, c, h, w = ftrain.shape

        ftrain_ = ftrain.clone()
        ftest_ = ftest.clone()
        ftrain_ = ftrain_.reshape(-1, *ftrain.size()[3:])
        ftest_ = ftest_.reshape(-1, *ftest.size()[3:])

        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.reshape(-1, *ftrain_norm.size()[3:])
        conv_weight = ftrain_norm.mean(-1, keepdim=True).mean(-2, keepdim=True)

        filter_weight = self.dw_gen(ftrain_, ftest_)
        cls_scores = self.dw(x=(ftest_, filter_weight))
        cls_scores = cls_scores.view(b * n2, n1, *cls_scores.size()[1:])
        cls_scores = cls_scores.view(1, -1, *cls_scores.size()[3:])
        cls_scores = F.conv2d(cls_scores, conv_weight, groups=b * n1 * n2, padding=1)
        cls_scores = cls_scores.view(b * n2, n1, *cls_scores.size()[2:])
        return cls_scores

    def get_global_pred(self, ftest, ytest, num_test, batch_size, K):
        h = ftest.shape[-1]
        ftest_ = ftest.view(batch_size, num_test, K, -1)
        ftest_ = ftest_.transpose(2, 3)
        ytest_ = ytest.unsqueeze(3)
        ftest_ = torch.matmul(ftest_, ytest_)
        ftest_ = ftest_.view(batch_size * num_test, -1, h, h)
        global_pred = self.global_clasifier(ftest_)
        return global_pred

    def get_test_score(self, score_list):
        return score_list.mean(-1).mean(-1)

    def forward1(self, xtrain, xtest, ytrain, ytest, global_labels=None):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        ftrain, ftest = self.process_feature(f, ytrain, num_train,
                                                num_test, batch_size)
        cls_scores = self.get_score(ftrain, ftest,
                                             num_train, num_test, batch_size)

        if not self.training:
            return self.get_test_score(cls_scores)

        global_pred = self.get_global_pred(ftest, ytest, num_test, batch_size, K)
        return global_pred, cls_scores

    def set_forward(self, batch):
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))

        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(images, mode=2)

        support_targets_one_hot = one_hot(
            support_targets.reshape(episode_size * self.way_num * self.shot_num),
            self.way_num,
        )
        support_targets_one_hot = support_targets_one_hot.reshape(
            episode_size, self.way_num * self.shot_num, self.way_num
        )
        query_targets_one_hot = one_hot(
            query_targets.reshape(episode_size * self.way_num * self.query_num),
            self.way_num,
        )
        query_targets_one_hot = query_targets_one_hot.reshape(
            episode_size, self.way_num * self.query_num, self.way_num
        )

        cls_scores = self.forward1(
            support_feat.to(self.device), query_feat.to(self.device),
            support_targets_one_hot, query_targets_one_hot
        )

        cls_scores = cls_scores.reshape(episode_size * self.way_num * self.query_num, -1)
        acc = accuracy(cls_scores, query_targets.reshape(-1).to(self.device), topk=1)
        return cls_scores, acc

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)
        # print(images.shape)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        # emb = self.emb_func(images)  # [80, 640]
        # print(emb.shape)
        (
            support_feat,
            query_feat,
            support_targets,
            query_targets,
        ) = self.split_by_episode(
            images, mode=2
        )  # [4,5,512,6,6] [4,
        # 75, 512,6,6] [4, 5] [300]
        # print(support_feat.shape, query_feat.shape, support_targets.shape, query_targets.shape)
        support_targets = support_targets.reshape(
            episode_size, self.way_num * self.shot_num
        ).contiguous()
        support_global_targets, query_global_targets = (
            global_targets[:, :, : self.shot_num],
            global_targets[:, :, self.query_num: ],
        )

        # support_feat, support_targets, support_global_targets = shuffle(
        #     support_feat,
        #     support_targets,
        #     support_global_targets.reshape(*support_targets.size()),
        # )
        # query_feat, query_targets, query_global_targets = shuffle(
        #     query_feat,
        #     query_targets.reshape(*query_feat.size()[:2]),
        #     query_global_targets.reshape(*query_feat.size()[:2]),
        # )

        # convert to one-hot
        support_targets_one_hot = one_hot(
            support_targets.reshape(episode_size * self.way_num * self.shot_num),
            self.way_num,
        )
        support_targets_one_hot = support_targets_one_hot.reshape(
            episode_size, self.way_num * self.shot_num, self.way_num
        )
        query_targets_one_hot = one_hot(
            query_targets.reshape(episode_size * self.way_num * self.query_num),
            self.way_num,
        )
        query_targets_one_hot = query_targets_one_hot.reshape(
            episode_size, self.way_num * self.query_num, self.way_num
        )
        # print(support_feat.shape, query_feat.shape, support_targets_one_hot.shape, query_targets_one_hot.shape)
        # [75, 64, 6, 6], [75, 5, 6, 6]
        output, cls_scores = self.forward1(
            support_feat, query_feat, support_targets_one_hot, query_targets_one_hot
        )
        loss1 = self.loss_func(output, query_global_targets.contiguous().reshape(-1))
        loss2 = self.loss_func(cls_scores, query_targets.reshape(-1))
        loss = loss1 + 0.5 * loss2
        cls_scores = torch.sum(
            cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1
        )  # [300, 5]
        acc = accuracy(cls_scores, query_targets.reshape(-1), topk=1)
        return output, acc, loss

# https://libfewshot-en.readthedocs.io/zh-cn/latest/tutorials/t5-add_a_new_classifier.html
# 完成class DynamicWeightsModel(MetaModel):
# 修改forward函数
# 完成set_forward函数
# 完成set_forward_loss函数
# 完成set_forward_adaptaion函数
# class DynamicWeightsModel(MetaModel):
#     def __init__(self, way_num, query_num, shot_num, **kwargs):
#         super(DynamicWeightsModel, self).__init__(**kwargs)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.classifier = Model()
#         self.loss_func = CrossEntropyLoss()
#         self._init_network()
#         self.way_num = way_num
#         self.query_num = query_num
#         self.shot_num = shot_num
#         # self.episode_size = 4

# #     def set_forward(
# #         self,
# #         batch,
# #     ):
# #         """
# #         :param batch:
# #         :return:
# #         """
# #         # episode_size = self.episode_size
# #         images, global_targets = batch
# #         images = images.to(self.device)
# #         global_targets = global_targets.to(self.device)
# #         episode_size = images.size(0) // (
# #             self.way_num * (self.shot_num + self.query_num)
# #         )
# #         # print(images.shape)
# #         # emb = self.emb_func(images)
# #         # print(emb.shape)
# #         (
# #             support_feat,
# #             query_feat,
# #             support_targets,
# #             query_targets,
# #         ) = self.split_by_episode(
# #             images, mode=2
# #         )  # [4,5,512,6,6] [4,
# #         # 75, 512,6,6] [4, 5] [300]

# #         # convert to one-hot
# #         support_targets_one_hot = one_hot(
# #             support_targets.reshape(episode_size * self.way_num * self.shot_num)
# #         )
# #         support_targets_one_hot = support_targets_one_hot.reshape(
# #             episode_size, self.way_num * self.shot_num, self.way_num
# #         )
# #         query_targets_one_hot = one_hot(
# #             query_targets.reshape(episode_size * self.way_num * self.query_num)
# #         )
# #         query_targets_one_hot = query_targets_one_hot.reshape(
# #             episode_size, self.way_num * self.query_num, self.way_num
# #         )
# #         _ , cls_scores = self.classifier.forward(
# #             support_feat, query_feat, support_targets_one_hot, query_targets_one_hot
# #         )
# #         # cls_scores = self.cam_layer.val_transductive(
# #         #        support_feat, query_feat, support_targets_one_hot, query_targets_one_hot
# #         # )

# #         cls_scores = cls_scores.reshape(
# #             episode_size * self.way_num * self.query_num, -1
# #         )
# #         acc = accuracy(cls_scores, query_targets.reshape(-1), topk=1)
# #         return cls_scores, acc

# #     def set_forward_loss(self, batch):
# #         """
# #         :param batch:
# #         :return:
# #         """
# #         # episode_size = self.episode_size
# #         images, global_targets = batch
# #         images = images.to(self.device)
# #         global_targets = global_targets.to(self.device)
# #         episode_size = images.size(0) // (
# #             self.way_num * (self.shot_num + self.query_num)
# #         )
# #         # print(images.shape)
# #         # emb = self.emb_func(images)
# #         # print(emb.shape)
# #         (
# #             support_feat,
# #             query_feat,
# #             support_targets,
# #             query_targets,
# #         ) = self.split_by_episode(
# #             images, mode=2
# #         )  # [4,5,512,6,6] [4,
# #         # 75, 512,6,6] [4, 5] [300]
# #         # print(support_feat.shape, query_feat.shape, support_targets.shape, query_targets.shape)
# #         support_targets = support_targets.reshape(
# #             episode_size, self.way_num * self.shot_num
# #         ).contiguous()
# #         support_global_targets, query_global_targets = (
# #             global_targets[:, :, : self.shot_num],
# #             global_targets[:, :, self.shot_num :],
# #         )

# #         # support_feat, support_targets, support_global_targets = shuffle(
# #         #     support_feat,
# #         #     support_targets,
# #         #     support_global_targets.reshape(*support_targets.size()),
# #         # )
# #         # query_feat, query_targets, query_global_targets = shuffle(
# #         #     query_feat,
# #         #     query_targets.reshape(*query_feat.size()[:2]),
# #         #     query_global_targets.reshape(*query_feat.size()[:2]),
# #         # )

# #         # convert to one-hot
# #         support_targets_one_hot = one_hot(
# #             support_targets.reshape(episode_size * self.way_num * self.shot_num)
# #         )
# #         support_targets_one_hot = support_targets_one_hot.reshape(
# #             episode_size, self.way_num * self.shot_num, self.way_num
# #         )
# #         query_targets_one_hot = one_hot(
# #             query_targets.reshape(episode_size * self.way_num * self.query_num)
# #         )
# #         query_targets_one_hot = query_targets_one_hot.reshape(
# #             episode_size, self.way_num * self.query_num, self.way_num
# #         )
# #         # print(support_feat.shape, query_feat.shape, support_targets_one_hot.shape, query_targets_one_hot.shape)
# #         # [75, 64, 6, 6], [75, 5, 6, 6]
# #         output, cls_scores = self.classifier.forward(
# #             support_feat, query_feat, support_targets_one_hot, query_targets_one_hot
# #         )
# #         loss1 = self.loss_func(output, query_global_targets.contiguous().reshape(-1))
# #         loss2 = self.loss_func(cls_scores, query_targets.reshape(-1))
# #         loss = loss1 + 0.5 * loss2
# #         cls_scores = torch.sum(
# #             cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1
# #         )  # [300, 5]
# #         acc = accuracy(cls_scores, query_targets.reshape(-1), topk=1)
# #         return output, acc, loss


#     def set_forward(self, batch):
#         images, global_targets = batch
#         images = images.to(self.device)
#         global_targets = global_targets.to(self.device)
#         episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))

#         support_feat, query_feat, support_targets, query_targets = self.split_by_episode(images, mode=2)

#         support_targets_one_hot = one_hot(
#             support_targets.reshape(episode_size * self.way_num * self.shot_num)
#         ).to(self.device)
#         support_targets_one_hot = support_targets_one_hot.reshape(
#             episode_size, self.way_num * self.shot_num, self.way_num
#         )
#         query_targets_one_hot = one_hot(
#             query_targets.reshape(episode_size * self.way_num * self.query_num)
#         ).to(self.device)
#         query_targets_one_hot = query_targets_one_hot.reshape(
#             episode_size, self.way_num * self.query_num, self.way_num
#         )

#         cls_scores = self.classifier.forward(
#             support_feat.to(self.device), query_feat.to(self.device),
#             support_targets_one_hot, query_targets_one_hot
#         )

#         cls_scores = cls_scores.reshape(episode_size * self.way_num * self.query_num, -1)
#         acc = accuracy(cls_scores, query_targets.reshape(-1).to(self.device), topk=1)
#         return cls_scores, acc

#     # def set_forward_loss(self, batch):
        
#     #     image, global_target = batch  # unused global_target
#     #     image = image.to(self.device)
#     #     (
#     #         support_image,
#     #         query_image,
#     #         support_target,
#     #         query_target,
#     #     ) = self.split_by_episode(image, mode=2)
#     #     xtrain = support_image
#     #     ytrain = one_hot(support_target).cuda()
#     #     xtest = query_image
#     #     ytest = one_hot(query_target).cuda()
#     #     pids = global_target[:, :, self.shot_num:].reshape(-1)
#     #     # print("all sizes:", xtrain.size(), ytrain.size(), xtest.size(), ytest.size(), pids.size())
#     #     print(pids.shape)
#     #     batch_size, num_train = xtrain.size(0), xtrain.size(1)
#     #     num_test = xtest.size(1)
#     #     K = ytrain.size(2)
#     #     ytrain = ytrain.transpose(1, 2)

#     #     xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
#     #     xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

#     #     x = torch.cat((xtrain, xtest), 0)
#     #     f = self.classifier.base(x)
#     #     ftrain, ftest = self.classifier.process_feature(f, ytrain, num_train,
#     #                                         num_test, batch_size)
#     #     cls_scores = self.classifier.get_score(ftrain, ftest,
#     #                                 num_train, num_test, batch_size)

#     #     if not self.classifier.training:
#     #         return self.classifier.get_test_score(cls_scores)

#     #     global_pred = self.classifier.get_global_pred(ftest, ytest, num_test, batch_size, K)
#     #     criterion = CrossEntropyLoss()
#     #     loss1 = criterion(global_pred, pids.view(-1))
#     #     loss2 = criterion(cls_scores, query_target.view(-1))
#     #     loss = loss1 + 0.5 * loss2
#     #     loss.backward()
#     #     # cls_scores = cls_scores.view(batch_size * num_test, -1)
#     #     # labels_test = query_target.view(batch_size * num_test)

#     #     # _, preds = torch.max(cls_scores.detach().cpu(), 1)
#     #     cls_scores = torch.sum(
#     #         cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1
#     #     )
#     #     acc = accuracy(cls_scores, query_target.reshape(-1).to(self.device), topk=1)

#     #     return global_pred, acc, loss
#     # def set_forward_loss(self, batch):
#     #     image, global_target = batch  # unused global_target
#     #     image = image.to(self.device)
#     #     (
#     #         support_image,
#     #         query_image,
#     #         support_target,
#     #         query_target,
#     #     ) = self.split_by_episode(image, mode=2)
#     #     xtrain = support_image
#     #     ytrain = one_hot(support_target).cuda()
#     #     xtest = query_image
#     #     ytest = one_hot(query_target).cuda()
#     #     pids = global_target.view(-1)
#     #     pids = pids[-self.query_num*self.way_num:]
#     #     print(pids.shape)
#     #     # print("all sizes:", xtrain.size(), ytrain.size(), xtest.size(), ytest.size(), pids.size())

#     #     batch_size, num_train = xtrain.size(0), xtrain.size(1)
#     #     num_test = xtest.size(1)
#     #     K = ytrain.size(2)
#     #     ytrain = ytrain.transpose(1, 2)

#     #     xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
#     #     xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

#     #     x = torch.cat((xtrain, xtest), 0)
#     #     f = self.classifier.base(x)
#     #     ftrain, ftest = self.classifier.process_feature(f, ytrain, num_train,
#     #                                          num_test, batch_size)
#     #     cls_scores = self.classifier.get_score(ftrain, ftest,
#     #                                 num_train, num_test, batch_size)

#     #     if not self.classifier.training:
#     #         return self.classifier.get_test_score(cls_scores)

#     #     global_pred = self.classifier.get_global_pred(ftest, ytest, num_test, batch_size, K)
#     #     criterion = CrossEntropyLoss()
#     #     loss1 = criterion(global_pred, pids.view(-1))
#     #     loss2 = criterion(cls_scores, query_target.view(-1))
#     #     loss = loss1 + 0.5 * loss2
#     #     # loss = criterion(cls_scores, query_target.view(-1))
        
#     #     # cls_scores = cls_scores.view(batch_size * num_test, -1)
#     #     # labels_test = query_target.view(batch_size * num_test)
#     #     cls_scores = torch.sum(
#     #         cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1
#     #     )
#     #     # accs = AverageMeter()
#     #     # _, preds = torch.max(cls_scores.detach().cpu(), 1)
#     #     # acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
#     #     # accs.update(acc.item(), labels_test.size(0))

#     #     # gt = (preds == labels_test.detach().cpu()).float()
#     #     # gt = gt.view(batch_size, num_test).numpy() #[b, n]
#     #     # acc = np.sum(gt, 1) / num_test
#     #     # acc = np.reshape(acc, (batch_size))
#     #     # accuracy = accs.avg
#     #     acc = accuracy(cls_scores, query_target.reshape(-1), topk=1)
#     #     return global_pred, accuracy, loss
#     def set_forward_loss(self, batch):
#         images, global_targets = batch
#         images = images.to(self.device)
#         global_targets = global_targets.to(self.device)
#         episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
#         # print(global_targets.shape)
#         support_feat, query_feat, support_targets, query_targets = self.split_by_episode(images, mode=2)
#         pids = global_targets.view(-1)
#         # print(pids.shape)
#         # support_targets = support_targets.reshape(
#         #     episode_size, self.way_num * self.shot_num
#         # ).contiguous().to(self.device)
#         support_global_targets, query_global_targets = (
#             global_targets[:, :, :self.shot_num].to(self.device),
#             global_targets[:, :, self.query_num:].to(self.device),
#         )
#         # print(query_global_targets.shape)
#         # aa = query_global_targets.contiguous().reshape(-1)
#         # print(aa.shape)
#         # support_feat, support_targets, support_global_targets = shuffle(
#         #     support_feat,
#         #     support_targets,
#         #     support_global_targets.reshape(*support_targets.size()),
#         # )
#         # query_feat, query_targets, query_global_targets = shuffle(
#         #     query_feat,
#         #     query_targets.reshape(*query_feat.size()[:2]),
#         #     query_global_targets.reshape(*query_feat.size()[:2]),
#         # )
#         support_targets_one_hot = one_hot(
#             support_targets
#         ).to(self.device)
#         # support_targets_one_hot = support_targets_one_hot.reshape(
#         #     episode_size, self.way_num * self.shot_num, self.way_num
#         # )
#         query_targets_one_hot = one_hot(
#             query_targets
#         ).to(self.device)
#         # query_targets_one_hot = query_targets_one_hot.reshape(
#         #     episode_size, self.way_num * self.query_num, self.way_num
#         # )

#         output, cls_scores = self.classifier.forward(
#             support_feat.to(self.device), query_feat.to(self.device),
#             support_targets_one_hot, query_targets_one_hot
#         )
#         loss1 = self.loss_func(output, query_global_targets.reshape(-1))
#         loss2 = self.loss_func(cls_scores, query_targets.reshape(-1).to(self.device))
#         loss = loss1 + 0.5 * loss2
#         cls_scores = torch.sum(
#             cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1
#         )
#         acc = accuracy(cls_scores, query_targets.reshape(-1).to(self.device), topk=1)
#         return output, acc, loss
#     # def set_forward_loss(self, batch):
#     #     image, global_target = batch  # unused global_target
#     #     image = image.to(self.device)
#     #     (
#     #         support_image,
#     #         query_image,
#     #         support_target,
#     #         query_target,
#     #     ) = self.split_by_episode(image, mode=2)
#     #     xtrain = support_image
#     #     ytrain = one_hot(support_target).cuda()
#     #     xtest = query_image
#     #     ytest = one_hot(query_target).cuda()
#     #     pids = global_target.view(-1)
#     #     # print("all sizes:", xtrain.size(), ytrain.size(), xtest.size(), ytest.size(), pids.size())

#     #     batch_size, num_train = xtrain.size(0), xtrain.size(1)
#     #     num_test = xtest.size(1)
#     #     K = ytrain.size(2)
#     #     ytrain = ytrain.transpose(1, 2)

#     #     xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
#     #     xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

#     #     x = torch.cat((xtrain, xtest), 0)
#     #     f = self.classifier.base(x)
#     #     ftrain, ftest = self.classifier.process_feature(f, ytrain, num_train,
#     #                                          num_test, batch_size)
#     #     cls_scores = self.classifier.get_score(ftrain, ftest,
#     #                                 num_train, num_test, batch_size)

#     #     if not self.classifier.training:
#     #         return self.classifier.get_test_score(cls_scores)

#     #     global_pred = self.classifier.get_global_pred(ftest, ytest, num_test, batch_size, K)
#     #     criterion = CrossEntropyLoss()
#     #     loss1 = criterion(global_pred, pids.view(-1))
#     #     loss2 = criterion(cls_scores, query_target.view(-1))
#     #     loss = loss1 + 0.5 * loss2
#     #     # loss = criterion(cls_scores, query_target.view(-1))
#     #     cls_scores = cls_scores.view(batch_size * num_test, -1)
#     #     labels_test = query_target.view(batch_size * num_test)

#     #     accs = AverageMeter()
#     #     _, preds = torch.max(cls_scores.detach().cpu(), 1)
#     #     acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
#     #     accs.update(acc.item(), labels_test.size(0))

#     #     gt = (preds == labels_test.detach().cpu()).float()
#     #     gt = gt.view(batch_size, num_test).numpy() #[b, n]
#     #     acc = np.sum(gt, 1) / num_test
#     #     acc = np.reshape(acc, (batch_size))
#     #     accuracy = accs.avg
#     #     # acc = accuracy(cls_scores, query_target.reshape(-1), topk=1)
#     #     return global_pred, accuracy, loss

# # class DynamicWeightsModel(MetaModel):
# #     def __init__(self, way_num, query_num, shot_num, batch_size=4, **kwargs):
# #         super(DynamicWeightsModel, self).__init__(**kwargs)
# #         self.classifier = Model()
# #         self.model = Model()
# #         self.loss_func = CrossEntropyLoss()
# #         self.way_num = way_num
# #         self.query_num = query_num
# #         self.shot_num = shot_num
# #         self.batch_size = batch_size
# #
# #
# #     # def set_forward(self, batch):
# #     #     # accs = AverageMeter()
# #     #     image, global_target = batch
# #     #     print(global_target.shape)
# #     #     # print(image.shape)
# #     #     # print(global_target.shape)
# #     #     image = image.to(self.device)
# #     #     global_target = global_target.to(self.device)
# #     #
# #     #     # feat = self.model.base(image)
# #     #     support_feat, query_feat, support_target, query_target = self.split_by_episode(image, mode=2)
# #     #     # print(support_feat.shape)
# #     #     # print(query_feat.shape)
# #     #     # print(support_target.shape)
# #     #     # print(query_target.shape)
# #     #     support_target = one_hot(support_target).cuda()
# #     #     query_target_oh = one_hot(query_target).cuda()
# #     #
# #     #
# #     #     num_test_examples = query_feat.size(1)
# #     #     classifier = self.model
# #     #     output, cls_scores = classifier(support_feat, query_feat, support_target, query_target_oh)
# #     #     cls_scores = cls_scores.reshape(self.batch_size * num_test_examples, -1)
# #     #     query_target = query_target.reshape(self.batch_size * num_test_examples)
# #     #
# #     #     _, preds = torch.max(cls_scores.detach().cpu(), 1)
# #     #     # acc = (torch.sum(preds == query_target.detach().cpu()).float()) / query_target.size(0)
# #     #     # accs.update(acc.item(), query_target.size(0))
# #     #     # gt = (preds == query_target.detach().cpu()).float()
# #     #     # gt = gt.view(self.batch_size, num_test_examples).numpy() #[b, n]
# #     #     # acc = np.sum(gt, 1) / num_test_examples
# #     #     # acc = np.reshape(acc, (self.batch_size))
# #     #     # accuracy = accs.avg
# #     #     # # print(support_feat.shape)
# #     #     # # print(query_feat.shape)
# #     #     # # print(support_target.shape)
# #     #     # # print(query_target.shape)
# #     #     # classifier = self.model
# #     #     # output, cls_scores = classifier(support_feat, query_feat, support_target, query_target_oh)
# #     #     acc = accuracy(cls_scores, query_target)
# #     #     # acc = accuracy(cls_scores, query_target.contiguous().reshape(-1))
# #     #     return output, acc
# #     #
# #     # def set_forward_loss(self, batch):
# #     #     accs = AverageMeter()
# #     #     image, global_target = batch
# #     #     print(global_target.shape)
# #     #     # print(image.shape)
# #     #     # print(global_target.shape)
# #     #     image = image.to(self.device)
# #     #     global_target = global_target.to(self.device)
# #     #
# #     #     # feat = self.model.base(image)
# #     #     support_feat, query_feat, support_target, query_target = self.split_by_episode(image, mode=2)
# #     #     # print(support_feat.shape)
# #     #     # print(query_feat.shape)
# #     #     # print(support_target.shape)
# #     #     # print(query_target.shape)
# #     #     support_target = one_hot(support_target).cuda()
# #     #     query_target_oh = one_hot(query_target).cuda()
# #     #
# #     #
# #     #     num_test_examples = query_feat.size(1)
# #     #     classifier = self.model
# #     #     output, cls_scores = classifier(support_feat, query_feat, support_target, query_target_oh)
# #     #     cls_scores = cls_scores.reshape(self.batch_size * num_test_examples, -1)
# #     #     query_target = query_target.reshape(self.batch_size * num_test_examples)
# #     #
# #     #     _, preds = torch.max(cls_scores.detach().cpu(), 1)
# #     #
# #     #     loss = self.loss_func(cls_scores, query_target.contiguous().reshape(-1))
# #     #
# #     #     # acc = (torch.sum(preds == query_target.detach().cpu()).float()) / query_target.size(0)
# #     #     # accs.update(acc.item(), query_target.size(0))
# #     #     # gt = (preds == query_target.detach().cpu()).float()
# #     #     # gt = gt.view(self.batch_size, num_test_examples).numpy() #[b, n]
# #     #     # acc = np.sum(gt, 1) / num_test_examples
# #     #     # acc = np.reshape(acc, (self.batch_size))
# #     #     # accuracy = accs.avg  # loss1 = self.loss_func(output, query_target.contiguous().reshape(-1))
# #     #
# #     #     # loss = loss1 + 0.5 * loss2
# #     #     # loss = self.loss_func(output, query_target.contiguous().view(-1))
# #     #     # loss = self.loss_func(output, query_target.contiguous().view(-1))
# #     #     acc = accuracy(cls_scores, query_target)
# #     #     # acc = accuracy(cls_scores, query_target.contiguous().reshape(-1))
# #     #     return output, acc, loss
# #     #
# #     # def set_forward_adaptation(self, support_set, support_target):
# #     #     extractor_lr = self.inner_param["extractor_lr"]
# #     #     classifier_lr = self.inner_param["classifier_lr"]
# #     #     fast_parameters = list(item[1] for item in self.named_parameters())
# #     #     for parameter in self.parameters():
# #     #         parameter.fast = None
# #     #     self.model.train()
# #     #     self.model.classifier.train()
# #     #     features, output = self.forward_output(support_set)
# #     #     loss = self.loss_func(output, support_target)
# #     #     grad = torch.autograd.grad(
# #     #         loss, fast_parameters, create_graph=True, allow_unused=True
# #     #     )
# #     #     fast_parameters = []
# #     #
# #     #     for k, weight in enumerate(self.named_parameters()):
# #     #         if grad[k] is None:
# #     #             continue
# #     #         fast_parameters.append((weight[0], weight[1] - classifier_lr * grad[k]))
# #     #
# #     #     for weight in self.model.parameters():
# #     #         weight.fast = None
# #     #
# #     #     return fast_parameters
# #     def forward_output(self, x):
# #         out1 = self.emb_func(x)
# #         out2 = self.classifier(out1)
# #         return out2
# #
# #     def set_forward(self, batch):
# #         image, global_target = batch  # unused global_target
# #         image = image.to(self.device)
# #         (
# #             support_image,
# #             query_image,
# #             support_target,
# #             query_target,
# #         ) = self.split_by_episode(image, mode=2)
# #         episode_size, _, c, h, w = support_image.size()
# #         output_list = []
# #         for i in range(episode_size):
# #             episode_support_image = support_image
# #             episode_query_image = query_image
# #             episode_support_target = one_hot(support_target).cuda()
# #             episode_query_target = one_hot(query_target).cuda()
# #             episode_support_tar = support_target[i].reshape(-1)
# #             self.set_forward_adaptation(episode_support_image, episode_query_image, episode_support_target, episode_query_target, episode_support_tar)
# #
# #             output = self.forward_output(episode_support_image, episode_query_image, episode_support_target, episode_query_target)
# #
# #             output_list.append(output)
# #
# #         output = torch.cat(output_list, dim=0)
# #         # loss = self.loss_func(output, query_target.contiguous().view(-1))
# #         acc = accuracy(output, query_target.contiguous().view(-1))
# #         return output, acc
# #
# #     def set_forward_loss(self, batch):
# #         image, global_target = batch  # unused global_target
# #         image = image.to(self.device)
# #         (
# #             support_image,
# #             query_image,
# #             support_target,
# #             query_target,
# #         ) = self.split_by_episode(image, mode=2)
# #         episode_size, _, c, h, w = support_image.size()
# #         output_list = []
# #         for i in range(episode_size):
# #             episode_support_image = support_image
# #             episode_query_image = query_image
# #             episode_support_target = one_hot(support_target).cuda()
# #             episode_query_target = one_hot(query_target).cuda()
# #             episode_support_tar = support_target[i].reshape(-1)
# #             self.set_forward_adaptation(episode_support_image, episode_query_image, episode_support_target, episode_query_target, episode_support_tar)
# #
# #             output = self.forward_output(episode_support_image, episode_query_image, episode_support_target, episode_query_target)
# #
# #             output_list.append(output)
# #
# #         output = torch.cat(output_list, dim=0)
# #         loss = self.loss_func(output, query_target.contiguous().view(-1))
# #         acc = accuracy(output, query_target.contiguous().view(-1))
# #         return output, acc, loss
# #
# #     def set_forward_adaptation(self, xtrain, xtest, ytrain, ytest,ytr):
# #         lr = self.inner_param["lr"]
# #         fast_parameters = list(self.classifier.parameters())
# #         for parameter in self.classifier.parameters():
# #             parameter.fast = None
# #
# #         self.emb_func.train()
# #         self.classifier.train()
# #         for i in range(
# #             self.inner_param["train_iter"]
# #             if self.training
# #             else self.inner_param["test_iter"]
# #         ):
# #             output , _ = self.classifier(xtrain, xtest, ytrain, ytest)
# #             loss = nn.CrossEntropyLoss(output, ytr)
# #             grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
# #             fast_parameters = []
# #
# #             for k, weight in enumerate(self.parameters()):
# #                 if weight.fast is None:
# #                     weight.fast = weight - lr * grad[k]
# #                 else:
# #                     weight.fast = weight.fast - lr * grad[k]
# #                 fast_parameters.append(weight.fast)
