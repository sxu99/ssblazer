import torch.nn as nn
import torch
import numpy as np
from torch.nn import init

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(
            0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(
            0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(
            0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(
            b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# For residual block
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out


class res(nn.Module):

    def __init__(self):

        super().__init__()

        # Inception
        self.conv2d_1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 1), stride=(1, 1)),
                                      nn.ReLU())
        self.conv2d_2 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=(1, 1), padding='same'),
                                      nn.ReLU())
        self.conv2d_3 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 5), stride=(1, 1), padding='same'),
                                      nn.ReLU())
        self.bidirectional = nn.LSTM(
            input_size=28, hidden_size=12, num_layers=1, bidirectional=True, batch_first=True)

        self.self_att = ScaledDotProductAttention(
            d_model=28, d_k=512, d_v=512, h=8)

        # Preprocess before resnet
        self.preprocess = nn.Sequential(
            conv3x3(4, 8),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=False)
        )

        # Res block
        self.resnet = nn.Sequential(self.make_layer(ResidualBlock, in_channels=8, out_channels=16, blocks=15),
                                 self.make_layer(ResidualBlock, in_channels=16, out_channels=32, blocks=15))

        self.pooling2d =nn.AvgPool2d(kernel_size=(5,1),stride=3)

        self.self_att_seq = ScaledDotProductAttention(
            d_model=32, d_k=512, d_v=512, h=8)

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(nn.Linear(1820, 128),
                                   nn.Dropout(0.5),
                                   nn.ReLU(),
                                   nn.Linear(128, 20),
                                   nn.Dropout(0.5),
                                   nn.ReLU())

        self.main_output = nn.Sequential(nn.Linear(20, 1), nn.Sigmoid())

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):

        cutted = input[:, 71:, :]
        cutted = cutted[:, :-71, :]
        cutted = cutted.permute(0, 2, 1)
        cutted = cutted.unsqueeze(3)

        c1 = self.conv2d_1(cutted)
        c2 = self.conv2d_2(cutted)
        c3 = self.conv2d_3(cutted)
        inception_concat = torch.cat((cutted, c1, c2, c3), 1).squeeze()
        inception_concat = inception_concat.permute(0, 2, 1)

        center_atten = self.self_att(
            inception_concat, inception_concat, inception_concat)

        center_flatten = self.flatten(center_atten)

        input = input.permute(0, 2, 1)
        input = input.unsqueeze(3)

        preprocessed_seq = self.preprocess(input)

        emb_seq=self.resnet(preprocessed_seq)
        emb_seq=self.pooling2d(emb_seq).squeeze()
        emb_seq = emb_seq.permute(0, 2, 1)
        emb_seq=self.self_att_seq(emb_seq,emb_seq,emb_seq)
        emb_seq_flatten=self.flatten(emb_seq)


        concat = torch.cat((emb_seq_flatten, center_flatten), 1).squeeze()
        dense_out = self.dense(concat)
        out = self.main_output(dense_out)

        return out
