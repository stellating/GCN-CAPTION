import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        # print('[models/reset_parameters support,adj]:',support.shape,adj.shape)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, vocab_size, embed_size, in_channel=256):
        super(GCNResnet, self).__init__()

        self.attrs = torch.LongTensor([225, 74, 778, 167, 97, 789, 19, 23, 46, 285, 602, 797, 798, 805, 806, 96, 88, 194, 807, 808, 809, 815, 819, 822, 826, 828, 282])
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, feature, adj):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        self.A = Parameter(adj.float())
        adj = gen_adj(self.A).detach().cuda()
        attributes = self.embedding(self.attrs.cuda()).cuda()
        x = self.gc1(attributes, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(1, 2)
        batch_size = feature.size(0)
        res = torch.zeros((batch_size,x.size(2)))
        for i in range(batch_size):
            f=feature[i]
            x1 = x[i]
            x2 = torch.matmul(f, x1)
            res[i]=x2
        return res

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


class DecoderRNN(nn.Module):
    def __init__(self, encoder, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.encoder = encoder


        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    # outputs = decoder(features, captions, lengths)
    def forward(self, features, adj, captions, lengths):
        """Decode image feature vectors and generates captions."""
        output = self.encoder(features, adj)
        output = self.embed(output.long().cuda())
        embeddings = self.embed(captions.long())
        # print(features.shape)   # torch.Size([512, 256])
        # print(features.unsqueeze(1).shape)  # torch.Size([512, 1, 256])
        # print(torch.cat((features.unsqueeze(1), embeddings), 1).shape) # torch.Size([512, 14, 256])
        # print("lengths shape is {}".format(np.array(lengths).shape))   # lengths shape is (512,)
        embeddings = torch.cat((output, embeddings), 1)
        # lengths = [length+1 for length in lengths]
        packed = pack_padded_sequence(embeddings, lengths,
                                      batch_first=True)  # <class 'torch.nn.utils.rnn.PackedSequence'>
        # print(np.sum(lengths))   # 4200   整个batch中所有句子中真实单词的个数
        # print(np.max(lengths))   # 16
        # print("packed[0] shape is {}".format(packed[0].shape))   # torch.Size([4200, 256]) 想法是正确的
        # print("packed[1] shape is {}".format(packed[1].shape))   # packed[1] shape is torch.Size([16])
        # print(type(packed))
        # print("packed shape is {}".format(packed.size()))
        hiddens, _ = self.lstm(packed)
        # print(type(hiddens))     # # <class 'torch.nn.utils.rnn.PackedSequence'>
        # print("hiddens[0] shape is {}".format(hiddens[0].shape)) # hiddens[0] shape is torch.Size([3668, 512])
        # print("hiddens[1] shape is {}".format(hiddens[1].shape)) # hiddens[1] shape is torch.Size([13])
        outputs = self.linear(hiddens[0])
        # print("outputs shape is {}".format(outputs.shape))       # outputs shape is torch.Size([3668, 9214])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []  # 用于存储多个torch.size(128)的predicted，每一个predicted对应一个这个batch的预测句子的每个word，多个predicted对应依次的多个word
        inputs = features.unsqueeze(1)  # torch.Size([512, 1, 256])  [batch_size, time_steps, embeddingVector_size]
        """
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        """
        for i in range(self.max_seg_length):  # max_seg_length = 20
            # 核心是输入的inputs的time_step是1
            hiddens, states = self.lstm(inputs,
                                        states)  # hiddens: (batch_size, time_steps, hidden_size)   torch.Size([512, 1, 512])代表所有time_step的隐层输出
            # len of states is 2
            # states[0]=hn shape is torch.Size([1, 512, 512])    第一个512代表batch_size,代表最后一个time_step的hn
            # states[1]=Cn shape is torch.Size([1, 512, 512])    第一个512代表batch_size,代表最后一个time_step的cn
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size) is the position corresponding the max
            sampled_ids.append(predicted)  # the unit of predicted arrays is the label of the vocab_size
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        # torch.size(128,20)  每个句子有20个word，每个word是digit，对应vocab里面word的position
        return sampled_ids

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.encoder.parameters(), 'lr': lr * lrp},
                {'params':self.embed.parameters(),'lr':lr},
                {'params': self.lstm.parameters(), 'lr': lr},
                {'params': self.linear.parameters(), 'lr': lr},
                ]


def gcn_resnet101(num_classes, embed_size, hidden_size, vocab, num_layers, pretrained=False, in_channel=256):
    model = models.resnet101(pretrained=pretrained)
    encoder = GCNResnet(model, num_classes, vocab_size=len(vocab), embed_size=embed_size, in_channel=in_channel)
    decoder = DecoderRNN(encoder, embed_size, hidden_size, len(vocab), num_layers).cuda()
    return decoder
