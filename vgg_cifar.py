'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

#torch.manual_seed(10)

# VGG configs
cfg = {
    'VGG06': ['B64', 'M', 'B128', 'M', 'B512', 'M'], #4 layers
    'VGG08': ['B64', 'M', 'B128', 'M', 'B256', 'M', 512, 'M', 512 , 'M'], #6 layers
    'VGG11': ['B64', 'M', 'B128', 'M', 'B256', 'B256', 'M', 'B512', 'B512', 'M', 'B512', 'B512', 'M'],  #9 layers
    'VGG13': ['B64', 64, 'M', 'B128', 128, 'M', 'B256', 256, 'M', 512, 512, 'M', 512, 512, 'M'], #11 layers
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 'D' ,'B512', 512, 512, 'M'], #14 layers
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], #17 layers
}



def load_pretrained_model(model, pretrained):
    '''
    Function to load pre-trained model for modified model definition
    to include NamedLayer support
    '''
    pt = list(pretrained.items())

    model_kvpair = model.state_dict()
    count=0
    for key, value in model_kvpair.items():
        flag = 0 # Flag to detect whether match found
        for key_q, value_q in pretrained.items():
            if key_q == key:#.replace('.named_layer.','.'):
                model_kvpair[key].data.copy_(value_q.data)
                count+=1
                flag = 1
        if flag == 0 and key[-7:] != 'tracked':
            print('Loading error:' + key + ' not found in pre-trained model!')

    if count != len(pt):
        print('Loading error: Pre-trained model has ' + str(len(pt)) + ' modules while only ' + str(count) + 'matches done!')

    del pretrained
    return 0


class NamedLayer(nn.Module):
    # A layer with a name
    def __init__(self, layer, number):
        super(NamedLayer, self).__init__()
        self.named_layer = layer
        self.number = number

    def forward(self, x):
        return self.named_layer(x)

# VGG Network class
class VGG(nn.Module):
    def __init__(self, vgg_name, drop_val=0.):
        super(VGG, self).__init__()
        self.features = self._make_modlist(cfg[vgg_name], drop_val)
        self.classifier = nn.Linear(512, 100)
        self.name = vgg_name
        self.drop_val = drop_val
        self._initialize_weights()

    def forward(self, x, slope = 0.):
        out, out_feat = self.organize_features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, out_feat

    def organize_features(self, x):
        layers = []
        in_channels = 3
        count = 0
        x_feat = None
        for i in cfg[self.name]:

            if (i == 'M') or (i == 'D'):
                x = self.features[count](x)

            else:
                x = self.features[count](x)
                count = count + 1
                x = self.features[count](x)
                if type(i) == str:
                    if i[0] == 'B':
                        if x_feat is None:
                            x_feat = {}
                        x_feat[self.features[count].number] = x
                x = F.relu(x)

            count = count + 1

        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        return x, x_feat


    def _make_modlist(self, cfg, drop_val):
        layers = []
        in_channels = 3
        index = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(p=drop_val)]
            elif type(x) == str:
                if x[0] == 'B':
                    x = int(x[1:])
                    layers += [NamedLayer(nn.Conv2d(in_channels, x, kernel_size = 3, padding=1), number = index),
                            NamedLayer(nn.BatchNorm2d(x), number = index)]
                    in_channels = x
                    index += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x)]
                in_channels = x
        return nn.ModuleList(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    #m.bias.data.zero_()
                    m.bias.data.normal_(0,0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.normal_(0,1.)


# Function to test network feedforward
def test():
    net = VGG('VGG11', drop_val = 0.0)
    x = Variable(torch.ones(1,3,32,32))
    net(x)

