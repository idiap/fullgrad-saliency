import torch.nn as nn
import torch.utils.model_zoo as model_zoo

cfg = { 
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'DC': [64, 'B64', 'M', 'B128', 128, 'M', 'B256', 256, 256, 'M', 'B512', 512, 512, 'M', 'B512', 512, 512, 'M'],
    'DAll': ['B64', 'B64', 'M', 'B128', 'B128', 'M', 'B256', 'B256', 'B256', 'M', 'B512', 'B512', 'B512', 'M', 'B512', 'B512', 'B512', 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
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
            if key_q == key.replace('.named_layer.','.'):
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


class VGG(nn.Module):

    def __init__(self, vgg_name, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(cfg[vgg_name], batch_norm=True)
        self.name = vgg_name
        #self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x, out_feat = self.organize_features(x)
        #x = self.features(x)
        x = x.view(x.size(0), -1)
        #x = x.sum()
        x = self.classifier(x)
        return x, out_feat

    def organize_features(self, x):
        in_channels = 3
        count = 0
        x_feat = None
        for i in cfg[self.name]:

            if (i == 'M'):
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
                
                count = count + 1
                x = self.features[count](x)

            count = count + 1

        return x, x_feat


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        index = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(v) == str:
                if v[0] == 'B':
                    x = int(v[1:])
                    layers += [NamedLayer(nn.Conv2d(in_channels, x, kernel_size = 3, padding=1), number = index),
                            NamedLayer(nn.BatchNorm2d(x), number=index), nn.ReLU(inplace=False)]
                    in_channels = x
                    index += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        #return nn.Sequential(*layers)
        return nn.ModuleList(layers)



def vgg16_bn():
    return VGG('DAll')

def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


#def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
#    if pretrained:
#        kwargs['init_weights'] = False
#    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
#    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
