import torchvision
import pretrainedmodels
from .CFEANet import CFEANet
from .CFEANetV2 import CFEANetV2
from .CFEANetV3 import CFEANetV3
from .eca_resnet import eca_resnet101
from .resnet_cbam import ResidualNet
print(pretrainedmodels.model_names)
model_dict = {'CFEANet':CFEANet,
              'CFEANetV2':CFEANetV2,
              'CFEANetV3':CFEANetV3,}

def get_model(num_classes, args):
    # res101 = torchvision.models.squeezenet1_0(pretrained=True)
    # se_resnet101=pretrainedmodels.se_resnet101(num_classes=1000, pretrained='imagenet')
    # eca101=eca_resnet101(k_size="3357", num_classes=1000, pretrained=False)
    CBAM= ResidualNet(network_type="ImageNet",depth=50, num_classes=1000, att_type="CBAM")
    model = model_dict[args.model_name](CBAM, num_classes)
    return model