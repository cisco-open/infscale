import torch, time, os
from torch.utils.data import DataLoader, Dataset
import deepspeed
from deepspeed.pipe import PipelineModule
from torchvision.models.resnet import resnet50, ResNet50_Weights

num_batches = 100
num_classes = 1000
batch_size = 64
image_w = 224
image_h = 224

class MyDataset(Dataset):
    def __init__(self, img_list):
        super(MyDataset, self).__init__()
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        return img

def flat_func(x):
    return torch.flatten(x, 1)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))

deepspeed.init_distributed(rank=local_rank, world_size=world_size)

net = resnet50()
layers = [
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4,
        net.avgpool,
        flat_func,
        net.fc
    ]

# generating inputs
inputs = [torch.randn(1, 3, image_w, image_h, dtype=torch.float) for i in range(num_batches * batch_size)]
dataset = MyDataset(inputs)
dataloader = DataLoader(inputs, batch_size=batch_size)

model = PipelineModule(layers, 2)
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config="ds_config.json",
    training_data=dataset)
model = model_engine

tik = time.time()
for i in range(num_batches):
    outputs = model.eval_batch(iter(dataloader))

tok = time.time()
print(f"{tok - tik}, {(num_batches * batch_size) / (tok - tik)}")
