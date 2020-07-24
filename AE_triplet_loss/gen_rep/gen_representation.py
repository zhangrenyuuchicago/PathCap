import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

#from torchvision.datasets import MNIST
import SlideDataset
import os


def to_img(x):
    #x = 0.5 * (x + 1)
    #x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 292, 292)
    return x

embed_size = 460
batch_size = 1024

img_transform = transforms.Compose([
    transforms.Resize([292,292]),
    transforms.ToTensor()
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_image_data = SlideDataset.SlideDataset('/home/zhangr/data/GTEx_Tiles_jpg/', img_transform)
dataloader = DataLoader(train_image_data, batch_size=batch_size, shuffle=False, num_workers=10)
#dataloader = DataLoader(train_image_data, batch_size=batch_size, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.encoder_l = nn.Linear(4608, embed_size)
        self.decoder_l = nn.Linear(embed_size, 4608)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_encoder_cnn = self.encoder_cnn(x)
        #print(f'embeding size: {x_embed.size()}')
        cnn_size = x_encoder_cnn.size()
        x_encoder_cnn = x_encoder_cnn.view((cnn_size[0], -1))
        x_embed = self.encoder_l(x_encoder_cnn)
        x_embed = self.sig(x_embed)
        x_decoder_l = self.decoder_l(x_embed)
        x_decoder_l = x_decoder_l.view(cnn_size)
        x_decoder_cnn = self.decoder_cnn(x_decoder_l)
        return x_embed, x_decoder_cnn


model = autoencoder()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('../checkpoint/checkpoint_autoencoder_7.pth')
model.load_state_dict(checkpoint)
criterion = nn.MSELoss()
model.eval()

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                             weight_decay=1e-5)

step = 0

#img_rep = {}
fout = open('representation_all_7epoch.txt', 'w')

for data in dataloader:
    img, img_name = data
    img = Variable(img).cuda()
    # ===================forward=====================
    rep, rec_img = model(img)
    # output = model(img)
    #loss = criterion(output, img)
    print(step)
    rep = rep.cpu().data
    for i in range(len(img_name)):
        cur_name = img_name[i]
        cur_rep = rep[i].numpy().reshape((-1)).tolist()
        #img_rep[cur_name] = cur_rep
    
        fout.write(cur_name )
        for j in cur_rep:
            fout.write('\t' + str(j))
        fout.write('\n')

    step += 1
fout.close()

