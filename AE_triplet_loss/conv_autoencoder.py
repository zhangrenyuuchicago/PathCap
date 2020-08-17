import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from datetime import datetime
import SlideDataset
import os

def to_img(x):
    #x = 0.5 * (x + 1)
    #x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 292, 292)
    x = x[0]
    return x

num_epochs = 100
batch_size = 512
learning_rate = 1e-3
embed_size = 460
mu = 0.1

img_transform = transforms.Compose([
    transforms.Resize([292,292]),
    transforms.ToTensor()
])

train_image_data = SlideDataset.SlideDataset('/home/zhangr/data/GTEx_Tiles_jpg/', img_transform)
dataloader = DataLoader(train_image_data, batch_size=batch_size, shuffle=True, num_workers=4)

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

model = autoencoder().cuda()
model = torch.nn.DataParallel(model).cuda()

criterion = nn.MSELoss()
criterion_embed = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

step = 0
writer = SummaryWriter(logdir="runs/autoencoder_" + datetime.now().strftime('%b%d_%H-%M-%S'))

ac_loss = 0

for epoch in range(num_epochs):
    for data in dataloader:
        img, pos_img, neg_img, base_name, pos_name, neg_name = data
        img = Variable(img).cuda()
        pos_img = Variable(pos_img).cuda()
        neg_img = Variable(neg_img).cuda()
        # ===================forward=====================
       
        img_embed, img_rec = model(img)
        pos_img_embed, pos_img_rec = model(pos_img)
        neg_img_embed, neg_img_rec = model(neg_img)

        rec_loss = criterion(img_rec, img)
        
        pos_loss = torch.mean((img_embed - pos_img_embed) * (img_embed - pos_img_embed), dim=1)
        neg_loss = torch.mean((img_embed - neg_img_embed) * (img_embed - neg_img_embed), dim=1) 
        triplet_loss = torch.mean(torch.clamp(pos_loss - neg_loss + 0.001, min=0.0))
        loss = rec_loss + mu*(triplet_loss)
        
        print(f'epoch: {epoch}, step: {step}, rec_loss: {rec_loss.data.cpu().numpy():0.6f}, triplet_loss: {triplet_loss.data.cpu().numpy():.6f}')
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('step/mse_loss', loss.data.item(), step)

        step += 1
        ac_loss += loss.data.item()

    
    writer.add_scalar('epoch/mse_loss', ac_loss, epoch)
    ac_loss = 0
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data.item()))
    
    torch.save(model.state_dict(), f'checkpoint/checkpoint_autoencoder_{epoch}.pth')

writer.close()
