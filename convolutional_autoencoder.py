import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 2))
        return x


num_epochs = 10
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST("data", transform=img_transform, download=True)
trainData, testData = train_test_split(dataset, train_size=0.8)
trainDataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
testDataloader = DataLoader(testData, batch_size=32, shuffle=False)

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in trainDataloader:
        img, _ = data
        img = Variable(img).cuda()
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data))

itr = iter(testDataloader)

for i in range(10):
    data = next(itr)
    img, _ = data
    save_image(img, 'conv/original_image_{}.png'.format(i))
    img = Variable(img).cuda()
    output = model(img)
    pic = to_img(output.cpu().data)
    save_image(pic, 'conv/reconstructed_image_{}.png'.format(i))
