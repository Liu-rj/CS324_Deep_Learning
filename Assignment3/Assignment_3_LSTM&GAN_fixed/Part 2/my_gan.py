import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2)))
        self.layers.append(
            nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        )
        self.layers.append(
            nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        )
        self.layers.append(
            nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        )
        self.layers.append(nn.Linear(1024, 784))

    def forward(self, z):
        # Generate images from z
        h = z
        for layer in self.layers:
            h = layer(h)
        return torch.tanh(h)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(784, 512), nn.LeakyReLU(0.2)))
        self.layers.append(nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2)))
        self.layers.append(nn.Linear(256, 1))

    def forward(self, img):
        # return discriminator score for img
        h = img
        for layer in self.layers:
            h = layer(h)
        return torch.sigmoid(h)


def train(args, dataloader, discriminator, generator, optimizer_G, optimizer_D):
    criterion = nn.BCELoss()
    device = args.device
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device).flatten(1)
            bs = imgs.shape[0]
            noise = torch.randn((bs, args.latent_dim), device=device)
            imgs_label = torch.ones(bs, 1, device=device)
            fake_label = torch.zeros(bs, 1, device=device)

            # Train Generator
            # ---------------
            gen_imgs = generator(noise)
            pred = discriminator(gen_imgs)
            loss_G = criterion(pred, imgs_label)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            # Train Discriminator
            # -------------------
            imgs_pred = discriminator(imgs)
            fake_pred = discriminator(gen_imgs.detach())
            loss_D = (
                criterion(imgs_pred, imgs_label) + criterion(fake_pred, fake_label)
            ) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                gen_imgs = gen_imgs.reshape(args.batch_size, 1, 28, 28)
                save_image(
                    gen_imgs[:25],
                    "images/{}.png".format(batches_done),
                    nrow=5,
                    normalize=True,
                )
                print(f"{batches_done} batches: images saved")
                # pass


def main(args):
    # Create output image directory
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Initialize models and optimizers
    device = args.device
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(args, dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "models/mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="device to train")
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="save every SAVE_INTERVAL iterations",
    )
    args = parser.parse_args()

    main(args)
