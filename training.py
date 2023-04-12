import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

import configuration
from models import Vae
from utils import utils
from dataset import prepare_dataset


def main(args):
    device = torch.device('cuda')

    # Getting dataset/dataloader
    img_transforms, mask_transforms = prepare_dataset.get_transforms(args.size)
    tr_dataset = prepare_dataset.MVTecDataset(root=args.dataroot,
        target=args.category,
        train=True,
        transforms=img_transforms,
        mask_transforms=mask_transforms)
    te_dataset = prepare_dataset.MVTecDataset(root=args.dataroot,
        target=args.category,
        train=False,
        transforms=img_transforms,
        mask_transforms=mask_transforms)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    te_dataloader = DataLoader(te_dataset, batch_size=1, shuffle=False)

    #Create model
    model = Vae.VanillaVAE(in_channels=3, latent_dim=256, hidden_dims=[32, 64, 128, 256, 512]).to(device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #KL weight
    lamb = utils.cycle_cosine(start=0.0, stop=1.0, n_epoch=args.nepochs)

    #Training
    for epoch in tqdm(range(args.nepochs)):
        for (x, _, _) in tr_dataloader:
            x = x.to(device)
            x_o, _, mu, var = model(x)
            loss = model.loss_function(x_o, x, mu, var, M_N=lamb[epoch])['loss']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        #Saving weights
        dict = {'model': model.state_dict()}
        torch.save(dict, args.root + '/weights/vae_' + args.category + '_' + str(args.nepochs) + '_cycle256.pth')

if __name__ == '__main__':
    args = configuration.argparser()
    main(args)