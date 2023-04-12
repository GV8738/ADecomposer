import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

import configuration
from models import epsVAE
from utils import utils
from dataset import prepare_dataset


def main(args):
    device = torch.device('cuda')
    cudnn.benchmark = True

    # Getting dataset/dataloader
    img_transforms, mask_transforms = prepare_dataset.get_transforms(args.size)
    tr_dataset = prepare_dataset.MVTecDataset(root=args.dataroot,
        target=args.category,
        train=True,
        transforms=img_transforms,
        mask_transforms=mask_transforms)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)

    #Create model
    model = epsVAE.εVAE(in_channels=3, latent_dim=256, hidden_dims=[32, 64, 128, 256, 512]).to(device)

    #Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    #KL weight
    lamb = utils.cycle_cosine(start=0.0, stop=1.0, n_epoch=args.nepochs)

    #Training
    for epoch in tqdm(range(args.nepochs)):
        for (x, _, _) in tr_dataloader:
            x = x.to(device)
            y, ε, _, mu, var = model(x)
            loss = model.loss_function(y, ε, x, mu, var, M_N=0.01, gamma=args.gamma)['loss']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(('Epoch %d/%d:' % (epoch, args.nepochs)).ljust(24) +'%.2f' % loss.item())

        #Saving weights
        dict = {'model': model.state_dict()}
        torch.save(dict, args.root + '/weights/epsVAE_' + args.category + '_' + str(args.nepochs) + '_cycle256.pth')

if __name__ == '__main__':
    args = configuration.argparser()
    main(args)