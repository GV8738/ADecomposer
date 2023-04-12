import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

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
    te_dataset = prepare_dataset.MVTecDataset(root=args.dataroot,
        target=args.category,
        train=False,
        transforms=img_transforms,
        mask_transforms=mask_transforms)
    tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=False)
    te_dataloader = DataLoader(te_dataset, batch_size=4, shuffle=False)

    #Create model
    model = epsVAE.εVAE(in_channels=3, latent_dim=256, hidden_dims=[32, 64, 128, 256, 512]).to(device)
    weights = torch.load(args.root + '/weights/epsVAE_' + args.category + '_200_cycle256.pth')
    model.load_state_dict(weights['model'])

    #Training
    i = 0
    for (x, _, label) in te_dataloader:
        x = x.to(device)
        y, ε, _, mu, var = model(x)
        if label.sum() != 0:
            if i == 1:
                vutils.save_image(x.cpu().data, args.root + '/samples/x_epsVAE_ano.png', normalize=True)
                vutils.save_image(y.cpu().data, args.root + '/samples/y_epsVAE_ano.png', normalize=True)
                vutils.save_image(ε.cpu().data, args.root + '/samples/eps_epsVAE_ano.png', normalize=True)
                vutils.save_image((x - y).abs().prod(1).unsqueeze(1).cpu().data, args.root + '/samples/dif_epsVAE_ano.png', normalize=True)
                break
            i += 1




if __name__ == '__main__':
    args = configuration.argparser()
    main(args)