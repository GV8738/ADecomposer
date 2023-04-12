import torch
import warnings
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


import configuration
from models import Vae
from utils import utils
from dataset import prepare_dataset

categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
              'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
warnings.filterwarnings("ignore")

def soft_threshold(x, a=0.5):
    return torch.sgn(x)*torch.nn.functional.relu(x.abs() - a, 0)

def main(args):
    device = torch.device('cuda')

    # Getting dataset/dataloader
    img_transforms, mask_transforms = prepare_dataset.get_transforms(args.size)
    te_dataset = prepare_dataset.MVTecDataset(root=args.dataroot,
        target=args.category,
        train=False,
        transforms=img_transforms,
        mask_transforms=mask_transforms)
    te_dataloader = DataLoader(te_dataset, batch_size=1, shuffle=False)

    # Create model
    model = Vae.VanillaVAE(in_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256, 512]).to(device)
    weights = torch.load(args.root + '/weights/vae_' + args.category + '_200_cycle.pth')
    #weights = torch.load(args.root + '/VAE/capsule.pth')
    model.load_state_dict(weights['model'])
    #model.load_state_dict(weights['state_dict'])
    model.eval()

    ρ = args.rho
    step = 0.001
    scores = []
    pixels = []
    labels = []
    masks = []
    multi_error = []
    multi_eps = []
    multi_clean = []
    originals = []
    j = 0
    for (x, mask, label) in te_dataloader:
        error = []
        epsilons = []
        cleans = []

        #Initialization
        x = x.to(device)
        ε = torch.zeros_like(x, dtype=torch.float32).to(device)
        u = torch.zeros_like(x, dtype=torch.float32).to(device)
        y = x

        #Optimization loop
        for i in range(args.iter):
            #Reconstruction
            ŷ = model.generate(y)

            #Update of y
            y = (1/(ρ + 1))*(ŷ + ρ*(x - ε + u))

            #Update of ε
            v = x - y + u
            ε[0,0,:] = (1 - (args.lamb/ρ)/torch.norm(v[:,0])).relu()*v[0,0]
            ε[0,1,:] = (1 - (args.lamb/ρ)/torch.norm(v[:,0])).relu()*v[0,1]
            ε[0,2,:] = (1 - (args.lamb/ρ)/torch.norm(v[:,0])).relu()*v[0,2]
            #ε = soft_threshold(x - y + u, a=args.lamb/args.rho)

            #Update of u
            u = u + ρ*(x - y - ε)

            #Error
            #error.append((ε.mean().item()))
            #epsilons.append(ε.cpu())
            #cleans.append(y.cpu())

            #Checking approximation
            #if ((y + ε) - x).sum() <= 0.001:
            #    break

        '''multi_error.append([error, label])
        if label != 0 and j <= 5:
            multi_eps.append(torch.cat(epsilons).cpu())
            multi_clean.append(torch.cat(cleans).cpu())
            originals.append(x.cpu())
            j += 1'''

        #Anomaly score
        A_map = ε.prod(1)
        A_score = ε.max().cpu().item()
        scores.append(A_score)
        pixels.extend(utils.min_max_norm(A_map).flatten().detach().cpu().numpy())
        utils.relabel(label)
        labels.append(label.item())
        masks.extend(mask.flatten().cpu().numpy().astype(int))

    scores = np.array(scores)
    pixels = np.array(pixels)
    labels = np.array(labels)
    masks = np.array(masks)

    #Image-level AUROC
    img_auroc = roc_auc_score(labels, scores)

    #Pixel-level AUROC
    pxl_auroc = roc_auc_score(masks, pixels)

    print('Image-level AUROC: ', img_auroc)
    print('Pixel-level AUROC: ', pxl_auroc)
    #vutils.save_image(multi_eps[-2].data, args.root + '/samples/' + args.category + '_ad_eps_' + str(args.iter) + '.png', normalize=True)
    #vutils.save_image(multi_clean[-2].data, args.root + '/samples/' + args.category + '_ad_y_' + str(args.iter) + '.png', normalize=True)
    #vutils.save_image(originals[-2].data, args.root + '/samples/' + args.category + '_ad_orig_' + str(args.iter) + '.png', normalize=True)

    #Plotting error
    '''for elem in multi_error:
        if elem[1] == 0:
            timesteps = np.arange(args.iter)
            plt.plot(timesteps, multi_error[-1][0])
            plt.xlabel('Iterations')
            plt.ylabel('Noise ε')
            plt.title('Evolution of noise ε')
            plt.show()
            break'''

if __name__ == '__main__':
    args = configuration.argparser()
    main(args)