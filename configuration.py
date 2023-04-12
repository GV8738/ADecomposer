import argparse

def argparser():
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--root', type=str, default='/export/livia/home/vision/gvargas/ADecomposer', help='Base path')
    parser.add_argument('--dataroot', type=str, default='/export/livia/home/vision/gvargas/data/MVTec/')

    #Dataset
    parser.add_argument('--category', type=str, default='bottle', help='Category from MVTec')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    #Training
    parser.add_argument('--train', type=bool, default=True, help='Option to start trainig, otherwise just testing')
    parser.add_argument('--nepochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--kl-weight', type=float, default=0.001, help='Weight for the KL divergence in loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='Weight for L1 norm on noise')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')

    #Optimization
    parser.add_argument('--iter', type=int, default=10, help='Iterations for optimization')
    parser.add_argument('--rho', type=float, default=0.5, help='Hyperparameter value')
    parser.add_argument('--lamb', type=float, default=1.0, help='Weight for the prior in the loss')

    return parser.parse_args()