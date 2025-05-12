from detr_utils import *
from os.path import join
from os import chdir, listdir

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch

_TRAIN_LABEL_DIR = r'./r_l' 
_TRAIN_IMG_DIR = r'./r_s' 

_TEST_LABEL_DIR = r'./s_l'
_TEST_IMG_DIR = r'./s_s'
_EPOCHS = 5


if __name__ == '__main__' : # I hate this shit, I still don't know what it is doing under the hood

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # device = 'cpu' # COPIUM

    print(f"Using {device} device")
    

    train_ds = CustomObjectDetectionDataset(
            _TRAIN_IMG_DIR,
            _TRAIN_LABEL_DIR,
            transform=None
            )

    test_ds = CustomObjectDetectionDataset(
                _TEST_IMG_DIR,
                _TEST_LABEL_DIR,
                transform=None
            )

    train_dl = DataLoader(
            train_ds,
            batch_size = 1
            )

    test_dl = DataLoader(
            test_ds,
            batch_size=1
            )

    model = WillFlow()
    model = model.to(device)


    loss_fn = MSELoss()
    optimizer = SGD(
                model.parameters(), # Params to optimize
                lr=0.0005, # Learning rate
                momentum=0.1 # Scalar product of inertia (mass) and velocity 
            )

    print('Starting training!')
    for e in range(_EPOCHS):
        path = r'./epoch_{}_data.csv'.format(e + 1)
        train_loop(
                    train_dl,
                    model,
                    loss_fn,
                    optimizer,
                    device=device,
                    path=path
                )

        test_loop(
                    test_dl,
                    model,
                    loss_fn,
                    device=device
                )

    print('Training complete!')
    save_model(model, r'./JUSTICE_WEIGHTS.pth')
    print('Did that do what you needed it to? Do you feel more accomplished?')
    print('Like a better person, perhaps?')

    inp = input('--->')
    if inp != 'JUST1C3':
        print('Try again, dunkath!')

    else :
        print('Those cut from the same cloth are fit to hang from the same rope!')





