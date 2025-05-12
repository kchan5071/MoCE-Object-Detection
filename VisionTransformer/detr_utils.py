import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy
from functools import partial

'''
STANDARD EXPECTED IMPLEMENTATION

Need 
    - Weights in the form of a state_dict as well as a string representation of the filepath
    - Images to make inference on in the form of a 640 x 640 x 3 tensor
        (NOTE - WillFlow expects a 3 x 640 x 640 tensor, clean_img takes a 640 x 640 x 3 and
        gets it into a useful state)
    
from detr_utils import WillFlow, clean_img, make_pred

model = WillFlow()
model.load_state_dict(
            torch.load(path_to_state_dict),
            weights_only=True
        )

pred_s : str = make_pred( # Format: class_label (1 for shark), x_ctr, y_ctr, width, height
        model,
        clean_img(data) # data expected to be 640 x 640 x 3 Tensor
    )

# Do whatever tf with pred_s u want

'''

# Code stolen from Halie to streamline loading data
# FIXME: KAI LOOK AT THE COMMENTS AT THE BOTTOM, SHITS IMPORTANT


class CustomObjectDetectionDataset(Dataset):
    def __init__(self,
                 image_dir,
                 label_dir,
                 transform=None
                 ):
        """
        Args:
            image_dir (string): Directory with all the images.
            label_dir (string): Directory with all the label text files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # image_files is an array of strings (filenames only)
        self.image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_files)

    def my_getitem(self, idx):
        """
        Retrieves a single sample (image and its labels) from the dataset at a given index
        Args:
            idx (int): Index of the sample to retrieve.

        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # opens and read img file + convert to RGB (to ensure 3-channel color representation)
        image = Image.open(img_path).convert("RGB") # I'm gonna trust you that you're not leaking memory all over the place here, Halie

        # Get corresponding label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        # Read bounding box from text file (format: label_id x_min x_max y_min y_max)
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                label, x_center, y_center, width, height = map(float, line.split()[:5])
                x_min = x_center - (width / 2)  # Integer division if pixel values must be integers
                y_min = y_center - (height / 2)
                x_max = x_center + (width / 2)
                y_max = y_center + (height / 2)
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(label)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        image = np.asarray(image)

        return image, target

    # Dear Kai,
    # Haile's dataset had no error handling, and I had to do some really jank
    # shit in my code that was keeping me from seeing more than one image

    # My solution to this was to wrap the functionality of __getitem__ in the
    # function above, and introduce error handling here. This will almost certainly
    # cause your code to be fucked later tho so heads-up. I think if I've done
    # this right (and Haile hasn't changed her Dataset implementation) you can
    # just copy-paste this in place of her definition and it shouldn't break any of
    # her shit, but @ me if something doesn't work

    def __getitem__(self, idx):
        try:
            image, target = CustomObjectDetectionDataset.my_getitem(self, idx)
        except Exception as E:
            msg = '''Error encountered trying to load data from the dataset
            using CustomObjectDetectionDataset: {}'''.format(str(E))
            print(msg)
            return -1, -1

        return image, target


class WillFlow(nn.Module):

  def __init__(self, num_encoder_layers=6, num_chunks=16, d_model=100, n_head=10):
    super().__init__()

    # Conv net to take 640 x 640 x 3 input down into token-sized
    # chunks (192 x 10 x 10)
    self.conv_net = nn.Sequential(
        nn.Conv2d(3, 6, 5, 2, 2),
        nn.Conv2d(6, 12, 5, 2, 2),
        nn.Conv2d(12, 24, 5, 2, 2),
        nn.Conv2d(24, 48, 5, 2, 2),
        nn.Conv2d(48, 96, 5, 2, 2),
        nn.Conv2d(96, 192, 5, 2, 2),
    )

    # FIXME: POSITIONAL ENCODING INFORMATION IS SCUFFED!
    self.pos_encoding = nn.Parameter(torch.rand(192, 100))

    # Transformer Encoder
    self._tsfm_encoder = nn.TransformerEncoderLayer(
        d_model,
        n_head
    )

    self.optimum_prime = nn.TransformerEncoder(
        self._tsfm_encoder,
        num_encoder_layers
    )

    # Output Predictions (BBox)
    self.pred_layer = nn.Sequential(
        nn.Linear(19200, 4),
        nn.Sigmoid()
    )



  def forward(self, x):
    x = self.conv_net(x).flatten(1)
    x = self.optimum_prime(x + 0.1 * self.pos_encoding)
    x = x.flatten(0) # Concat tokens
    x = self.pred_layer(x)
    return x


def halie_shoosh(position_labels : dict, print_vals=False):
      # I forget what motivated the name
      # Take your best guess

      # Anyway, this is here to turn the class label /
      # position dictionary into a single tensor, because
      # that's what my code expects

      if type(position_labels) is not dict:
          raise Exception('Uhh... this ain\'t a dict! Type: {}'.format(type(position_labels)))

      pos, key = position_labels.values()
      pos = pos.numpy().reshape((1, -1))
      key = key.numpy().reshape((1, -1))
      pk = from_numpy(np.hstack((key, pos)))
      if print_vals:
        print("Dict: {}".format(position_labels))
        print("Tensor: {}".format(pk))

      return pk




def clean_img(img, device='cpu'):
    # Function responsible for taking the output image (in the form of an ndarray 
    # and turning it into a useful image tensor with the information that the
    # model expects

    # The order of operations is as follows:
    # The ndarray is converted to a tensor
    # The channel order is swapped so the number of channels is now at index 0
    # The type of the tensor is changed to torch.float32 to be compatable with
    #   the rest of the model
    # Finally, the tensor is moved to the device (if none is specified, the cpu is used)
    if type(img) is np.ndarray:
        img = from_numpy(img)

    if len(img.shape) == 4:
        img = img.squeeze(0)

    return img.permute(2, 0, 1).to(torch.float32).to(device)

def clean_label(label, device='cpu') :
    
    # Similar to clean_img, used to encapsulate all the messiness associated with
    # parsing teh output data into a format that is actually useble
    
    # Importantly, for the cost function, the label isn't needed (lol) and the
    # input must be 1-dimensional (hence the reshape)
    # See above docs for uses of to()

    return label.reshape(5)[1:].to(torch.float32).to(device)

def train_loop(
        dataloader, 
        model, 
        loss_fn, 
        optimizer, 
        dbug=False, 
        device='cpu',
        path=None):

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    f_name = None
    if path:

        # Open a file to write the loss to every 20 batches
        f_name = open(path, 'w')

    for batch, (X, y) in enumerate(dataloader):

        if (X is -1) or (y is -1): # FIXME: THIS JANK ASS
          print("Foo!")
          continue

        # Shoosh, Halie
        y = halie_shoosh(y)

        # Compute prediction and loss
        X = clean_img(X, device)
        pred = model(X)
        y = clean_label(y, device)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # This shit is legitimately magic, and not well documented magic at that 
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            print("Pred: {} ||| Truth: {} ||| Loss: {}".format(pred, y, loss.item()))
            # Write loss to a file for later plotting shenaynays
            if f_name:
                # Sloppy string, should be 'batch_number, loss\n'
                f_name.write(str(batch) + ', ' + str(loss.item()) + '\n')

    if f_name:
        f_name.close()
        print('Training data saved at {}'.format(path))

def test_loop(dataloader, model, loss_fn, dbug=False, device='cpu'):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    # cprint = partial(cprint, dbug=dbug)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

          try:
            if X is -1 or y is -1: # FIXME: THIS JANK ASS
              print("Foo!")
              continue

            # Shoosh, Halie
            # cprint('Value of y prior to shooshing in test_loop: {}', y)

            y = halie_shoosh(y)
            X = clean_img(X, device)
            pred = model(X)
            y = clean_label(y, device)
            test_loss += loss_fn(pred, y).item()
          except Exception as E:
            print("Error in test_loop:  {}".format(str(E)))
            print("X: {} ||| y: {}".format(X, y))

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def save_model(model, path_name) : 
    # Saves the model in the form of a state_dict at the
    # specified path

    # If the file already exists, attempts to append (COPY)_ to the
    # start of the file name to preserve uniqueness. Also, bitches at you
    # about it.

    # Returns the path name to which the file was saved. If torch.save throws
    # an exception, behavior is undefined but program will not terminate.
   

    if path_name in os.listdir():
        path_name = '(COPY)_' + path_name
        print('Error - another file with the same name found in the path!')
        print('Saving instead at {}'.format())

    try:
        torch.save(
                model.state_dict(),
                path_name
                )

    except Exception as E:
        print('Unable to save model at {} due to the following error: {}'.format(path_name, str(E)))

    return path_name

def make_pred(model : WillFlow, data : torch.Tensor) -> str :
    # Wrapper to take a prediction from the model and return it as a
    # string in the format of:

    # 'class_label, x_ctr, y_ctr, width, height'

    # WARNING - THIS SHOULD ONLY BE USED FOR INFERENCE! INCLUDING THIS IN
    # TRAINING OF A WillModel WILL MOST LIKELY BREAK THE COMPUTATIONAL 
    # GRAPH!

    pred = model(data)
    pred_n = pred.detach().numpy() # Moving to numpy to minimize shenanigans, I hope

    ret = ''
    ret += str(
            # pred_n[0]
            str(1)
            )
    ret += ', '

    # Prediction is made in the format of:

    # [class_label, x_min, y_min, x_max, y_max]

    # Thus, it needs to be converted to the desired format
    x_dist = (pred_n[2] - pred_n[0]) / 2
    y_dist = (pred_n[3] - pred_n[1]) / 2
    x_ctr = pred_n[0] + x_dist
    y_ctr = pred_n[1] + y_dist

    ret += str(x_ctr) + ', '
    ret += str(y_ctr) + ', '
    ret += str(x_dist) + ', '
    ret += str(y_dist) 

    return ret

