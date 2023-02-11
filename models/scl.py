import scattering_transform
from scattering_transform import SCLTrainingWrapper

kwargs={
    "image_size":160,                          # size of image
    "set_size": 9,                               # number of questions + 1 answer
    "conv_channels": [1, 16, 16, 32, 32, 32],    # convolutional channel progression, 1 for greyscale, 3 for rgb
    "conv_output_dim": 80,                       # model dimension, the output dimension of the vision net
    "attr_heads": 10,                            # number of attribute heads
    "attr_net_hidden_dims": [128],               # attribute scatter transform MLP hidden dimension(s)
    "rel_heads": 80,                             # number of relationship heads
    "rel_net_hidden_dims": [64, 23, 5] 
}

SCL_model=scattering_transform.SCL(**kwargs)