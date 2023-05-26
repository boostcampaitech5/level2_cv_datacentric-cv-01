import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

