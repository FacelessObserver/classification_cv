import random
import numpy as np
import torch as t

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
t.manual_seed(SEED)
t.cuda.manual_seed(SEED)
t.backends.cudnn.deterministic = True

DEVICE = "cpu"

BATCH_SIZE = 100

HIDDEN_NEURONS = 20
EPOCHS = 100
LEARNING_RATE = 0.01

CRITERION = t.nn.CrossEntropyLoss()
OPTIMIZER = t.optim.SGD
OPTIMIZER_KWARGS = {}

MODEL_PATH = "models/cv_net.pth"
