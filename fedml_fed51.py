"""
no category splitting, just size splitting
5 clients: 20K, 15K, 5K, 5K, 5K
"""

from torch.utils.data import random_split
from fedml import getCIFAR100TrainingDataset, runEntireSimulation, saveStuff
import torch

torch.manual_seed(0)
trainingdata = getCIFAR100TrainingDataset()
train_split_datasest = random_split(trainingdata, [20000, 15000, 5000, 5000, 5000])

sim_weights, sim_metrics = runEntireSimulation(
    trainingDataSplit=train_split_datasest,
    clientsPerRound=4,
    iterPerClient=1,
    batchSize=128,
    momentum=0.9,
    weightDecay=5e-4,
    epochLRSchedule=[(50, 0.1), (30, 0.02)],
)

name = 'fedml_fed51'

saveStuff(sim_weights, './fedml-weights/{}.weights'.format(name))
saveStuff(sim_metrics, './fedml-metrics/{}.metrics'.format(name))

