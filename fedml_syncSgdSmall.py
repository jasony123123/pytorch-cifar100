"""
10 clients, IID
"""

from torch.utils.data import random_split
from fedml import getCIFAR100TrainingDataset, runEntireSimulation, saveStuff

trainingdata = getCIFAR100TrainingDataset()
train_split_datasest = random_split(trainingdata, [5000 for i in range(10)])

sim_weights, sim_metrics = runEntireSimulation(
    trainingDataSplit=train_split_datasest,
    clientsPerRound=10,
    iterPerClient=1,
    batchSize=128,
    momentum=0.9,
    weightDecay=5e-4,
    epochLRSchedule=[(50, 0.1), (30, 0.02)],
)

name = 'fedml_syncSgdSmall'

saveStuff(sim_weights, './fedml-weights/{}.weights'.format(name))
saveStuff(sim_metrics, './fedml-metrics/{}.metrics'.format(name))

