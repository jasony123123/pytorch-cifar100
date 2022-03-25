"""
5 clients, each with 20 classes
all clients every round
"""

from torch.utils.data import random_split
from fedml import getCIFAR100TrainingDataset, runEntireSimulation, saveStuff

trainingdata = getCIFAR100TrainingDataset()
datasplit = [[] for i in range(5)]
for i in range(50000):
    datasplit[trainingdata[i][1] % 5].append(trainingdata[i])

sim_weights, sim_metrics = runEntireSimulation(
    trainingDataSplit=datasplit,
    clientsPerRound=5,
    iterPerClient=1,
    batchSize=128,
    momentum=0.9,
    weightDecay=5e-4,
    epochLRSchedule=[(50, 0.1), (30, 0.02)],
)

name = 'fedml_fed4'

saveStuff(sim_weights, './fedml-weights/{}.weights'.format(name))
saveStuff(sim_metrics, './fedml-metrics/{}.metrics'.format(name))