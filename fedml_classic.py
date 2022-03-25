"""
classic training
"""

from fedml import getCIFAR100TrainingDataset, runEntireSimulation, saveStuff

trainingdata = getCIFAR100TrainingDataset()

sim_weights, sim_metrics = runEntireSimulation(
    trainingDataSplit=[trainingdata],
    clientsPerRound=1,
    iterPerClient=1,
    batchSize=128,
    momentum=0.9,
    weightDecay=5e-4,
    epochLRSchedule=[(50, 0.1), (30, 0.02)],
)

name = 'fedml_classic'

saveStuff(sim_weights, './fedml-weights/{}.weights'.format(name))
saveStuff(sim_metrics, './fedml-metrics/{}.metrics'.format(name))