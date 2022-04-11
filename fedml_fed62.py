"""
soft category splitting - https://arxiv.org/pdf/2102.02079.pdf
set # clients, parition label data into a small # of splits - divided equlally amongst clients
# client 50
# splits 10
"""

from fedml import getCIFAR100TrainingDataset, runEntireSimulation, saveStuff
import random

nClients = 50
nSplits = 10 # <<500 && 100*nSplits >= nClients

nLabels = 100
trainingdata = getCIFAR100TrainingDataset()
random.seed(0)
labelQueue = [(i%nLabels) for i in range(nLabels * nSplits)]
random.shuffle(labelQueue)
labelToClients = {}
for l in range(nLabels):
    labelToClients[l] = []
for i in range(len(labelQueue)):
    labelToClients[labelQueue[i]].append(i % nClients)
datasplit = [[] for i in range(nClients)]
for i in range(50000):
    l = trainingdata[i][1]
    datasplit[labelToClients[l][0]].append(trainingdata[i])
    labelToClients[l] = labelToClients[l][1:] + labelToClients[l][:1]
    
sim_weights, sim_metrics = runEntireSimulation(
    trainingDataSplit=datasplit,
    clientsPerRound=15,
    iterPerClient=2,
    batchSize=128,
    momentum=0.9,
    weightDecay=5e-4,
    epochLRSchedule=[(50, 0.1), (30, 0.02)],
)

name = 'fedml_fed62'

saveStuff(sim_weights, './fedml-weights/{}.weights'.format(name))
saveStuff(sim_metrics, './fedml-metrics/{}.metrics'.format(name))