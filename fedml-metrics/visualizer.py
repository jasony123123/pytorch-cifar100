import torch
import matplotlib.pyplot as plt

names = [
         'fedml_fed1',
         'fedml_fed2',
         'fedml_fed3',
         'fedml_syncSgdSmall'
]
for name in names:
    metrics = torch.load('{}.metrics'.format(name))
    plt.plot([x['train_acc1'] for x in metrics], label='train_acc1')
    plt.plot([x['test_acc1'] for x in metrics], label='test_acc1')
    plt.title('{}_acc1'.format(name))
    plt.show()
    plt.plot([x['train_loss'] for x in metrics], label='train_loss')
    plt.plot([x['test_loss'] for x in metrics], label='test_loss')
    plt.title('{}_loss'.format(name))
    plt.show()