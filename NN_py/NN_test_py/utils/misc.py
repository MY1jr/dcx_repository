import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AveragePerEle(object):
    '''
    compute and store the average value of each element
    '''
    def __init__(self, size=(1,1)):
        self.size = size  # tuple
        self.reset()

    def reset(self):
        self.val = np.zeros(self.size)
        self.avg = np.zeros(self.size)
        self.sum = np.zeros(self.size)
        self.count = np.zeros(self.size)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
