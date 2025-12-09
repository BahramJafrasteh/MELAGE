import torch
import torch.utils.data as data
import abc

class baseData(data.Dataset):
    def __init__(self):
        super(baseData, self).__init__()
    @abc.abstractmethod
    def options(self, opt):
        raise NotImplementedError("Subclass should be implemented.")

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclass should be implemented.")

    @abc.abstractmethod
    def name(self):
        raise NotImplementedError("Subclass should be implemented.")