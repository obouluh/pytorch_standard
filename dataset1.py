from torch.utils.data import Dataset

class MyDataset(Dataset):
    # 加载数据，预处理数据集，划分训练集和测试集等操作
    def __init__(self) :
        # super().__init__()
        pass
    # 返回第index个样本的数据和标签
    def __getitem__(self, index) :
        # return super().__getitem__(index)
        pass
    # 返回数据集大小
    def __len__(self):
        pass

