from torch.utils.data import Dataset

class PosPairDataset(Dataset):
    def __init__(self, pairs) -> None:
        super().__init__()
        self.pairs = pairs
        
    def __len__(self,):
        return len(self.pairs)
    
    def __getitem__(self, index):
        pos_u, pos_v = self.pairs[index]
        return pos_u, pos_v