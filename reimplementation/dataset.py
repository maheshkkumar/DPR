import torch
from torch.utils.data import Dataset


class IlluminationDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        if idx > 10:
            assert idx < 10, "Stop loader"
        # TODO: complete this later
        source_img = torch.rand(1, 512, 512)
        source_light = torch.rand(9, 1, 1)
        target_img = torch.rand(1, 512, 512)
        target_light = torch.rand(9, 1, 1)

        return source_img, source_light, target_img, target_light

    def __len__(self):
        return 10
