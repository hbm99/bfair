
import pandas as pd
from bfair.datasets.build_tools.multifairface import create_mixed_dataset

from bfair.datasets.fairface import load_dataset as load_fairface
import datasets as db

from .base import Dataset

CIFAR_IMAGE_COLUMN = 'img'
IMAGE_COLUMN = 'image'

SIZE = 15000


def load_dataset(split_seed=None, **kwargs):
    return NoisyMultiFairFaceDataset.load(split_seed=split_seed)


class NoisyMultiFairFaceDataset(Dataset):

    @classmethod
    def load(cls, split_seed = 0):
        df_ff = load_fairface(split_seed)

        source_noisy_dataset = db.load_dataset("cifar100")
        df_noisy = pd.DataFrame.from_dict(source_noisy_dataset)

        # Remove undesired classifications (people related)
        df_noisy = df_noisy[~df_noisy['fine_label'].isin([2, 11, 35, 46, 98])]
        df_noisy = df_noisy[~df_noisy['coarse_label'].isin([14])]

        new_df_noisy = pd.DataFrame(columns=df_ff.columns)

        new_df_noisy[IMAGE_COLUMN] = df_noisy[CIFAR_IMAGE_COLUMN]

        print(new_df_noisy.head(5))

        new_df_noisy = pd.concat([df_ff, new_df_noisy])

        # Shuffle the rows
        new_df_noisy = new_df_noisy.sample(frac=1, random_state=split_seed).reset_index(drop=True)

        mixed_data = create_mixed_dataset(new_df_noisy, SIZE)

        return NoisyMultiFairFaceDataset(data=mixed_data, split_seed=split_seed)

        
        




    

        
        

