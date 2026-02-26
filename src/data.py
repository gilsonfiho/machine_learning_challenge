import os
import scipy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image


class BMWDataset(Dataset):
    """Dataset customizado para BMW10 com balanceamento de classes"""
    
    def __init__(self, dataframe, img_dir, transform=None, augment=False, augment_factor=1):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.augment_factor = augment_factor
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        samples = []
        for idx, row in self.dataframe.iterrows():
            samples.append((row['filepath'], row['class_final']))
            # Augmentação para classes minoritárias
            if self.augment and self.augment_factor > 1:
                for _ in range(self.augment_factor - 1):
                    samples.append((row['filepath'], row['class_final']))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        full_path = os.path.join(self.img_dir, filepath)
        image = Image.open(full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_bmw10_data(data_path, annos_file='bmw10_annos.mat'):
    """
    Carrega dataset BMW10
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("\n" + "="*70)
    print("CARREGANDO DATASET BMW10")
    print("="*70)
    
    # Carregar anotações
    annos_path = os.path.join(data_path, annos_file)
    mat = scipy.io.loadmat(annos_path)
    annots = mat['annos'][0]

    filepaths, labels = [], []
    for entry in annots:
        img_path = entry[0][0]
        label = int(entry[1][0][0])
        filepaths.append(img_path)
        labels.append(label)

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    # Filtrar classes 3, 4, 5
    wanted = [3, 4, 5]
    df['label_filtered'] = df['label'].apply(lambda x: x if x in wanted else 0)
    
    # Mapear para [0, 1, 2, 3]
    mapping = {0: 0, 3: 1, 4: 2, 5: 3}
    df['class_final'] = df['label_filtered'].map(mapping)

    print("\nDISTRIBUIÇÃO ORIGINAL")
    print("-" * 70)
    for i in range(4):
        count = (df['class_final'] == i).sum()
        pct = (count / len(df)) * 100
        print(f"  Classe {i}: {count:5d} amostras ({pct:5.1f}%)")
    print(f"  Total:     {len(df):5d} amostras")
    print("-" * 70)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df['class_final'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df['class_final'], random_state=42
    )

    print(f"\n✓ Treino:      {len(train_df):5d} ({len(train_df)/len(df)*100:5.1f}%)")
    print(f"✓ Validação:   {len(val_df):5d} ({len(val_df)/len(df)*100:5.1f}%)")
    print(f"✓ Teste:       {len(test_df):5d} ({len(test_df)/len(df)*100:5.1f}%)")
    print("="*70 + "\n")

    return train_df, val_df, test_df
