import os

import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder

SAMPLE_RATE = 16000


# idx_to_label = {
#     0: 'an',    #anger
#     1: 'co',    #contempt
#     2: 'ha',    #happy
#     3: 'an',    #angry
#     4: 'sa',    #sad
# }

# encoder = LabelEncoder()
# encoder.fit(['an', 'co', 'ha', 'sa'])

class EmoFilmDataset(Dataset):
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def __getitem__(self, idx):
        wav_path = os.path.join(self.path, self.data[idx][0])
        wav, sr = torchaudio.load(wav_path)
        label = self.data[idx][1]
        # label = encoder.transform([label])
        return wav.view(-1), torch.tensor(label).long()

    def __len__(self):
        return len(self.data)

    # def collate_fn(self, samples):
    #     wavs, labels, languages, genders = [], [], [], []
    #     for wav, label, language, gender in samples:
    #         wavs.append(wav)
    #         labels.append(label)
    #         languages.append(language)
    #         genders.append(gender)
    #     return wavs, labels, languages, genders
    
    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels