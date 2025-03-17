import os
from typing import Sequence

import lmdb
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import numpy as np
try:
    from proto.meta_audio_file_pb2 import MetaAudioFile
except:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from proto.meta_audio_file_pb2 import MetaAudioFile

class MetaAudioDataset(Dataset):

    def __init__(self,
                 db_path: str):
        super().__init__()
        self._db_path = db_path
        self._env = None
        self._keys = None

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, readonly=True, lock=False, writemap=True)
        return self._env
    
    @property
    def keys(self) -> Sequence[str]:
        if self._keys is None:
            with self.env.begin() as txn:
                self._keys = [key for key, _ in txn.cursor()]
        return self._keys
    
    @property
    def max(self) -> float:
        max_value = 0.
        for key in self.keys:
            with self.env.begin() as txn:
                serialized = txn.get(key)
            meta_audio_file = MetaAudioFile()
            meta_audio_file.ParseFromString(serialized)

            emb = np.frombuffer(meta_audio_file.encoder_outputs.embeddings.data, dtype=np.float32)
            max_value = max(max_value, emb.max())
        return max_value
    
    @property
    def min(self) -> float:
        min_value = 0.
        for key in self.keys:
            with self.env.begin() as txn:
                serialized = txn.get(key)
            meta_audio_file = MetaAudioFile()
            meta_audio_file.ParseFromString(serialized)

            emb = np.frombuffer(meta_audio_file.encoder_outputs.embeddings.data, dtype=np.float32)
            min_value = min(min_value, emb.min())
        return min_value
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index: int):
        key = self.keys[index]
        with self.env.begin() as txn:
            serialized = txn.get(key)
        meta_audio_file = MetaAudioFile()
        meta_audio_file.ParseFromString(serialized)

        audio_data = np.frombuffer(meta_audio_file.audio_file.data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / (2**15 - 1)
        audio_data = audio_data.reshape(meta_audio_file.audio_file.num_channels, -1)

        metadata = {
            "note": meta_audio_file.metadata.note,
            "note_str": meta_audio_file.metadata.note_str,
            "instrument": meta_audio_file.metadata.instrument,
            "instrument_str": meta_audio_file.metadata.instrument_str,
            "pitch": meta_audio_file.metadata.pitch,
            "velocity": meta_audio_file.metadata.velocity,
            "qualities": list(meta_audio_file.metadata.qualities),
            "family": meta_audio_file.metadata.family,
            "source": meta_audio_file.metadata.source,
        }

        embeddings = np.frombuffer(meta_audio_file.encoder_outputs.embeddings.data, dtype=np.float32).copy()
        embeddings = embeddings.reshape(meta_audio_file.encoder_outputs.embeddings.shape)

        datapoint = {
            "audio_data": audio_data,
            "metadata": metadata,
            "embeddings": embeddings
        }
        return datapoint
    
class FilterPitchSampler(Sampler):
    def __init__(self, dataset: MetaAudioDataset, pitch: Sequence[int]):
        self.dataset = dataset
        self.pitch = pitch
        self.indices = [i for i, data in enumerate(dataset) if data["metadata"]["pitch"] in pitch]

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
class BalancedFamilySampler(Sampler):
    def __init__(self, dataset: MetaAudioDataset, pitch: Sequence[int]):
        self.dataset = dataset
        self.pitch = pitch
        self.family_indices = self._get_family_indices()

    def _get_family_indices(self):
        family_indices = {}
        for i, data in enumerate(self.dataset):
            if data["metadata"]["pitch"] in self.pitch:
                family = data["metadata"]["family"]
                if family not in family_indices:
                    family_indices[family] = []
                family_indices[family].append(i)
        return family_indices

    def __iter__(self):
        min_count = min(len(indices) for indices in self.family_indices.values())
        balanced_indices = []
        for indices in self.family_indices.values():
            balanced_indices.extend(np.random.choice(indices, min_count, replace=False))
        np.random.shuffle(balanced_indices)
        return iter(balanced_indices)

    def __len__(self):
        return min(len(indices) for indices in self.family_indices.values())