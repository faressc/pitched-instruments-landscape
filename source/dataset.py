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

    def __init__(self, db_path: str, max_num_samples: int = -1):
        super().__init__()
        self._db_path = db_path
        self._env = None
        self._keys = None
        self._max_num_samples = max_num_samples

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, readonly=True, lock=False, writemap=True)
        return self._env
    
    @property
    def keys(self) -> Sequence[str]:
        key_count = 0
        if self._keys is None:
            self._keys = []
            with self.env.begin() as txn:
                for key, _ in txn.cursor():
                    self._keys.append(key)
                    if self._max_num_samples != -1:
                        key_count += 1
                        if key_count >= self._max_num_samples:
                            break
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
    def __init__(self, dataset: MetaAudioDataset, pitch: Sequence[int], shuffle: bool):
        self.dataset = dataset
        self.pitch = pitch
        self.indices = [i for i, data in enumerate(dataset) if data["metadata"]["pitch"] in pitch]
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
class SingleElementSampler(Sampler):
    """Sampler that returns a single element from the dataset."""
    
    def __init__(self, dataset, index=0):
        """
        Args:
            dataset: Dataset to sample from
            index: The index of the element to sample
        """
        self.dataset = dataset
        self.index = index
        
    def __iter__(self):
        yield self.index
        
    def __len__(self):
        return 1

    
    
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