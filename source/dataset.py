import os
from typing import Sequence, Optional

import lmdb
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import numpy as np
from tqdm import tqdm
try:
    from proto.meta_audio_file_pb2 import MetaAudioFile
except:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from proto.meta_audio_file_pb2 import MetaAudioFile

class MetaAudioDataset(Dataset):

    def __init__(self, db_path: str, max_num_samples: int = -1, has_audio: bool = True):
        super().__init__()
        self._db_path = db_path
        self._env = None
        self._keys = None
        self._max_num_samples = max_num_samples
        self._has_audio = has_audio

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, readonly=True, lock=False, writemap=True)
        return self._env
    
    @property
    def keys(self) -> Sequence[str]:
        if self._keys is None:
            key_count = 0
            self._keys = []
            with self.env.begin() as txn:
                for key, _ in tqdm(txn.cursor(), total=txn.stat()['entries'], desc="Loading keys", unit="key"):
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

        if self._has_audio:
            audio_data = np.frombuffer(meta_audio_file.audio_file.data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / (2**15 - 1)
            audio_data = audio_data.reshape(meta_audio_file.audio_file.num_channels, -1)
        else:
            audio_data = np.array([])

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

        embeddings = self.preprocess_embedding(embeddings)

        datapoint = {
            "audio_data": audio_data,
            "metadata": metadata,
            "embeddings": embeddings
        }
        return datapoint

    @staticmethod
    def preprocess_embedding(emb):
        return MetaAudioDataset.normalize_embedding(emb.squeeze().swapaxes(0,1))
    @staticmethod
    def normalize_embedding(emb):
        return (emb + 25.) / (33.5 + 25.)
    
    @staticmethod
    def denormalize_embedding(normalized_embedding):
        return normalized_embedding * (33.5 + 25.) - 25.


    
    
class CustomSampler(Sampler):
    def __init__(self, dataset: Dataset, pitch: Optional[Sequence[int]] = None, max_inst_per_family: int = -1, velocity: Optional[Sequence[int]] = None, shuffle: bool = False):
        self.dataset = dataset
        self.pitch = pitch
        self.max_inst_per_family = max_inst_per_family
        self.velocity = velocity
        self.families = []
        self.chosen_instruments = []
        self._indices = None
        self.shuffle = shuffle

    @property
    def indices(self):
        if self._indices is None:
            self._indices = []
            for i, data in tqdm(enumerate(self.dataset), total=len(self.dataset), desc="Building sampler indices"):
                add_index = True
                if self.pitch and data["metadata"]["pitch"] not in self.pitch:
                    add_index = False
                if self.velocity and data["metadata"]["velocity"] not in self.velocity:
                    add_index = False
                if self.max_inst_per_family > 0 and add_index:
                    family = data["metadata"]["family"]
                    if family not in self.families:
                        self.families.append(family)
                        self.chosen_instruments.append([])
                    family_index = self.families.index(family)
                    if len(self.chosen_instruments[family_index]) < self.max_inst_per_family:
                        self.chosen_instruments[family_index].append(data["metadata"]["instrument"])
                    elif data["metadata"]["instrument"] not in self.chosen_instruments[family_index]:
                        add_index = False 
                if add_index:
                    self._indices.append(i)
        return self._indices

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