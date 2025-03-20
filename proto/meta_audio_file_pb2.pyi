from typing import List, Optional, Union, Sequence
from enum import IntEnum
from google.protobuf.message import Message

class MetaAudioFile(Message):
    class DType(IntEnum):
        INT8 = 0
        INT16 = 1
        INT32 = 2
        INT64 = 3
        FLOAT32 = 4
        FLOAT64 = 5

    class AudioFile(Message):
        data: bytes
        sample_rate: int
        dtype: 'MetaAudioFile.DType'
        num_channels: int
        num_samples: int
        
        def __init__(
            self,
            *,
            data: Optional[bytes] = None,
            sample_rate: Optional[int] = None,
            dtype: Optional[Union[int, 'MetaAudioFile.DType']] = None,
            num_channels: Optional[int] = None,
            num_samples: Optional[int] = None,
        ) -> None: ...

    class Metadata(Message):
        class InstrumentSource(IntEnum):
            ACOUSTIC = 0
            ELECTRONIC = 1
            SYNTHETIC = 2

        class InstrumentFamily(IntEnum):
            BASS = 0
            BRASS = 1
            FLUTE = 2
            GUITAR = 3
            KEYBOARD = 4
            MALLET = 5
            ORGAN = 6
            REED = 7
            STRING = 8
            SYNTH_LEAD = 9
            VOCAL = 10
            
        note: int
        note_str: str
        instrument: int
        instrument_str: str
        pitch: int
        velocity: int
        qualities: List[int]
        family: 'MetaAudioFile.Metadata.InstrumentFamily'
        source: 'MetaAudioFile.Metadata.InstrumentSource'
        
        def __init__(
            self,
            *,
            note: Optional[int] = None,
            note_str: Optional[str] = None,
            instrument: Optional[int] = None,
            instrument_str: Optional[str] = None,
            pitch: Optional[int] = None,
            velocity: Optional[int] = None,
            qualities: Optional[Sequence[int]] = None,
            family: Optional[Union[int, 'MetaAudioFile.Metadata.InstrumentFamily']] = None,
            source: Optional[Union[int, 'MetaAudioFile.Metadata.InstrumentSource']] = None,
        ) -> None: ...

    class EncodecOutputs(Message):
        class AudioCodes(Message):
            data: bytes
            shape: List[int]
            dtype: 'MetaAudioFile.DType'
            
            def __init__(
                self,
                *,
                data: Optional[bytes] = None,
                shape: Optional[Sequence[int]] = None,
                dtype: Optional[Union[int, 'MetaAudioFile.DType']] = None,
            ) -> None: ...

        class Embeddings(Message):
            data: bytes
            shape: List[int]
            dtype: 'MetaAudioFile.DType'
            
            def __init__(
                self,
                *,
                data: Optional[bytes] = None,
                shape: Optional[Sequence[int]] = None,
                dtype: Optional[Union[int, 'MetaAudioFile.DType']] = None,
            ) -> None: ...
            
        audio_codes: 'MetaAudioFile.EncodecOutputs.AudioCodes'
        embeddings: 'MetaAudioFile.EncodecOutputs.Embeddings'
        
        def __init__(
            self,
            *,
            audio_codes: Optional['MetaAudioFile.EncodecOutputs.AudioCodes'] = None,
            embeddings: Optional['MetaAudioFile.EncodecOutputs.Embeddings'] = None,
        ) -> None: ...

    audio_file: 'MetaAudioFile.AudioFile'
    metadata: 'MetaAudioFile.Metadata'
    encoder_outputs: 'MetaAudioFile.EncodecOutputs'
    
    def __init__(
        self,
        *,
        audio_file: Optional['MetaAudioFile.AudioFile'] = None,
        metadata: Optional['MetaAudioFile.Metadata'] = None,
        encoder_outputs: Optional['MetaAudioFile.EncodecOutputs'] = None,
    ) -> None: ...

