from pathlib import Path
import subprocess
from typing import Sequence, Iterable
import numpy as np

def search_for_audios(path: str, extensions: Sequence[str]) -> Iterable[Path]:
    path = Path(path)
    audios = []
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")
    for ext in extensions:
        audios.append(path.rglob(f'*.{ext}'))
        audios.append(path.rglob(f'*.{ext.upper()}'))
    # rglob returns a generator list of lists, so we need to flatten it
    for audio in audios:
        for a in audio:
            yield a

def load_audio_file(path: str, sample_rate: int) -> tuple[Sequence[np.ndarray], float, int]:
    
    num_channels = get_audio_channels(path)

    processes = []
    for i in range(num_channels): 
        process = subprocess.Popen(
            [
                'ffmpeg', '-loglevel', 'panic', '-i', path, '-f', 's16le', '-ar', str(sample_rate),
                 '-filter_complex', 'channelmap=%d-0'%i, '-'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(process)
    
    # Each sample is two bytes because of the s16le format
    audio_per_channel = []
    for process in processes:
        audio = process.stdout.read()
        audio = np.frombuffer(audio, dtype=np.int16)
        audio_per_channel.append(audio)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {process.stderr.read()}")
    return audio_per_channel, get_audio_duration(path), get_audio_channels(path)

def write_audio_file(data: Iterable[np.ndarray], path: str, sample_rate: int):
    num_channels = len(data)
    data_interleaved = np.vstack(data).reshape((-1,), order='F')
    process = subprocess.Popen(
        [
            'ffmpeg', '-y', '-f', 's16le', '-ar', str(sample_rate), '-ac', str(num_channels),
            '-i', '-', path
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.stdin.write(data_interleaved.tobytes())
    process.stdin.close()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {process.stderr.read()}")

def get_audio_channels(path: str) -> int:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 
            'stream=channels', '-of', 'default=noprint_wrappers=1:nokey=1', path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    return int(result.stdout.strip())

def get_audio_duration(path: str) -> float:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    return float(result.stdout.strip())