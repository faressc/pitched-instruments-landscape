from pathlib import Path
import subprocess
from typing import Sequence, Iterable, Tuple
import numpy as np

def load_audio_file(path: str, sample_rate: int) -> Tuple[np.ndarray, float, int]:
    
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
    audio = [process.stdout.read() for process in processes]
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {process.stderr.read()}")
    return np.frombuffer(b''.join(audio), dtype=np.int16), get_audio_duration(path), get_audio_channels(path)

def write_audio_file(data: Sequence[np.ndarray], path: str, sample_rate: int):
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
    if process.stdin is None:
        raise RuntimeError("Failed to open pipe to ffmpeg stdin")
    if process.stderr is None:
        raise RuntimeError("Failed to open pipe to ffmpeg stderr")
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