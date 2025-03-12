import numpy as np
from utils import config
from pathlib import Path
from pedalboard.io import AudioFile
from transformers import EncodecModel, AutoProcessor
import librosa
from einops import rearrange

import utils.debug

def normalize(data):
    data_norm = max(max(data), abs(min(data)))
    return data / data_norm

def load_and_process_audio(file_path, target_sampling_rate):
    with AudioFile(file_path) as f:
        data = f.read(f.frames).flatten().astype(np.float32)
    print(f"Data loaded from {file_path}.")
    print(f"Resampling data from {f.samplerate} to {target_sampling_rate}.")
    data_resampled = librosa.resample(data, orig_sr=f.samplerate, target_sr=target_sampling_rate)
    return data_resampled

def write_wav(data, sampling_rate, file_path):
    with AudioFile(file_path, 'w', sampling_rate, 1) as f:
        f.write(data)

def main():
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params('params.yaml')

    # Load the parameters from the dictionary into variables
    input_file = params['preprocess']['input_file']

    audio_sample = load_and_process_audio(input_file, target_sampling_rate=24000)
    audio_sample = audio_sample[:24000*10]
    write_wav(audio_sample, 24000, "test.wav")

    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    print("The processor sampling rate is: ", processor.sampling_rate)
    
    # pre-process the inputs
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

    encoder_outputs = model.encode(inputs['input_values'], bandwidth=24.0)

    # The model quantizer expects the audio codes to be of shape (quantizer_dim, batch_size, time_steps)
    audio_codes = encoder_outputs.audio_codes.view(1, encoder_outputs.audio_codes.size(2), encoder_outputs.audio_codes.size(3))
    audio_codes = rearrange(audio_codes, 'b q t -> q b t')
    embeddings = model.quantizer.decode(audio_codes)
    audio_values = model.decoder(embeddings)
    write_wav(audio_values[0].detach().numpy(), 24000, "test_reconstructed_from_embedding.wav")

    # Feedforward the audio codes to the decoder to get the audio values
    audio_values_direct = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales).audio_values
    write_wav(audio_values_direct[0].detach().numpy(), 24000, "test_reconstructed_directly.wav")
    exit()

    # torch.save({
    #     'X_ordered_training': X_ordered_training,
    #     'y_ordered_training': y_ordered_training,
    #     'X_ordered_testing': X_ordered_testing,
    #     'y_ordered_testing': y_ordered_testing
    # }, output_file_path)
    # print("Preprocessing done and data saved.")

if __name__ == "__main__":
    main()