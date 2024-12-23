import os
import io
from pydub import AudioSegment
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_data(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    with open(os.path.join(in_dir, "speaker-info.txt"), encoding="utf-8") as f:
        next(f)  # skip header

        for line in tqdm(f):
            parts = line.strip().split()
            speaker = parts[0]
            # age = parts[1]
            # gender = parts[2]
            # accent = parts[3]
            # region = parts[4]
            # comment = parts[5] or ""

            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

            for base_name in os.listdir(os.path.join(in_dir, "txt", speaker)):
                with open(os.path.join(in_dir, "txt", speaker, base_name), 'r') as file:
                    raw_text = file.read()

                text = _clean_text(raw_text, cleaners)
                base_name = base_name.split(".")[0]

                text_path = os.path.join(in_dir, "txt", speaker, base_name)
                flac_path_mic1 = os.path.join(in_dir, "wav48_silence_trimmed", speaker,
                                              "{}_mic1.flac".format(base_name))
                flac_path_mic2 = os.path.join(in_dir, "wav48_silence_trimmed", speaker,
                                              "{}_mic2.flac".format(base_name))

                if os.path.exists(text_path):
                    with open(
                            os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                            "w",
                    ) as f1:
                        f1.write(text)

                if os.path.exists(flac_path_mic1):
                    flac1 = AudioSegment.from_file(flac_path_mic1, format="flac")
                    stream1 = io.BytesIO()
                    flac1.export(stream1, format="wav")
                    wav1, _ = librosa.load(stream1, sampling_rate)
                    wav1 = wav1 / max(abs(wav1)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}_mic1.wav".format(base_name)),
                        sampling_rate,
                        wav1.astype(np.int16),
                    )

                if os.path.exists(flac_path_mic2):
                    flac2 = AudioSegment.from_file(flac_path_mic2, format="flac")
                    stream2 = io.BytesIO()
                    flac2.export(stream2, format="wav")
                    wav2, _ = librosa.load(stream2, sampling_rate)
                    wav2 = wav2 / max(abs(wav2)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}_mic2.wav".format(base_name)),
                        sampling_rate,
                        wav2.astype(np.int16),
                    )
