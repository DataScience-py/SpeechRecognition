# create handler work with model
# input audio file, output text  or text in file
# stream audio output and return text output
import logging
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import sounddevice as sd
import numpy as np

class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        <SPACE> 0
        а 1
        б 2
        в 3
        г 4
        д 5
        е 6
        ё 7
        ж 8
        з 9
        и 10
        й 11
        к 12
        л 13
        м 14
        н 15
        о 16
        п 17
        р 18
        с 19
        т 20
        у 21
        ф 22
        х 23
        ц 24
        ч 25
        ш 26
        щ 27
        ъ 28
        ы 29
        ь 30
        э 31
        ю 32
        я 33
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """Use a character map and convert integer labels to an text sequence"""
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("<SPACE>", " ")


text_transform = TextTransform()

class AudioProcessor:
    def __init__(self, sample_rate=16000, duration=4, transform=None, log_file='audio_processor.log', model_file_path='model//model_scripted.pt'):
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.audio_transform = torchaudio.transforms.MelSpectrogram()
            
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Для вывода в консоль
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.model = torch.jit.load(model_file_path)
        
    def recognize_from_file(self, file_path):
        self.logger.info(f"Loading audio file: {file_path}")
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Если частота дискретизации не совпадает, изменяем ее
        if sample_rate != self.sample_rate:
            self.logger.info(f"Resampling audio from {sample_rate} to {self.sample_rate}")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        audio = waveform.flatten()
        mel_spectrogram = self.process_audio(audio)
        output = self.send_to_model(mel_spectrogram)
        return output

    def record_audio(self):
        self.logger.info("Recording audio...")
        audio = sd.rec(int(self.sample_rate * self.duration), samplerate=self.sample_rate, channels=1)
        sd.wait()  # Wait until recording is finished
        self.logger.info("Recording finished.")
        return audio.flatten()

    def process_audio(self, audio):
        audio_tensor = torch.tensor(audio)
        if self.transform:
            mel_spectrogram = self.transform(audio_tensor)
        else:
            spec = self.audio_transform(audio_tensor).squeeze(0).transpose(0, 1)
            mel_spectrogram = spec.unsqueeze(0).unsqueeze(1).transpose(2, 3)
        mel_spectrogram = mel_spectrogram.to(next(self.model.parameters()).device)
        self.logger.info("Audio processed into Mel Spectrogram.")
        return mel_spectrogram
    
    def decode_output(self, output, blank_label=34, collapse_repeated=True):
        self.logger.info("Decoding output...")
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        for i, args in enumerate(arg_maxes):
            decode = []
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(text_transform.int_to_text(decode))
        self.logger.info("Output decoded.")
        return ''.join(decodes)

    def send_to_model(self, mel_spectrogram):
        self.logger.info("Sending to model...")
        output = self.model(mel_spectrogram)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        self.logger.info("Data sent to model.")
        output = self.decode_output(output)
        self.logger.info("Data processing finished.")
        return output

    def run(self):
        audio = self.record_audio()
        mel_spectrogram = self.process_audio(audio)
        output = self.send_to_model(mel_spectrogram)
        return output
    
    
audio_listen = AudioProcessor()