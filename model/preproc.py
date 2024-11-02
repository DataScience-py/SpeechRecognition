import torch
import torchaudio
from sympy.utilities.iterables import signed_permutations
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd

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
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()


text_transform = TextTransform()

def data_processing_valid(data, data_type="valid"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, utterance) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def data_processing_train(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, utterance) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def GreedyDecoder(output, labels, label_lengths, blank_label=34, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


def normalization_text(text: str) -> str:
        """
        Normalizes the text by removing all non-alphabetic characters and converting it to lowercase.
        :param text: The text to be normalized.
        :return: The normalized text.
        """
        return ''.join(char for char in text.lower() if char in ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

class AudioDataset(Dataset):
    def __init__(self, data_path_csv: str, data_path_audio: str, sep='\t'):
        self.data_path_audio = data_path_audio
        self.data_path_csv = data_path_csv
        self.data = pd.read_csv(data_path_csv, sep=sep, usecols=['path', 'sentence'])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data_path_audio  + self.data.iloc[idx, 0]
        sentence = self.data.iloc[idx, 1]
        sentence = normalization_text(sentence)
        audio = torchaudio.load(path)[0]
        return audio, sentence
