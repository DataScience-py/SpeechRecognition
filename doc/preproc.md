# Preprocessing doc

Processing of audio files and transformation for learning speech recognition.

## requirements

torch

torchaudio

pandas

```bash
pip install torch torchaudio pandas
```

or

```bash
pip3 install torch torchaudio pandas
```

## Classes

### TextTransform

Coding of the text and decoding by symbols.

#### Methods

__init__()

create dictionary of symbols and integers.

input: None

output: None

text_to_int()

transform string to list of integers.

input: text: str

output: int_sequence: List[int]

translate symbols to integers.

int_to_text()

transform list of integers to string.

input: labels: List[int]

output: string: str

### AudioDataset(Dataset)

#### Methods

__init__()

create dataset from audio files.

input: data_path_csv: str, data_path_audio: str, sep='/t'

output: None

__len__()

length of the dataset.

input: None

output: int: length of the dataset

__getitem__()

get data from the dataset.

input: index: int

output: data: torch.Tensor, labels: torch.Tensor, input_lengths: torch.Tensor, output_lengths: torch.Tensor

## Functions

### load audio

#### train data transforms

sample rate = 16000

n_mels = 128

freq_msk_param = 30

time_mask_param = 100

torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),

torchaudio.transforms.FrequencyMasking(freq_mask_param=30),

torchaudio.transforms.TimeMasking(time_mask_param=100),

data_processing_train()

Audio processing in training

input: data: torch.Tensor

output: spectrogram: torch.Tensor, labels: torch.Tensor, input_lengths: torch.Tensor, output_lengths: torch.Tensor

#### valid data transforms

sample rate = 16000

n_mels = 128

torchaudio.transforms.MelSpectrogram(sample_rate, n_mels)

data_processing_valid()

Processing audio of validations

input: data: torch.Tensor

output: spectrogram: torch.Tensor, labels: torch.Tensor, input_lengths: torch.Tensor, output_lengths: torch.Tensor

### Decoder text

GreedyDecoder()

Decoding the output of the model and the correct answer

input: output, labels, label_lengths, blank_label=34, collapse_repeated=True

output: decodes, targets

### Normalization target

normalization_text()

clean text with other symbols.

input: text: str

output: text: str
