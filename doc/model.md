# Model doc

Describes a model for speech recognition. You can change the model for your needs and use it in your project.

## requirements

torch.nn

torch.nn.functional

```bash
pip install torch
```

or

```bash
pip3 install -r requirements.txt
```

## Classes

### CNNLayerNorm

This layer is a combination of `nn.LayerNorm`

#### Methods

__init__()

Create a new instance of `CNNLayerNorm`
1 LayerNorm

input: n_feats

output: None

forward()

Apply layer norm to the input

input: x: torch.Tensor (batch, channel, feature, time)

layer norm (batch, channel, time, feature)

output: x: torch.Tensor (batch, chanel, feature, time)

### ResidualCNN

About this layer can read [here](https://arxiv.org/pdf/1603.05027.pdf)
includes CNNLayerNorm

#### Methods

__init__()

Create a new instance of `ResidualCNN`

input: in_channels, out_channels, kernel, stride, dropout, n_feats

output: None

forward()

Apply ResidualCNN to the input.

input: x: torch.Tensor (batch, channel, feature, time)

save x

2 CNNLayerNorm

2 function gelu

2 CNN

2 Dropout

output: x + save x: torch.Tensor (batch, chanel, feature, time)

### BidirectionalGRU

Recurrent neural network realize in GRU

#### Methods

__init__()

Create a new instance of `BidirectionalGRU`

input: rnn_dim, hidden_size, dropout, batch_first

output: None

forward()

Apply BidirectionalGRU to the input.

input: x: torch.Tensor (batch, channel, feature, time)

1 layer norm

1 function gelu

1 GRU layer

1 Dropout

output: x: torch.Tensor (batch, chanel, feature, time)

### SpeechRecognitionModel

Collects all parts of the model for speech recognition (ResidualCNN, BidirectionalGRU)

#### Methods

__init__()

Create a new instance of `SpeechRecognitionModel`

input:  n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1

output: None

forward()

Apply SpeechRecognitionModel to the input.

input: x: torch.Tensor (batch, channel, feature, time)

n_cnn_layers ResidualCNN

view x (batch_size, size[1] * size[2], size[3])

1 Linear layer

n_rnn_layers  BidirectionalGRU

1 classifier (Linear layer, gelu, dropout, linear layer)
