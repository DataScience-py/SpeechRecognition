# Training Script for Speech Recognition Model

This script is designed to train a speech recognition model using audio data. It utilizes PyTorch for model training and evaluation.

## Requirements

- `model.py`: Contains the `SpeechRecognitionModel` class.

- `preproc.py`: Contains preprocessing functions and classes.

- `numpy`: For numerical operations.

- `torch`: For building and training the neural network.

- `torch.optim`: For optimization algorithms.

- `torch.utils.data`: For data loading utilities.

## Installation

To install the required packages, run:

```bash
pip install numpy torch
```

or

```bash
pip install -r requirements.txt
```

Functions

avg_wer(wer_scores, combined_ref_len)

Calculates the average word error rate (WER).

Parameters:

wer_scores: List of WER scores.

combined_ref_len: Total length of reference words.

Returns: Average WER as a float.

_levenshtein_distance(ref, hyp)

Calculates the Levenshtein distance between two sequences.

Parameters:

ref: Reference sequence (string).

hyp: Hypothesis sequence (string).

Returns: Levenshtein distance as an integer.

word_errors(reference, hypothesis, ignore_case=False, delimiter=" ")

Computes the word-level Levenshtein distance between reference and hypothesis.

Parameters:

reference: The reference sentence (string).

hypothesis: The hypothesis sentence (string).

ignore_case: Whether to ignore case (boolean).

delimiter: Delimiter for splitting sentences (string).

Returns: Tuple of edit distance and length of reference words.

char_errors(reference, hypothesis, ignore_case=False, remove_space=False)

Computes the character-level Levenshtein distance between reference and hypothesis.

Parameters:

reference: The reference sentence (string).

hypothesis: The hypothesis sentence (string).

ignore_case: Whether to ignore case (boolean).

remove_space: Whether to remove internal space characters (boolean).

Returns: Tuple of edit distance and length of reference characters.

wer(reference, hypothesis, ignore_case=False, delimiter=" ")

Calculates the word error rate (WER).

Parameters:

reference: The reference sentence (string).

hypothesis: The hypothesis sentence (string).

ignore_case: Whether to ignore case (boolean).

delimiter: Delimiter for splitting sentences (string).

Returns: WER as a float.

cer(reference, hypothesis, ignore_case=False, remove_space=False)

Calculates the character error rate (CER).

Parameters:

reference: The reference sentence (string).

hypothesis: The hypothesis sentence (string).

ignore_case: Whether to ignore case (boolean).

remove_space: Whether to remove internal space characters (boolean).

Returns: CER as a float.

IterMeter Class

A utility class to keep track of total iterations during training.

Methods:

step(): Increments the iteration count.

get(): Returns the current iteration count.

train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)

Trains the model for one epoch.

Parameters:

model: The speech recognition model.

device: The device to run the model on (CPU or GPU).

train_loader: DataLoader for training data.

criterion: Loss function.

optimizer: Optimizer for updating model weights.

scheduler: Learning rate scheduler.

epoch: Current epoch number.

iter_meter: Iteration meter for tracking progress.

test(model, device, test_loader, criterion, epoch, iter_meter)

Evaluates the model on the test dataset.

Parameters:

model: The speech recognition model.

device: The device to run the model on (CPU or GPU).

test_loader: DataLoader for test data.

criterion: Loss function.

epoch: Current epoch number.

iter_meter: Iteration meter for tracking progress.

main(learning_rate=5e-4, batch_size=20, epochs=10, train_dataset=None, test_dataset=None)

Main function to set up and run the training and testing process.

Parameters:

learning_rate: Learning rate for the optimizer (float).

batch_size: Batch size for training (int).

epochs: Number of training epochs (int).

train_dataset: Dataset for training (AudioDataset).

test_dataset: Dataset for testing (AudioDataset).

Returns: The trained model.
