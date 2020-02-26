#!/usr/bin/env python
# coding: utf-8
# https://github.com/LearnedVector/Wav2Letter/blob/master/Google%20Speech%20Command%20Example.ipynb


import collections
import cProfile
import hashlib
import itertools
import os
import pstats
import re
from datetime import datetime
from io import StringIO

from tqdm import tqdm

import torch
import torchaudio
from torch import nn, topk
from torch.optim import Adadelta
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MFCC

audio_backend = "soundfile"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 0
pin_memory = False
non_blocking = pin_memory

labels = [
        "-",
        "*",
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "go",
        "happy",
        "house",
        "learn",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "visual",
        "wow",
        "yes",
        "zero",
]
shuffle = False
drop_last = True

# audio, self.sr, window_stride=(160, 80), fft_size=512, num_filt=20, num_coeffs=13
n_mfcc = 13
melkwargs = {
    'n_fft': 512,
    'n_mels': 20,
    'hop_length': 80,
}
sample_rate = 16000

batch_size = 512  # max number of sentences per batch
optimizer_params = {
    "lr": 1.0,
    "eps": 1e-8,
    "rho": 0.95,
}

hidden_size = 8
num_layers = 1

max_epoch = 80
clip_norm = 0.

training_percentage = 10.
validation_percentage = 5.
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

dtstamp = datetime.now().strftime("%y%m%d.%H%M%S")


# Profiling performance
pr = cProfile.Profile()
pr.enable()

torchaudio.set_audio_backend(audio_backend)
mfcc = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)
mfcc.to(device)


class Coder:
    def __init__(self, labels):
        self.max_length = max(map(len, labels))

        labels = list(collections.OrderedDict.fromkeys(list("".join(labels))))
        self.length = len(labels)
        enumerated = list(enumerate(labels))
        flipped = [(sub[1], sub[0]) for sub in enumerated]

        d1 = collections.OrderedDict(enumerated)
        d2 = collections.OrderedDict(flipped)
        self.mapping = {**d1, **d2}

    def _map_and_pad(self, iterable, fillwith):
        iterable = [self.mapping[i] for i in iterable]  # map with dict
        iterable += [fillwith] * (self.max_length-len(iterable))  # add padding
        return iterable

    def encode(self, iterable):
        if isinstance(iterable[0], list):
            return [self.encode(i) for i in iterable]
        else:
            return self._map_and_pad(iterable, self.mapping["*"])

    def decode(self, tensor):
        if isinstance(tensor[0], list):
            return [self.decode(t) for t in tensor]
        else:
            return "".join(self._map_and_pad(tensor, self.mapping[1]))


coder = Coder(labels)
encode = coder.encode
decode = coder.decode
vocab_size = coder.length


# @torch.jit.script
def process_datapoint(item):
    transformed = item[0].to(device, non_blocking=non_blocking)
    target = item[2]
    # pick first channel, apply mfcc, tranpose for pad_sequence
    transformed = mfcc(transformed)
    transformed = transformed[0, ...].transpose(0, -1)
    target = encode(target)
    target = torch.tensor(target, dtype=torch.long, device=transformed.device)
    return transformed, target


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
        filename: File path of the data sample.
        validation_percentage: How much of the data set to use for validation.
        testing_percentage: How much of the data set to use for testing.

    Returns:
        String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name).encode("utf-8")
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


class FILTERED_SPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(self, tag, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training_percentage < 100.:
            testing_percentage = (100. - training_percentage - validation_percentage)
            self._walker = list(filter(lambda x: which_set(x, validation_percentage, testing_percentage) == tag, self._walker))


class PROCESSED_SPEECHCOMMANDS(FILTERED_SPEECHCOMMANDS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, n):
        item = super().__getitem__(n)
        return process_datapoint(item)

    def __next__(self):
        item = super().__next__()
        return process_datapoint(item)


class MemoryCache(torch.utils.data.Dataset):
    """
    Wrap a dataset so that, whenever a new item is returned, it is saved to memory.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._id = id(self)
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n]:
            return self._cache[n]

        item = self.dataset[n]
        self._cache[n] = item

        return item

    def __len__(self):
        return len(self.dataset)


def datasets():
    root = "./"

    training = PROCESSED_SPEECHCOMMANDS("training", root, download=True)
    training = MemoryCache(training)
    validation = PROCESSED_SPEECHCOMMANDS("validation", root, download=True)
    validation = MemoryCache(validation)
    testing = PROCESSED_SPEECHCOMMANDS("testing", root, download=True)
    testing = MemoryCache(testing)

    return training, validation, testing


def collate_fn(batch):

    tensors = [b[0] for b in batch if b]
    targets = [b[1] for b in batch if b]

    input_lengths = [t.shape[0] for t in tensors]
    target_lengths = [len(t) for t in targets]

    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors = tensors.transpose(1, -1)

    return tensors, targets, input_lengths, target_lengths


class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x)
        return x


class Wav2Letter(nn.Module):
    """Wav2Letter Speech Recognition model
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or power spectrums speech signals

        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, num_features, num_classes):
        super().__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride)
        self.layers = nn.Sequential(
            # PrintLayer(),
            nn.Conv1d(num_features, 250, 48, 2),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
            # nn.Conv1d(250, 250, 7),
            # nn.ReLU(),
            # nn.Conv1d(250, 250, 7),
            # nn.ReLU(),
            nn.Conv1d(250, 2000, 32),
            nn.ReLU(),
            nn.Conv1d(2000, 2000, 1),
            nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )

    def forward(self, batch):
        """Forward pass through Wav2Letter network than
            takes log probability of output
        Args:
            batch (int): mini batch of data
            shape (batch, num_features, frame_len)
        Returns:
            Tensor with shape (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(batch)

        # compute log softmax probability on graphemes
        log_probs = nn.functional.log_softmax(y_pred, dim=1)
        log_probs = log_probs.transpose(1, 2).transpose(0, 1)

        # print(log_probs.shape)
        return log_probs


class BiLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.directions = 2
        # self.layers = nn.GRU(num_features, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        # self.layers = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        # https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967
        # self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.hidden2class = nn.Linear(self.directions*hidden_size, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (torch.autograd.Variable(torch.zeros(self.directions*num_layers, batch_size, hidden_size)).to(device),
                torch.autograd.Variable(torch.zeros(self.directions*num_layers, batch_size, hidden_size)).to(device))

    def forward(self, inputs):
        inputs = inputs.transpose(-1, -2)
        # outputs, _ = self.layers(inputs)
        # print(inputs.shape)
        outputs, self.hidden = self.lstm(inputs, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        # outputs = outputs.view(batch_size, 2*hidden_size, -1)
        outputs = self.hidden2class(outputs)

        log_probs = nn.functional.log_softmax(outputs, dim=1)
        log_probs = log_probs.transpose(0, 1)
        return log_probs


def forward(inputs, targets):

    inputs = inputs.to(device, non_blocking=non_blocking)
    targets = targets.to(device, non_blocking=non_blocking)
    outputs = model(inputs)

    this_batch_size = len(inputs)
    input_lengths = torch.full(
        (this_batch_size,), outputs.shape[0], dtype=torch.long, device=outputs.device
    )
    target_lengths = torch.tensor(
        [target.shape[0] for target in targets], dtype=torch.long, device=targets.device
    )

    # CTC
    # https://pytorch.org/docs/master/nn.html#torch.nn.CTCLoss
    # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
    # outputs: input length, batch size, number of classes (including blank)
    # targets: batch size, max target length
    # input_lengths: batch size
    # target_lengths: batch size
    return criterion(outputs, targets, input_lengths, target_lengths)


def greedy_decoder(outputs):
    """Greedy Decoder. Returns highest probability of class labels for each timestep

    Args:
        outputs (torch.Tensor): shape (input length, batch size, number of classes (including blank))

    Returns:
        torch.Tensor: class labels per time step.
    """
    _, indices = topk(outputs, k=1, dim=-1)
    return indices[..., 0]


loader_training = DataLoader(
    training, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, drop_last=drop_last,
    num_workers=num_workers, pin_memory=pin_memory,
)
loader_validation = DataLoader(
    validation, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, drop_last=drop_last,
    num_workers=num_workers, pin_memory=pin_memory,
)

model = Wav2Letter(n_mfcc, vocab_size)
# model = BiLSTM(1, vocab_size)
# model = BiLSTM(n_mfcc, vocab_size)

# model = torch.jit.script(model)
model = model.to(device, non_blocking=non_blocking)

optimizer = Adadelta(model.parameters(), **optimizer_params)
criterion = torch.nn.CTCLoss()

best_loss = 1.

for epoch in range(max_epoch):

    model.train()

    sum_loss = 0.
    for inputs, targets, _, _ in tqdm(loader_training):

        loss = forward(inputs, targets)
        sum_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    # Average loss
    sum_loss_training = sum_loss / len(loader_training)

    with torch.no_grad():

        model.eval()

        sum_loss = 0.
        for inputs, targets, _, _ in loader_validation:

            loss = forward(inputs, targets)
            sum_loss += loss.item()

        # Average loss
        sum_loss_validation = sum_loss / len(loader_validation)

    print(f"{epoch}: {sum_loss_training:.5f}, {sum_loss_validation:.5f}")

    if (loss < best_loss).all():
        # Save model
        torch.save(model.state_dict(), f"./model.{dtstamp}.{epoch}.ph")
        best_loss = sum_loss


# Save model
torch.save(model.state_dict(), f"./model.{dtstamp}.{epoch}.ph")

# Switch to evaluation mode
model.eval()

print(targets[0])
print(decode(targets.tolist()[0]))

output = model(inputs)[:, 0, :]
output = greedy_decoder(output)

print(output)
print(decode(output.tolist()))

# Print performance
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(20)
print(s.getvalue())
