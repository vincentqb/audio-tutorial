#!/usr/bin/env python
# coding: utf-8
# https://github.com/LearnedVector/Wav2Letter/blob/master/Google%20Speech%20Command%20Example.ipynb


import collections
import cProfile
import pstats
from io import StringIO
from datetime import datetime

import torch
from torch import nn, topk
from torch.optim import Adadelta
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchaudio
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
vocab_size = len(labels) + 2

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

max_epoch = 80
clip_norm = 0.

dtstamp = datetime.now().strftime("%y%m%d.%H%M%S")

torchaudio.set_audio_backend(audio_backend)
mfcc = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs)


class Coder:
    def __init__(self, labels):
        self.max_length = max(map(len, labels))

        labels = list(collections.OrderedDict.fromkeys(list("".join(labels))))
        enumerated = list(enumerate(labels))
        flipped = [(sub[1], sub[0]) for sub in enumerated]

        d1 = collections.OrderedDict(enumerated)
        d2 = collections.OrderedDict(flipped)
        self.mapping = {**d1, **d2}

    def _map_and_pad(self, iterable, fillwith):
        iterable = [self.mapping[i] for i in iterable]  # map with dict
        iterable += [fillwith] * (self.max_length-len(iterable))  # add padding
        return iterable

    def encode(self, iterable, device):
        iterable = self._map_and_pad(iterable, self.mapping["*"])
        return torch.tensor(iterable, dtype=torch.long, device=device)

    def decode(self, tensor):
        tensor = tensor.tolist() if hasattr(tensor, "tolist") else tensor
        return self._map_and_pad(tensor, self.mapping[1])


coder = Coder(labels)
encode = coder.encode
decode = coder.decode


def process_datapoint(item):
    waveform = item[0]
    target = item[2]
    # pick first channel, apply mfcc, tranpose for pad_sequence
    specgram = mfcc(waveform)[0, ...].transpose(0, -1)
    target = encode(target, device=specgram.device)
    return specgram, target


class PROCESSED_SPEECHCOMMANDS(SPEECHCOMMANDS):
    def __getitem__(self, n):
        return process_datapoint(super().__getitem__(n))

    def __next__(self):
        return process_datapoint(super().__next__())


class MemoryCache(torch.utils.data.Dataset):
    """
    Wrap a dataset so that, whenever a new item is returned, it is saved to disk.
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

    dataset = PROCESSED_SPEECHCOMMANDS(root, download=True)
    dataset = MemoryCache(dataset)
    # dataset = SPEECHCOMMANDS(root, download=download)

    return dataset


def collate_fn(batch):

    tensors = [b[0] for b in batch if b]
    targets = [b[1] for b in batch if b]

    # tensors = [process_waveform(b[0]) for b in batch if b]
    # targets = [process_target(b[2]) for b in batch if b]

    # truncate tensor list
    # length = 2**10
    # a = max(0, min([tensor.shape[-1] for tensor in tensors]) - length)
    # m = randint(0, a)
    # n = m + length
    # tensors = [t[..., m:n] for t in tensors]

    input_lengths = [t.shape[0] for t in tensors]
    target_lengths = [len(t) for t in targets]

    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors = tensors.transpose(1, -1)

    return tensors, targets, input_lengths, target_lengths


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

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
        super(Wav2Letter, self).__init__()

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
            # nn.Conv1d(250, 250, 7),
            # nn.ReLU(),
            # nn.Conv1d(250, 250, 7),
            # nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.ReLU(),
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

        return log_probs


def greedy_decoder(outputs):
    """Greedy Decoder. Returns highest probability of class labels for each timestep

    Args:
        outputs (torch.Tensor): shape (1, num_classes, output_len)

    Returns:
        torch.Tensor: class labels per time step.
    """
    _, indices = topk(outputs, k=1, dim=1)
    return indices[:, 0, :]


train = datasets()
loader_train = DataLoader(
    train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
    num_workers=num_workers, pin_memory=pin_memory,
)

model = Wav2Letter(n_mfcc, vocab_size)
model = torch.jit.script(model)
model = model.to(device, non_blocking=non_blocking)

optimizer = Adadelta(model.parameters(), **optimizer_params)
criterion = torch.nn.CTCLoss()

# Profiling performance
pr = cProfile.Profile()
pr.enable()

best_loss = 1.

for epoch in range(max_epoch):

    for inputs, targets, _, _ in tqdm(loader_train):

        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        outputs = model(inputs)
        outputs = outputs.transpose(1, 2).transpose(0, 1)

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
        loss = criterion(outputs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    print(epoch, loss)

    if (loss < best_loss).all():
        # Save model
        torch.save(model.state_dict(), f"./model.{dtstamp}.{epoch}.ph")
        best_loss = loss


# Save model
torch.save(model.state_dict(), f"./model.{dtstamp}.{epoch}.ph")

# Switch to evaluation mode
model.eval()

sample = inputs[0].unsqueeze(0).to(device, non_blocking=non_blocking)
target = targets[0].to(device, non_blocking=non_blocking)

print(targets[0])
print(decode(targets[0]))

output = model(sample)
output = greedy_decoder(output)

print(output)
print(decode(output[0]))

# Print performance
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(20)
print(s.getvalue())
