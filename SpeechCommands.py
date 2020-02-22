#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cProfile, pstats
from io import StringIO

pr = cProfile.Profile()
pr.enable()


# [example](https://github.com/LearnedVector/Wav2Letter/blob/master/Google%20Speech%20Command%20Example.ipynb)

# In[3]:


import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS


# In[4]:


torchaudio.set_audio_backend("soundfile")
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[22]:


labels = [
        '-', '*',
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

import collections


def build_mapping(labels):
    labels = list(collections.OrderedDict.fromkeys(list("".join(labels))))
    enumerated = list(enumerate(labels))
    flipped = [(sub[1], sub[0]) for sub in enumerated]

    d1 = collections.OrderedDict(enumerated)
    d2 = collections.OrderedDict(flipped)
    return {**d1, **d2}

def padding(l, max_length, fillwith):
    return l  + [fillwith] * (max_length-len(l))

def map_with_dict(mapping, l):
    return [mapping[t] for t in l]

def apply_with_padding(l, mapping, max_length, fillwith):
    l = map_with_dict(mapping, l)
    l = padding(l, max_length, mapping["*"])
    return l


test = "house"
max_length = max(map(len, labels))
vocab_size = len(labels) + 2

mapping = build_mapping(labels)

# test = apply(mapping, test)
# test = padding(test, max_length, mapping["*"])

encode = lambda l: apply_with_padding(l, mapping, max_length, mapping["*"])
decode = lambda l: apply_with_padding(l, mapping, max_length, mapping[1])

decode(encode(test))


# In[6]:


from torchaudio.transforms import MFCC

num_features = 13

melkwargs = {
    'n_fft': 512,
    'n_mels': 20,
    'hop_length': 80,
}

mfcc = MFCC(sample_rate=16000, n_mfcc=num_features, melkwargs=melkwargs)

# audio, self.sr, window_stride=(160, 80),
# fft_size=512, num_filt=20, num_coeffs=13

def process_waveform(waveform):
    # pick first channel, apply mfcc, tranpose for pad_sequence
    return mfcc(waveform)[0, ...].transpose(0, -1)

def process_target(target):

    # targets = []
    # for b in batch:
    #     if b:
    #         token = sp.encode_as_pieces(b[2])
    #         print(len(token))
    #         token = " ".join(token)
    #         targets.append(token)

    # return " ".join(sp.encode_as_ids(target))
    
    # return torch.IntTensor(sp.encode_as_ids(target))
    # print(target)
    # return torch.IntTensor(encode(target))
    return torch.tensor(encode(target), dtype=torch.long, device=device)


# In[7]:


class PROCESSED_SPEECHCOMMANDS(SPEECHCOMMANDS):
    def __getitem__(self, n):
        return self._process(super().__getitem__(n))
        
    def _process(self, item):
        # waveform, sample_rate, label, speaker_id, utterance_number
        waveform = process_waveform(item[0])
        label = process_target(item[2])
        return waveform, label

    def __next__(self):
        return self._process(super().__next__())
    


# In[8]:


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


# In[9]:


#     waveform, sample_rate, label, speaker_id, utterance_number

def datasets():

    download = True
    root = "./"

    dataset = PROCESSED_SPEECHCOMMANDS(root, download=download)
    dataset = MemoryCache(dataset)

    return dataset


# In[10]:


train = datasets()


# In[11]:


from torch.utils.data import DataLoader
from random import randint



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


batch_size = 512  # max number of sentences per batch
loader_train = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


# In[12]:


#     waveform, sample_rate, label, speaker_id, utterance_number

def datasets():

    download = True
    root = "./"

    dataset = SPEECHCOMMANDS(root, download=download)

    return dataset

train = datasets()


# In[13]:


from torch import nn


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x
    
    

class Wav2Letter(nn.Module):
    """Wav2Letter Speech Recognition model
        Architecture is based off of Facebooks AI Research paper
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or
        power spectrums speech signals
        TODO: use cuda if available
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
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(batch)

        # compute log softmax probability on graphemes
        log_probs = nn.functional.log_softmax(y_pred, dim=1)

        return log_probs


model = Wav2Letter(num_features, vocab_size)


# In[14]:


import torchaudio
from torch.optim import Adadelta

model = Wav2Letter(num_features, vocab_size)

model = model.to(device)

optimizer_params = {
    "lr": 1.0,
    "eps": 1e-8,
    "rho": 0.95,
}
optimizer = Adadelta(model.parameters(), **optimizer_params)

max_epoch = 10
clip_norm = 10.

criterion = torch.nn.CTCLoss()

from tqdm import tqdm

max_files = 1
for epoch in range(max_epoch):
    # print(epoch)
    
    i_files = 0
    for inputs, targets, _, _ in tqdm(loader_train):

        # if i_files > max_files:
        #     break

        inputs = inputs.to(device)
        targets = targets.to(device)

        # print(i_files, max_files)

        if inputs is None or targets is None:
            continue

        # print("input", inputs.shape)
        outputs = model(inputs)
        # (input length, batch size, number of classes)
        # input_lengths = [len(o) for o in outputs]

        outputs = outputs.transpose(1, 2).transpose(0, 1)
        # print("output", outputs.shape)
        # print("target", targets.shape)
        
        # print(inputs.shape)
        # print(outputs.shape)
        # print(targets.shape)
        # print(len(targets))
        # print(targets.shape)
        # print(input_lengths)
        # input_lengths = [len(o) for o in outputs]
        # print(len(input_lengths))
        # target_lengths = [len(t) for t in targets]
        # print(target_lengths)
        # ctc_loss(input, target, input_lengths, target_lengths)

        # input_lengths = [outputs.shape[0]] * outputs.shape[1]
        
        # CTC arguments
        # https://pytorch.org/docs/master/nn.html#torch.nn.CTCLoss
        # better definitions for ctc arguments
        # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
        mini_batch_size = len(inputs)
        
        input_lengths = torch.full((mini_batch_size,), outputs.shape[0], dtype=torch.long, device=outputs.device)
        target_lengths = torch.tensor([target.shape[0] for target in targets], dtype=torch.long, device=targets.device)
        
        # print(torch.isnan(outputs).any())
        # print(torch.isnan(targets).any())
        # print(torch.isnan(input_lengths).any())
        # print(torch.isnan(target_lengths).any())
        # print(outputs.shape)
        # print(targets.shape)
        # print(input_lengths.shape)
        # print(target_lengths.shape)

        # outputs: input length, batch size, number of classes (including blank) 
        # targets: batch size, max target length
        # input_lengths: batch size
        # target_lengths: batch size
        loss = criterion(outputs, targets, input_lengths, target_lengths)

        # print("stepping")
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        
        i_files += 1
    
    print(epoch, loss)


# In[17]:


from torch import topk

def GreedyDecoder(outputs):
    """Greedy Decoder. Returns highest probability of
        class labels for each timestep

    Args:
        outputs (torch.Tensor): 
            shape (1, num_classes, output_len)
    
    Returns:
        torch.Tensor: class labels per time step.
    """
    _, indices = topk(ctc_matrix, k=1, dim=1)
    return indices[:, 0, :]


# In[29]:


sample = inputs[0].unsqueeze(0).to(device)
target = targets[0].to(device)

print(decode(targets[0].tolist()))

output = model(sample)
print(output.shape)

greedy_output = GreedyDecoder(output)

print(greedy_output.shape)
print(greedy_output)
print(decode(greedy_output.tolist()[0]))


# In[15]:


pr.disable()
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(20)
print(s.getvalue())
