"""Implements a dataloader"""

from collections import OrderedDict

import abc
import datasets
import functools
import logging
import numpy as np
from typing import Callable, Dict, Mapping, List, Optional, Sequence
from DTI_JIALI.base import Alphabet
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
from typing import TypeVar, Optional, List
import torch.distributed as dist
import os
from DTI_JIALI.base import DTI, MetaAdapterConfig
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

class protein_encoder(object):
    """Callable to convert an unprocessed a protein to a
    processed tokens.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, protein: str):
        # RoBERTa uses an eos token, while ESM-1 does not.
        # batch_size = len(raw_batch)
        # seq_str_list = raw_batch
        protein = protein
        encoded_protein = self.alphabet.encode(protein)
        if self.truncation_seq_length:
            encoded_protein = encoded_protein[:self.truncation_seq_length]

        seq_len = len(encoded_protein)

        tokens = torch.empty(
            (1,
                seq_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
            ),
            dtype=torch.int64,
        )

        tokens.fill_(self.alphabet.padding_idx)
        if self.alphabet.prepend_bos:
            tokens[0] = self.alphabet.cls_idx

        seq = torch.tensor(encoded_protein, dtype=torch.int64)
        tokens[
        0,
        int(self.alphabet.prepend_bos): len(encoded_protein)
                                        + int(self.alphabet.prepend_bos),
        ] = seq
        if self.alphabet.append_eos:
            tokens[0, len(encoded_protein) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        tokens = tokens.squeeze(0)


        return tokens


class molecule_encoder:
    def __init__(self):
        self.char_to_idx = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8,
            '(': 9, ')': 10, '[': 11, ']': 12, '=': 13, '#': 14, '@': 15, '+': 16, '-': 17,
            '1': 18, '2': 19, '3': 20, '4': 21, '5': 22, '6': 23, '7': 24, '8': 25, '9': 26
        }
        self.pad_idx = 27

    def __call__(self, smiles):
        # Convert the SMILES string into a list of characters
        chars = list(smiles)

        # Convert each character into its corresponding index
        token_ids = [self.char_to_idx.get(char, self.pad_idx) for char in chars]



        # Convert the padded token IDs list into a PyTorch tensor
        tensor = torch.tensor(token_ids, dtype=torch.long)

        return tensor

def pad_sequence(seq, max_len):
    # Get the current length of the sequence
    seq_len = len(seq)

    # If the sequence is already at the maximum length, return it as is
    if seq_len == max_len:
        return seq

    # If the sequence is shorter than the maximum length, pad it with zeros
    elif seq_len < max_len:
        padding = torch.zeros(max_len - seq_len, dtype=seq[0].dtype)
        padded_seq = torch.cat((seq, padding), dim=0)
        return padded_seq

    # If the sequence is longer than the maximum length, truncate it
    else:
        truncated_seq = seq[:max_len]
        return truncated_seq


class TaskDataset(Dataset):
    def __init__(self, data, tokenizer_prot, tokenizer_mol):
        self.data = [(prots, mols, label) for prots, mols, label in data]
        self.tokenizer_prot = tokenizer_prot
        self.tokenizer_mol = tokenizer_mol

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prots, mols, label = self.data[idx]
        tokenized_prots = [self.tokenizer_prot(p) for p in prots]
        tokenized_mols = [self.tokenizer_mol(m) for m in mols]
        return tokenized_prots, tokenized_mols, label


def collate_fn(samples):
    '''
    the input will either be [prot1,prot2][empty mol]; [empty prot][mol1,mol2] or [prot1],[mol1]

    '''
    padded_proteins = []
    padded_molecules = []
    labels = []

    for prots, mols, label in samples:
        padded_proteins.append(prots)
        padded_molecules.append(mols)
        labels.append(label)

    reorder_prot = list(zip(*padded_proteins))
    reorder_mol = list(zip(*padded_molecules))

    # Pad sequences dynamically
    max_prot_lens = [max(len(p) for p in b) for b in reorder_prot]
    max_mol_lens = [max(len(m) for m in b) for b in reorder_mol]

    padded_proteins = [torch.stack([pad_sequence(p, max_prot_lens[i]) for p in b]) for i, b in
                       enumerate(reorder_prot)]
    padded_molecules = [torch.stack([pad_sequence(m, max_mol_lens[i]) for m in b]) for i, b in
                        enumerate(reorder_mol)]
    labels = torch.tensor(labels)

    return padded_proteins, padded_molecules, labels

T_co = TypeVar('T_co', covariant=True)


class MultiTaskBatchSampler(Sampler[T_co]):
    """Defines a sampler to sample multiple datasets with temperature sampling
    in a distributed fashion."""

    def __init__(self, dataset_sizes: List[int], batch_size: int, temperature: float,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 seed: int = 0, shuffle: bool = True) -> None:
        """Constructor for MultiTaskBatchSampler.

        This will not mix a batch of data contains different tasks

        Args:
            dataset_sizes: a list of integers, specifies the number of samples in
                each dataset.
            batch_size: integer, specifies the batch size.
            temperature: float, temperature used for temperature sampling. The larger
                the value, the datasets are sampled equally, and for value of 0, the datasets
                will be sampled according to their number of samples.
            num_replicas: integer, specifies the number of processes.
            rank: integer, specifies the rank of the current process/
            seed: integer, random seed.
            shuffle: bool, if set to true, the datasets will be shuffled in each epoch.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        # By default we drop the last elements if dataset is not divisible by the number of ranks.
        self.rank_dataset_sizes = [dataset_size // self.num_replicas for dataset_size in self.dataset_sizes]
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
        self.total_sizes = [(dataset_size // self.num_replicas) * self.num_replicas for dataset_size in
                            self.dataset_sizes]
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.num_batches_per_epoch = (np.sum(
            dataset_sizes) + self.batch_size - 1) // self.batch_size // self.num_replicas #num batches per epoch for one gpu
        self.shuffle = shuffle

    def generate_tasks_distribution(self):
        """Given the dataset sizes computes the weights to sample each dataset
        according to the temperature sampling."""
        total_size = sum(self.dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        # Defines torch generator, to make random choices consistent across cores in
        # different epochs, the seed needs to be set based on seed and epoch.
        # This operation is very important when we need to synchronize the tasks and models on different gpus
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # Shuffles the datasets if shuffle is set to true.
        indices = []
        for dataset_size in self.dataset_sizes:
            if self.shuffle:
                indices.append(torch.randperm(dataset_size, generator=generator).tolist())
            else:
                indices.append(list(range(dataset_size)))

        # Shards the datasets across the all processes.
        self.rank_indices = []
        for i in range(len(self.dataset_sizes)):
            self.rank_indices.append(indices[i][self.rank:self.total_sizes[i]:self.num_replicas])

        # To make the model consistent across different processes, since the
        # model is based on tasks, we need to make sure the same task is selected
        # across different processes.
        tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

        # Chooses the tasks which will be used in each batch in one epoch.
        # With passing generator, we make sure this choice is consistent across
        # different processes.
        batch_task_assignments = torch.multinomial(tasks_distribution,
                                                   self.num_batches_per_epoch, replacement=True, generator=generator)

        for batch_task in batch_task_assignments:
            # Gets the number of samples of the selected datasets available for the
            # current rank. keep it to per rank
            num_task_samples = self.rank_dataset_sizes[batch_task]
            # Computes the random samples from the chosen dataset.
            indices = torch.randint(low=0, high=num_task_samples, size=(self.batch_size,), generator=generator).tolist()
            # Converts the selected indices to the global indices on the given dataset.
            results = (self.dataset_offsets[batch_task] + torch.tensor(self.rank_indices[batch_task])[indices]).tolist()
            yield results

    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch


'''
Example check
'''


def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    print(f"Running basic DDP on rank {rank}.")
    setup(rank, world_size)
    data = [(["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"], ['CCCC', 'CHCHC'], [0]),
            (["KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",'PVSGAQLAEELS'], ['CCCCCHC', 'CCCCC'], [2]),
            (['MKTVRQERLKSIV', 'PVSGAQLAEELS'], ['CCNCC','CCCCC'], [4]),
            (['MKTVRQERLKSIV', 'PVSGAQLAEELS'], ['CCNCC','CCCCC'], [5]),
            (['MKTVRQERLKSIV', 'PVSGAQLAEELS'], ['CCNCC','CCCCC'], [6])]



    # create model and move it to GPU with id rank
    alphabet = Alphabet.from_architecture("ESM-1b")
    protein_tokenizer = protein_encoder(alphabet)
    model_data = torch.load('/home/lijiali/projects/DTI_JIALI/DTI_JIALI/base/esm2_t6_8M_UR50D.pt', map_location="cpu")
    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    adapter_config = MetaAdapterConfig()
    adapter_config.device = 'cpu'
    state_dict = upgrade_state_dict(state_dict)
    model = DTI(
        adapter_config=adapter_config,
        protein_encoder_config=cfg,
        protein_tokenizer=alphabet,
    )
    model.load_state_dict(state_dict,strict=False)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    molecule_tokenizer = molecule_encoder()

    # Create dataset and dataloader
    dataset = TaskDataset(data, protein_tokenizer, molecule_tokenizer)
    dataset_sizes = [3,2]
    train_batch_size = 2
    temperature = 0.5
    num_replicas = world_size

    batch_sampler_multi = MultiTaskBatchSampler(dataset_sizes, train_batch_size,
                                     temperature, rank=rank,
                                     num_replicas=num_replicas)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler_multi)
    for batch in dataloader:
        print(batch)


    print('cool')



    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

# Example usage
if __name__ == '__main__':

    run_demo(demo_basic,2)



    # Create example data

    # data = [(["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"], ['CCCC', 'CHCHC'], [0]),
    #         (["KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",'PVSGAQLAEELS'], ['CCCCCHC', 'CCCCC'], [2]),
    #         (['MKTVRQERLKSIV', 'PVSGAQLAEELS'], ['CCNCC','CCCCC'], [4]),
    #         (['MKTVRQERLKSIV', 'PVSGAQLAEELS'], ['CCNCC','CCCCC'], [5]),
    #         (['MKTVRQERLKSIV', 'PVSGAQLAEELS'], ['CCNCC','CCCCC'], [6])]
    #
    # # Set protein and molecule tokenizer
    # alphabet = Alphabet.from_architecture("ESM-1b")
    # protein_tokenizer = protein_encoder(alphabet)
    # model_data = torch.load('/home/lijiali/projects/DTI_JIALI/DTI_JIALI/base/esm2_t6_8M_UR50D.pt', map_location="cpu")
    # cfg = model_data["cfg"]["model"]
    # state_dict = model_data["model"]
    # adapter_config = MetaAdapterConfig()
    # adapter_config.device = 'cpu'
    # state_dict = upgrade_state_dict(state_dict)
    # model = DTI(
    #     adapter_config=adapter_config,
    #     protein_encoder_config=cfg,
    #     protein_tokenizer=alphabet,
    # )
    # model.load_state_dict(state_dict,strict=False)
    # model.cuda()
    #
    #
    #
    # molecule_tokenizer = molecule_encoder()
    #
    # # Create dataset and dataloader
    # dataset = TaskDataset(data, protein_tokenizer, molecule_tokenizer)
    # dataset_sizes = [3,2]
    # train_batch_size = 2
    # temperature = 0
    # rank = None
    # num_replicas = None
    #
    # batch_sampler_multi = MultiTaskBatchSampler(dataset_sizes, train_batch_size,
    #                                  temperature, rank=rank,
    #                                  num_replicas=num_replicas)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, batch_sampler=batch_sampler_multi)
    # task = 'test'
    # model.eval()
    # # Iterate through batches
    # for batch in dataloader:
    #     # tokenized_proteins, tokenized_molecules, labels = batch
    #     # print(tokenized_proteins[0].shape) #batch_size x seq_len
    #     # print(tokenized_molecules)
    #     # print(labels)
    #     tokenized_proteins, tokenized_molecules, labels = batch
    #
    #     with torch.no_grad():
    #         pred = model(protein_tokens=tokenized_proteins, mol_tokens=tokenized_molecules, task=task)
    #
    #     print('cool')



