import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from DTI_JIALI.base import DTI
from DTI_JIALI.base import Alphabet
from DTI_JIALI.base import MetaAdapterConfig
import re
from typing import Sequence, Tuple, List

def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict







class DTITrainer:
    def __init__(self, adapter_config=None,
                 data_args=None, dataset_sizes=None,
                 multi_task_compute_metrics=None):
        super().__init__()

        self.adapter_config = adapter_config








    def train(self, protein_model_path=None):
        protein_model_data = torch.load(protein_model_path, map_location="cpu")
        protein_cfg = protein_model_data["cfg"]["model"]
        protein_state_dict = protein_model_data["model"]
        protein_alphabet = Alphabet.from_architecture("ESM-1b")
        adapter_config = self.adapter_config
        adapter_config.device = 'cpu'
        protein_state_dict = upgrade_state_dict(protein_state_dict)

        model = DTI(
            adapter_config=self.adapter_config,
            protein_encoder_config=protein_cfg,
            protein_tokenizer=protein_alphabet,
        )

        model.load_state_dict(protein_state_dict, strict=False)






















