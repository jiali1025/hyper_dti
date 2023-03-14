import sys
import torch
import datasets
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import EvaluationStrategy

from DTI_JIALI.third_party.models import DTI_model, EsmConfig
from DTI_JIALI.third_party.trainers import DTITrainer
from DTI_JIALI.adapters import AdapterController, AutoAdapterConfig
from DTI.data import AutoTask
from hyperformer.third_party.utils import TaskCollator, check_output_dir
from hyperformer.metrics import build_compute_metrics_fn
from hyperformer.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
    AdapterTrainingArguments
from hyperformer.utils import freezing_params, get_last_checkpoint_path, create_dir,\
    handle_metrics, get_training_args

logger = logging.getLogger(__name__)
