
import torch
import torch.nn as nn
from DTI_JIALI.base import (SamplerOutput, LayerNormOutput,
                              AdapterDTIBlockOutput, AdapterOutput)
from dataclasses import dataclass
from collections import OrderedDict
import torch.nn.functional as F
from transformers.activations import get_activation
from DTI_JIALI.base import MultiheadAttention  # noqa
import math
from DTI_JIALI.base import Alphabet
from typing import Union
from DTI_JIALI.base import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead
import re
"""Implements an Adapter and Hyper-adapter Layers."""


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)

def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class TaskHyperNet(nn.Module):
    """This module generates the task-embeddings from the initial feeded task embeddings."""

    def __init__(self, config):
        super(TaskHyperNet, self).__init__()
        self.task_hidden_dim = config.task_hidden_dim
        self.projected_task_embedding_dim = config.projected_task_embedding_dim
        self.task_embeding_generator = nn.Sequential(
            linear_layer(config.task_embedding_dim, self.task_hidden_dim),
            nn.ReLU(),
            linear_layer(self.task_hidden_dim, self.projected_task_embedding_dim))

    def forward(self, task_embedding):
        task_embedding = task_embedding.view(-1)
        return self.task_embeding_generator(task_embedding).view(-1)

class LayerNormHyperNet(nn.Module):
    """This module generates the weight and bias for the task conditioned layer norm."""

    def __init__(self, config):
        super(LayerNormHyperNet, self).__init__()
        self.task_embedding_dim = config.projected_task_embedding_dim \
            if config.train_task_embeddings else config.task_embedding_dim
        self.weight_generator = linear_layer(self.task_embedding_dim, config.input_dim)
        self.bias_generator = linear_layer(self.task_embedding_dim, config.input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input) # N * input_dim

class TaskEmbeddingController(nn.Module):
    """
    Main module controlling task embeddings.
    The task embedding can be randomly initialized or can be produced by a simple task embedding net
    """

    def __init__(self, config):
        super(TaskEmbeddingController, self).__init__()
        self.device = config.device
        self.task_embedding_dim = config.task_embedding_dim
        self.tasks = config.tasks
        self.task_to_task_embeddings = {task: task for task in self.tasks}
        if config.task_to_embeddings is not None:
            self.task_to_task_embeddings = config.task_to_embeddings
            self.tasks = self.task_to_task_embeddings.values()
        self.set_task_embeddings(self.tasks)
        self.train_task_embeddings = config.train_task_embeddings
        if self.train_task_embeddings:
            self.task_hyper_net = TaskHyperNet(config)

    def get_task(self, task):
        return self.task_to_task_embeddings[task]

    def set_task_embeddings(self, tasks):
        self.task_to_embeddings = nn.ParameterDict(dict())
        for task in tasks:
            task_embedding = torch.Tensor(torch.randn(self.task_embedding_dim)).to(self.device)
            self.task_to_embeddings[task] = nn.Parameter(task_embedding)

    def forward(self, task):
        task_mapped = self.get_task(task)
        task_embedding = self.task_to_embeddings[task_mapped]
        if self.train_task_embeddings:
            return self.task_hyper_net(task_embedding)
        return task_embedding

class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.weight_init_range = config.weight_init_range
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = linear_layer(self.input_dim, self.down_sample_size, std=self.weight_init_range)
        self.up_sampler = linear_layer(self.down_sample_size, self.input_dim, std=self.weight_init_range)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)

class AdapterHyperNet(nn.Module):
    """This module generates the weights for the meta adapter layers."""

    def __init__(self, config, input_dim, output_dim):
        super(AdapterHyperNet, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_task_embeddings = config.train_task_embeddings
        self.task_embedding_dim = config.projected_task_embedding_dim if \
            config.train_task_embeddings else config.task_embedding_dim
        # Considers weight and bias parameters for generating adapter weights.
        self.weight_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(self.task_embedding_dim, self.input_dim))

    def forward(self, task_embedding):
        # task_embedding = torch.rand(512)
        task_embedding = task_embedding.view(-1)
        weight = self.weight_generator(task_embedding).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(task_embedding).view(-1)
        return weight, bias

class AdapterLayersHyperNet(nn.Module):
    """
    This module generates the weights for all the meta adapter layers
    given the task embeddings and layer id.
    This module is good for one pretrained model, however, if there are two pretrained
    models with different architectures, there will be some challenges. since the input and output dim is different
    for different models


    """

    def __init__(self, config, input_dim, output_dim):
        super(AdapterLayersHyperNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(config.projected_task_embedding_dim, self.input_dim))

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.input_dim, self.output_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return SamplerOutput(weight=weight, bias=bias)

class AdapterLayersHyperNetController(nn.Module):
    """
    This modules contains the hyper-nets for the feed forward
    and self-attention modules and it generates the adapter's weights and
    layer norm's weights for all the layers of transformers.
    Still it is for a single model, not for two different models
    """

    def __init__(self, config, num_layers=6):
        super(AdapterLayersHyperNetController, self).__init__()
        self.num_layers = num_layers
        self.layer_norm_epsilon = 1e-6
        self.max_position_embeddings = 2
        self.device = config.device
        self.task_embedding_dim = config.task_embedding_dim
        self.layer_id_embeddings = nn.Embedding(self.num_layers,
                                                self.task_embedding_dim).to(self.device)
        # self.token_type_embeddings = nn.Embedding(self.max_position_embeddings,
        #                                          self.task_embedding_dim).to(self.device)
        config.task_embedding_dim = self.task_embedding_dim * 2
        self.task_hypernet = TaskHyperNet(config)
        config.task_embedding_dim = self.task_embedding_dim
        self.unique_hyper_net_layer_norm = config.unique_hyper_net_layer_norm
        if self.unique_hyper_net_layer_norm:
            self.LayerNorm = nn.LayerNorm(config.projected_task_embedding_dim, eps=self.layer_norm_epsilon)
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        # Defines the adapters hyper-nets.
        self.feed_forward_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                       self.input_dim, self.down_sample_size)
        self.feed_forward_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.down_sample_size, self.input_dim)
        self.self_attention_up_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                         self.input_dim, self.down_sample_size)
        self.self_attention_down_sampler_hyper_net = AdapterLayersHyperNet(config,
                                                                           self.down_sample_size, self.input_dim)
        # Defines the layer norms' hyper net.
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.train_task_embeddings = config.train_task_embeddings
        config.train_task_embeddings = True
        if self.add_layer_norm_before_adapter:
            self.feed_forward_pre_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_pre_layernorm_hypernet = LayerNormHyperNet(config)
        if self.add_layer_norm_after_adapter:
            self.feed_forward_post_layernorm_hypernet = LayerNormHyperNet(config)
            self.self_attention_post_layernorm_hypernet = LayerNormHyperNet(config)
        config.train_task_embeddings = self.train_task_embeddings

    def get_embedding(self, task_embedding, layer_id):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long, device=self.device)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor)
        layer_embedding = layer_embedding.view(-1)
        embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1)], axis=0)
        embeddings = self.task_hypernet(embeddings.view(-1))
        if self.unique_hyper_net_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        return embeddings

    def forward(self, task_embedding, layer_id):
        embeddings = self.get_embedding(task_embedding, layer_id)
        # Generates the adapters weights in feed-forward and self-attention modules.
        feed_forward_down = self.feed_forward_down_sampler_hyper_net(embeddings)
        feed_forward_up = self.feed_forward_up_sampler_hyper_net(embeddings)
        self_attention_down = self.self_attention_down_sampler_hyper_net(embeddings)
        self_attention_up = self.self_attention_up_sampler_hyper_net(embeddings)
        feed_forward_output = AdapterOutput(up=feed_forward_up, down=feed_forward_down)
        self_attention_output = AdapterOutput(up=self_attention_up, down=self_attention_down)
        # Generates the weights and baises for pre and post layer norms.
        if self.add_layer_norm_before_adapter:
            weight, bias = self.feed_forward_pre_layernorm_hypernet(embeddings)
            feed_forward_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_pre_layernorm_hypernet(embeddings)
            self_attention_output.pre_norm = LayerNormOutput(weight=weight, bias=bias)
        if self.add_layer_norm_after_adapter:
            weight, bias = self.feed_forward_post_layernorm_hypernet(embeddings)
            feed_forward_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
            weight, bias = self.self_attention_post_layernorm_hypernet(embeddings)
            self_attention_output.post_norm = LayerNormOutput(weight=weight, bias=bias)
        return AdapterDTIBlockOutput(feed_forward=feed_forward_output,
                                    self_attention=self_attention_output)


"""Implements the adapters' configurations."""

@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    conditional_layer_norm = False
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    train_adapters_blocks = True


class MetaAdapterConfig(AdapterConfig):
    """Implements Meta adapter in which a hyper-network generates the parameters of
     adapter layers. In this case we have a task embeddings which is feed to the
     hyper-network to allow it generate the weights for the adapter layers."""
    task_embedding_dim = 512
    tasks = ['test','main']
    task_embedding_dir = None
    hidden_dim = 128
    input_dim = 320 #get from the model embedded dim
    device = 'cpu' # get from model
    train_task_embeddings = False
    task_to_embeddings = None
    projected_task_embedding_dim = 64
    task_hidden_dim = 128
    parametric_task_embedding = False
    # If Specified, uses one hypernet to generates the adapters weights.
    unique_hyper_net = False
    unique_hyper_net_layer_norm = True
    # We consider only one hyper-net for all the blocks of transformer.
    efficient_unique_hyper_net = False


ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig),
     ("meta-adapter", MetaAdapterConfig)])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
                .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))

class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers.
    This is the conventional adapter approach"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.task_to_adapter = {task: task for task in self.tasks}
        # If a dictionary from task to adapter is given, the task is over-written by the given adapters.
        if config.task_to_adapter is not None:
            self.task_to_adapter = config.task_to_adapter
            self.tasks = self.task_to_adapter.values()
        self.adapters = self.construct_adapters(self.tasks)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.input_dim)

    def set_task_to_adapter_map(self, mapping):
        self.task_to_adapter = mapping

    def get_task(self, task):
        return self.task_to_adapter[task]

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        for task in tasks:
            self.adapters[task] = Adapter(self.config)
        return self.adapters

    def disable_adapters(self, tasks):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = False

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = True

    def get_adapter(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[task]

    def forward(self, task, inputs):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        task = self.get_task(task)
        # Enables the adapter layer for the given task.
        self.enable_adapters(task)
        # Disable other adapters.
        other_tasks = [x for x in self.tasks if x != task]
        self.disable_adapters(other_tasks)
        adapter = self.get_adapter(task)
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs # add inputs to outputs
        return outputs


class MetaAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a hyper-network.
    In this case, task-embeddings are fixed, and the task
    embeddings will be initialized to random."""
    # This one seems to be the real used one, this one and the AdapterLayersHyperNetController is a bit repeated

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.adapters = nn.ModuleDict(dict())
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.meta_up_sampler = AdapterHyperNet(config, self.input_dim, self.down_sample_size)
        self.meta_down_sampler = AdapterHyperNet(config, self.down_sample_size, self.input_dim)
        # down is before up input_size -> down_size -> input_size
        self.activation_type = config.non_linearity.lower()
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.conditional_layer_norm = config.conditional_layer_norm
        if self.add_layer_norm_after_adapter:
            if self.conditional_layer_norm:
                self.post_layernorm_hypernet = LayerNormHyperNet(config)
            else:
                self.post_layer_norm = nn.LayerNorm(self.input_dim)
        if self.add_layer_norm_before_adapter:
            if self.conditional_layer_norm:
                self.pre_layernorm_hypernet = LayerNormHyperNet(config)
            else:
                self.pre_layer_norm = nn.LayerNorm(self.input_dim)

    def call_adapter(self, inputs, task_embedding):
        weight_up, bias_up = self.meta_up_sampler(task_embedding)
        weight_down, bias_down = self.meta_down_sampler(task_embedding)
        # The following is a forward process with weight and bias dynamically generate from hypernet
        # the weight and bias of down and up is only depend on the hypernet and will not be kept for next round
        # Although the gradient will be passed back to update the parameters of hypernet
        down = F.linear(inputs, weight=weight_down, bias=bias_down)
        middle = get_activation(self.activation_type)(down)
        output = F.linear(middle, weight=weight_up, bias=bias_up)
        return output

    def apply_pre_layer_norm(self, inputs, task_embeddings):
        """Applies pre layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.pre_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.pre_layer_norm(inputs)

    def apply_post_layer_norm(self, inputs, task_embeddings):
        """Applies post layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.post_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.post_layer_norm(inputs)

    def forward(self, task_embedding, inputs):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        z = self.apply_pre_layer_norm(inputs, task_embedding) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, task_embedding)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_post_layer_norm(outputs, task_embedding)
        outputs = outputs + inputs
        return outputs

class AutoAdapterController(nn.Module):
    """Generic adapter controller class to instantiate different adapter
    controller classes."""

    @classmethod
    def get(cls, config):
        if isinstance(config, MetaAdapterConfig):
            return MetaAdapterController(config)
        elif isinstance(config, AdapterConfig):
            return AdapterController(config)
        raise ValueError("Unrecognized adapter config", config)



'''Implement the protein pretrained model with hypernet adapters'''


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

class TransformerLayer(nn.Module):
    """Transformer layer block. I have added in the adapter networks generated by hypernet into it"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        config,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
        train_adapter=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)
        self.train_adapter=train_adapter
        if self.train_adapter:
            self.adapter_controller = MetaAdapterController(config=config)




    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False,task=None, task_embedding=None
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )

        adapter_residual_attn = x
        x = self.adapter_controller(task_embedding, x)
        x = adapter_residual_attn + x


        x = residual + x



        residual = x

        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        adapter_residual_ffn = x
        x = self.adapter_controller(task_embedding, x)
        x = adapter_residual_ffn + x

        x = residual + x

        return x, attn

class ESM2(nn.Module):
    def __init__(
        self,
        adapter_config,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.adapter_config = adapter_config
        self.task_embedding_controller = TaskEmbeddingController(self.adapter_config)

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    config=self.adapter_config,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        # self.lm_head = RobertaLMHead(
        #     embed_dim=self.embed_dim,
        #     output_dim=self.alphabet_size,
        #     weight=self.embed_tokens.weight,
        # )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False, task=None, task_embedding=None):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                task=task,
                task_embedding=self.task_embedding_controller(task),
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        # x = self.lm_head(x)

        result = {"embedding": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts


        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

class DTI(nn.Module):
    def __init__(
        self,
        adapter_config,
        protein_encoder_config,
        protein_tokenizer,
    ):
        super().__init__()
        self.protein_encoder = ESM2(
        adapter_config=adapter_config,
        num_layers=protein_encoder_config.encoder_layers,
        embed_dim=protein_encoder_config.encoder_embed_dim,
        attention_heads=protein_encoder_config.encoder_attention_heads,
        alphabet=protein_tokenizer,
        token_dropout=protein_encoder_config.token_dropout,
    )
        self.out_put_protein = nn.Linear(protein_encoder_config.encoder_embed_dim, 1)

    def forward(self, protein_tokens, mol_tokens, protein_repr_layers=[],
                protein_need_head_weights=False, protein_return_contacts=False, task=None,
                task_embedding=None):

        num_proteins = len(protein_tokens)
        num_mols = len(mol_tokens)

        if num_proteins == 2:
            protein_enc_list = []

            results1 = self.protein_encoder(tokens=protein_tokens[0].to(device="cuda"), repr_layers=protein_repr_layers,
                                       need_head_weights=protein_need_head_weights,
                                       return_contacts=protein_return_contacts,
                                       task=task, task_embedding=task_embedding)
            protein_enc_inf1 = results1["embedding"]
            protein_enc_inf1 = torch.mean(protein_enc_inf1, 1)
            protein_enc_list.append(protein_enc_inf1)

            results2 = self.protein_encoder(tokens=protein_tokens[1].to(device="cuda"), repr_layers=protein_repr_layers,
                                       need_head_weights=protein_need_head_weights,
                                       return_contacts=protein_return_contacts,
                                       task=task, task_embedding=task_embedding)
            protein_enc_inf2 = results2["embedding"]
            protein_enc_inf2 = torch.mean(protein_enc_inf2, 1)
            protein_enc_list.append(protein_enc_inf2)




        else:
            protein_enc_list = []
            protein_enc_inf1 = self.protein_encoder(tokens=protein_tokens[0].to(device="cuda"), repr_layers=protein_repr_layers,
                                       need_head_weights=protein_need_head_weights,
                                       return_contacts=protein_return_contacts,
                                       task=task, task_embedding=task_embedding)

            protein_enc_list.append(protein_enc_inf1)

        if num_mols == 2:
            mol_enc_list = []
            mol_enc_inf1 = mol_tokens[0].to(device="cuda")
            mol_enc_list.append(mol_enc_inf1)
            mol_enc_inf2 = mol_tokens[1].to(device="cuda")
            mol_enc_list.append(mol_enc_inf2)

        else:
            mol_enc_list = []
            mol_enc_inf1 = mol_tokens[0].to(device="cuda")
            mol_enc_list.append(mol_enc_inf1)


        reduced_protein_information = torch.sum(torch.stack(protein_enc_list), dim=0)



        pred = self.out_put_protein(reduced_protein_information)

        return pred




if __name__ == '__main__':
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    model_data = torch.load('/home/lijiali/projects/DTI_JIALI/DTI_JIALI/base/esm2_t6_8M_UR50D.pt', map_location="cpu")
    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    alphabet = Alphabet.from_architecture("ESM-1b")
    adapter_config = MetaAdapterConfig()
    adapter_config.device = 'cpu'
    state_dict = upgrade_state_dict(state_dict)



    # model = ESM2(
    #     adapter_config=adapter_config,
    #     num_layers=cfg.encoder_layers,
    #     embed_dim=cfg.encoder_embed_dim,
    #     attention_heads=cfg.encoder_attention_heads,
    #     alphabet=alphabet,
    #     token_dropout=cfg.token_dropout,
    # )
    model = DTI(
        adapter_config=adapter_config,
        protein_encoder_config=cfg,
        protein_tokenizer=alphabet,
    )

    model.load_state_dict(state_dict,strict=False)
    # check the new layers, they are initialized with random number or with 0s
    # for name, param in model.state_dict().items():
    #     if 'adapter_controller' in name:
    #         print(name)
    #         print(param)
    # for name, param in model.state_dict().items():
    #     if 'adapter_controller' in name:
    #         print(name)
    #         print(param.device)

    batch_converter = alphabet.get_batch_converter()
    del state_dict
    del model_data
    model.cuda()
    # for name, param in model.state_dict().items():
    #
    #     print(name)
    #     print(param.device)
    model.eval()
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3", "K A <mask> I S Q"),
    ]
    task = 'test'
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens=batch_tokens.to(device="cuda")
    print(batch_tokens.device)



    with torch.no_grad():
        pred = model(protein_tokens=batch_tokens,task=task)







    print('cool')