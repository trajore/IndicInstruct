"""
Code adapted from GPT-NeoX
https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py
"""
import math
import torch.nn as nn


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def xavier_uniform_init_method():
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
    """

    def init_(tensor):
        return nn.init.xavier_uniform_(tensor)

    return init_


def xavier_normal_init_method():
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
    """

    def init_(tensor):
        return nn.init.xavier_normal_(tensor)

    return init_


def small_init_init_method(dim):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    """
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def scaled_biderman(sigma, num_layers, hidden_size, intermediate_size, wout_or_w2):
    """Init method based on N(0, sigma/sqrt(2*num_layers) for W_out and W2, else N(0, sigma) for the rest. sigma is calculated as sqrt(2/(hidden_size + intermediate_size))"""

    sigma = math.sqrt(2 / (hidden_size + intermediate_size))
    if wout_or_w2:
        std = sigma / math.sqrt(2.0 * num_layers)
    else:
        std = sigma

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim):
    std = 2 / n_layers / math.sqrt(dim)

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class InitializationScheme:
    def __init__(self, config):
        self.config = config
        if config.init_scheme == "normal":
            init_method = init_method_normal(self.config.initializer_range)
        elif config.init_scheme == "scaled_normal":
            init_method = scaled_init_method_normal(
                self.config.initializer_range, self.config.num_hidden_layers
            )
        elif config.init_scheme == "scaled_biderman":
            print("scaled_biderman is being used for llama")
            self._init_method_wout_w2 = scaled_biderman(
                self.config.initializer_range,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                True,
            )
            init_method = scaled_biderman(
                self.config.initializer_range,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                False,
            )
        elif config.init_scheme == "xavier_uniform":
            init_method = xavier_uniform_init_method()
        elif config.init_scheme == "xavier_normal":
            init_method = xavier_normal_init_method()
        elif config.init_scheme == "wang_init":
            init_method = wang_init_method(config.num_hidden_layers, config.hidden_size)
        elif config.init_scheme == "small_init":
            init_method = small_init_init_method(config.hidden_size)
        elif config.init_scheme == "small_and_wang_init":
            init_method = small_init_init_method(config.hidden_size)
            self._output_init_method = wang_init_method(
                config.num_hidden_layers, config.hidden_size
            )
        else:
            raise NotImplementedError(f"Unknown init method {config.init_scheme}")

        self._init_method = init_method

    def _init_weights(self, model):
        # initialize all the models
        if self.config.init_scheme == "scaled_biderman":
            # for i in model.named_parameters():
            #     print(i)
            # print("Padding idx is", self.config.pad_token_id)
            for i, module in enumerate(model.named_parameters()):
                if "o_proj" in module[0] or "down_proj" in module[0]:
                    # print(module[0], "is initialized with scaled_biderman with division by sqrt(2*num_layers)")
                    self._init_method_wout_w2(module[1].data)
                else:
                    # print(module[0], "is initialized with scaled_biderman without division by sqrt(2*num_layers)")
                    self._init_method(module[1].data)
                    if "embed_tokens" in module[0] and self.config.pad_token_id is not None:
                        # print("padding_idx is not None")
                        module[1].data[self.config.pad_token_id].zero_()
        else:
            for i, module in enumerate(model.parameters()):
                if isinstance(module, nn.Linear):
                    self._init_method(module.weight.data)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    self._init_method(module.weight.data.normal_)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()

            if self.config.init_scheme == "small_and_wang_init":
                module = model.lm_head
                self._output_init_method(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
