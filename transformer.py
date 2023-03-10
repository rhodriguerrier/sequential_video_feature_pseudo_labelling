import copy
from typing import Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import numpy as np


class ViT(Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=512, nhead=4)
        self.num_encoders = 1
        self.cls_token = nn.Parameter(torch.rand(1, 512))
        self.rgb_token = nn.Parameter(torch.rand(1, 1, 512))
        self.flow_token = nn.Parameter(torch.rand(1, 1, 512))
        self.relu = nn.ReLU()
        self.rgb_linear_layer_1 = nn.Linear(1024, 1024)
        self.rgb_linear_layer_2 = nn.Linear(1024, 512)
        self.flow_linear_layer_1 = nn.Linear(1024, 1024)
        self.flow_linear_layer_2 = nn.Linear(1024, 512)
        self.cls_linear_layer = nn.Linear(512, 8)
        self.positional_embeddings = get_positional_embeddings(6, 48, 512)

    def forward(self, rgb_features, flow_features, attn_mask=None):
        # Convert 1024D features to 512D for RGB and Flow
        #rgb_tokens = self.rgb_linear_layer_2(self.relu(self.rgb_linear_layer_1(rgb_features)))
        #flow_tokens = self.flow_linear_layer_2(self.relu(self.flow_linear_layer_1(flow_features)))

        rgb_tokens = self.rgb_linear_layer_2(rgb_features)
        flow_tokens = self.flow_linear_layer_2(flow_features)

        # Add relevant modality embedding
        rgb_tokens = rgb_tokens + self.rgb_token
        flow_tokens = flow_tokens + self.flow_token

        # Add same positional encodings to each modality
        pos_embeddings = get_positional_embeddings(rgb_features.size(0)+1, rgb_tokens.size(1), 512)
        tagged_rgb_tokens = rgb_tokens + pos_embeddings[:-1]
        tagged_flow_tokens = flow_tokens + pos_embeddings[:-1]

        # Add in CLS token and add positional encoding to this too
        tagged_cls_tokens = torch.unsqueeze(self.cls_token.repeat(rgb_tokens.size(1), 1) + pos_embeddings[-1], dim=0)
        trans_input = torch.cat((tagged_rgb_tokens, tagged_flow_tokens, tagged_cls_tokens))

        # Only apply attention mask if provided
        output = trans_input
        for _ in range(self.num_encoders):
            if attn_mask is not None:
                output, attn_scores = self.encoder_layer(output, src_key_padding_mask=attn_mask)
            else:
                output, attn_scores = self.encoder_layer(output)
        output = self.cls_linear_layer(output[-1])
        return output

    def extract_logits_ft(self, rgb_features, flow_features, attn_mask=None):
        rgb_tokens = self.rgb_linear_layer_2(rgb_features)
        flow_tokens = self.flow_linear_layer_2(flow_features)

        # Add relevant modality embedding
        rgb_tokens = rgb_tokens + self.rgb_token
        flow_tokens = flow_tokens + self.flow_token

        # Add same positional encodings to each modality
        pos_embeddings = get_positional_embeddings(rgb_features.size(0)+1, rgb_tokens.size(1), 512)
        tagged_rgb_tokens = rgb_tokens + pos_embeddings[:-1]
        tagged_flow_tokens = flow_tokens + pos_embeddings[:-1]

        # Add in CLS token and add positional encoding to this too
        tagged_cls_tokens = torch.unsqueeze(self.cls_token.repeat(rgb_tokens.size(1), 1) + pos_embeddings[-1], dim=0)
        trans_input = torch.cat((tagged_rgb_tokens, tagged_flow_tokens, tagged_cls_tokens))

        # Only apply attention mask if provided
        output = trans_input
        for _ in range(self.num_encoders):
            if attn_mask is not None:
                output, attn_scores = self.encoder_layer(output, src_key_padding_mask=attn_mask)
            else:
                output, attn_scores = self.encoder_layer(output)
        return self.cls_linear_layer(output[-1]), output[-1]


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def get_positional_embeddings(sequence_len, batch_size, token_dim):
    result = torch.ones(sequence_len, batch_size, token_dim)
    for i in range(sequence_len):
        pos_vec = torch.ones(token_dim)
        for j in range(token_dim):
            pos_vec[j] = np.sin(i / (10000 ** (j/ token_dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / token_dim)))
        result[i] = pos_vec.repeat(batch_size, 1)
    return result
