from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_varlen_qkvpacked_func,flash_attn_varlen_kvpacked_func
import torch
import torch.nn as nn, Tensor
from typing import Optional
import copy

from .sinekan import KANFeedForwardBlock
from .utils import PositionalEncoding, TokenEmbedding

class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, **factory_kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        self.factory_kwargs = factory_kwargs

    def forward(
        self,
        q,
        k=None,
        v=None,
        padding_mask=None,
        is_cross=False,
        causal=False
    ):
        b, l_q, _ = q.shape
        device = q.device

        if not is_cross:
            # Self-attention
            k, v = q, q

        # Projections
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        # Build key padding mask
        if padding_mask is None:
            padding_mask = torch.ones(
                (b, k.shape[1]), dtype=torch.bool, device=device
            )
        else:
            padding_mask = ~padding_mask  # Convert to FlashAttention format (true for padding)
        
            if padding_mask.shape[0] != b or padding_mask.shape[1] != k.shape[1]:
                raise ValueError(
                    f"padding_mask shape {padding_mask.shape} does not match batch size {b} or sequence length {k.shape[1]}"
                )

        if not is_cross:
            # Self-attention (Q=K=V)
            qkv = torch.stack([q_proj, k_proj, v_proj], dim=1)  # (b, 3, l, d)
            qkv = qkv.transpose(1, 2)  # (b, l, 3, d)

            qkv, indices, cu_seqlens, max_s, _ = unpad_input(qkv, padding_mask)
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

            out_unpad = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                dropout_p=self.dropout,
                softmax_scale=None,
                causal=causal,
            )
        else:
            # Cross-attention
            q_mask = torch.ones((b, l_q), dtype=torch.bool, device=device)
            q_packed, q_indices, cu_q, max_q, _ = unpad_input(q_proj, q_mask)

            kv = torch.stack([k_proj, v_proj], dim=2)  # (b, l_k, 2, d)
            kv, _, cu_k, max_k, _ = unpad_input(kv, padding_mask)

            q_packed = q_packed.view(-1, self.num_heads, self.head_dim)
            kv = kv.view(-1, 2, self.num_heads, self.head_dim)

            out_unpad = flash_attn_varlen_kvpacked_func(
                q_packed,
                kv,
                cu_q,
                cu_k,
                max_q,
                max_k,
                dropout_p=self.dropout,
                softmax_scale=None,
                causal=causal,
            )

        # Pad and return
        out = out_unpad.reshape(-1, self.embed_dim)
        out = pad_input(out, q_indices if is_cross else indices, b, l_q)

        return self.out_proj(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation : nn.Module = torch.nn.functional.gelu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            d_model,
            nhead,
            dropout=dropout,
            **factory_kwargs
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        

    def _sa_block(self, x, padding_mask, is_causal):
        x = self.self_attn(x, x, x, causal=is_causal, padding_mask=padding_mask)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_pad_mask=None, is_causal=False):
        '''
        Arguments:
            src: (batch_size, seq_len, d_model)
            src_mask: (batch_size, seq_len, seq_len)
            is_causal: bool
        '''
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_pad_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_pad_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        is_kan=False,
        kan_grid_size=8,
        kan_ff_dims=None,
        dropout=0.1,
        activation: nn.Module = torch.nn.functional.gelu,
        layer_norm_eps=1e-5,
        norm_first=False,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.is_kan - is_kan
        self.self_attn = FlashMHA(
            d_model,
            nhead,
            dropout=dropout,
            **factory_kwargs,
        )
        self.cross_attn = FlashMHA(
            d_model,
            nhead,
            dropout=dropout,
            **factory_kwargs,
        )
        if self.is_kan:
            self.kan_ff = KANFeedForwardBlock(d_model, kan_ff_dims, grid_size=kan_grid_size, device=device)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
            self.dropout = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def _sa_block(self, x, padding_mask=None, is_causal=False):
        x = self.self_attn(x, padding_mask=padding_mask, causal=is_causal)
        return self.dropout1(x)

    def _crossa_block(self, x, memory, padding_mask=None, is_causal=False):
        x = self.cross_attn(x, k=memory, v=memory, padding_mask=padding_mask, is_cross=True, causal=is_causal)
        return self.dropout2(x)

    def _ff_block(self, x):
        if self.is_kan:
            x = self.kan_ff(x)
        else:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_padding_mask, tgt_is_causal)
            x = x + self._crossa_block(self.norm2(x), memory, memory_padding_mask, memory_is_causal)
            if self.is_kan:
                x = self._ff_block(self.norm3(x))
            else:    
                x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._crossa_block(x, memory, memory_padding_mask, memory_is_causal))
            if self.is_kan:
                x = self.norm3(self._ff_block(x))
            else:
                x = self.norm3(x + self._ff_block(x))

        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.layers = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.norm = norm

    def forward(self, src, src_padding_mask=None, is_causal=False):
        output = src
        for mod in self.layers:
            output = mod(output, src_pad_mask=src_padding_mask, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = nn.Modulelist() 
        for i in range(num_layers):
            
            layer = copy.deepcopy(decoder_layer)
            layer.is_kan = (i == num_layers - 1)

            if layer.is_kan:
                layer.kan_ff = KANFeedForwardBlock(
                    layer.self_attn.embed_dim,
                    kan_ff_dims=[layer.self_attn.embed_dim, 2 * layer.self_attn.embed_dim, layer.self_attn.embed_dim],
                    grid_size=layer.kan_ff.grid_size if hasattr(layer, 'kan_ff') else 8,
                    device=next(layer.parameters()).device
                )

            self.layers.append(layer)
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_padding_mask=None,
        memory_padding_mask=None,
        tgt_is_causal=True,
        memory_is_causal=False,
    ):
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=memory_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.functional.gelu,
        layer_norm_eps=1e-5,
        norm_first=True,
        bias=True,
        device='cpu',
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(
        self,
        src,
        tgt,
        src_padding_mask=None,
        tgt_padding_mask=None,
        memory_padding_mask=None,
        src_is_causal=False,
        tgt_is_causal=True,
        memory_is_causal=False,
    ):
        memory = self.encoder(src, src_padding_mask=src_padding_mask, is_causal=src_is_causal)
        output = self.decoder(
            tgt,
            memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return output

class Model(nn.Module):
    """
    Transformer-based model for sequence-to-sequence tasks.

    Args:
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        emb_size (int): Size of the embedding.
        nhead (int): Number of attention heads.
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 512.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Model, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=False,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        """
        Forward pass of the model.

        Args:
            src (Tensor): Source input.
            trg (Tensor): Target input.
            src_mask (Tensor): Mask for source input.
            tgt_mask (Tensor): Mask for target input.
            src_padding_mask (Tensor): Padding mask for source input.
            tgt_padding_mask (Tensor): Padding mask for target input.
            memory_key_padding_mask (Tensor): Padding mask for memory.

        Returns:
            Tensor: Output tensor.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)
