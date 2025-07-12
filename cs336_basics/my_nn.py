import torch
import torch.nn as nn
from typing import Optional
from jaxtyping import Int, Float
from torch import Tensor
import torch.nn.functional as F


class My_linear(nn.Module):
    def __init__(self, d_in, d_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty((d_in, d_out)))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, in_features: torch.Tensor):
        return in_features@self.weight


class My_embedding(nn.Module):
    def __init__(self, vocab_size: int,
                 d_model: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = nn.Parameter(torch.empty((vocab_size, d_model)))
        nn.init.normal_(self.weights, -d_model**0.5, d_model**0.5)

    def forward(self, token_ids):
        return self.weights[token_ids]


class My_swiglu(nn.Module):
    def __init__(self, d_model, d_ff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = My_linear(d_model, d_ff)
        self.weight2 = My_linear(d_ff, d_model)
        self.weight3 = My_linear(d_model, d_ff)

    def forward(self, x):
        w1_x = self.weight1(x)
        w3_x = self.weight3(x)
        silu = w1_x * w1_x.sigmoid()
        swiglu = silu * w3_x
        return self.weight2(swiglu)


def my_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]  # 用于缩放因子

    # 1. 计算打分矩阵 (..., queries, keys)
    scores = Q @ K.transpose(-2, -1) / d_k**0.5

    # 2. 应用 mask（若有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3. softmax → attention weights
    attn_weights = F.softmax(scores, dim=-1)  # over keys axis

    # 4. 加权求和：注意力输出 (..., queries, d_v)
    output = attn_weights @ V
    return output


class My_MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V projections (each maps d_model -> d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Optional[Float[Tensor, "batch seq seq"]] = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        B, T, _ = x.shape
        H = self.num_heads
        d_h = self.d_head

        # 1. Linear projections (can be merged for efficiency)
        # Combine Q, K, V projections into a single linear layer for efficiency
        # But here, since self.q_proj, self.k_proj, self.v_proj are separate, you can do:
        Q = self.q_proj(x)  # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # If you want to merge, you can define:
        # self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # and then:
        # qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        # Q, K, V = qkv.chunk(3, dim=-1)
        # This is more efficient and common in practice.

        # About bias: Yes, Q, K, V projections typically include bias in standard transformer implementations.
        # nn.Linear by default includes bias=True.

        # 2. reshape → (B, H, T, d_head)
        def split_heads(t):
            return t.view(B, T, H, d_h).transpose(1, 2)

        Q, K, V = map(split_heads, (Q, K, V))

        # 3. Apply attention
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T, T)
        attn_out = my_scaled_dot_product_attention(
            Q, K, V, mask=mask)  # (B, H, T, d_head)

        # 4. Merge heads → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(attn_out)


def My_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... seq d_k"],
    token_positions: Int[Tensor, " ... seq"],
) -> Float[Tensor, " ... seq d_k"]:
    """
    应用 Rotary Positional Embedding（RoPE）到输入 query 或 key 向量。
    支持任意 batch 维度和广播位置。
    """
    assert d_k % 2 == 0, "d_k 必须是偶数，用于复数旋转编码"

    # 1. 生成 RoPE 的旋转频率（维度长度一半）
    half_d = d_k // 2
    dim_range = torch.arange(half_d, dtype=torch.float32,
                             device=in_query_or_key.device)
    freqs = 1.0 / (theta ** (dim_range / half_d))  # shape: (half_d,)

    # 2. 基于位置和频率生成旋转角度（pos × freq）
    pos = token_positions.to(torch.float32).unsqueeze(-1)  # (..., seq, 1)
    angles = pos * freqs  # (..., seq, half_d)

    # 3. 拼接成旋转矩阵中的 cos/sin
    cos = angles.cos()  # (..., seq, half_d)
    sin = angles.sin()

    # 4. 交错提取 even / odd：x[..., 0::2], x[..., 1::2]
    x1 = in_query_or_key[..., 0::2]
    x2 = in_query_or_key[..., 1::2]
    # 5. RoPE 旋转
    rope_x1 = x1 * cos - x2 * sin
    rope_x2 = x1 * sin + x2 * cos
    rope_out = torch.empty_like(in_query_or_key)
    rope_out[..., 0::2] = rope_x1
    rope_out[..., 1::2] = rope_x2  # (..., seq, d_k)

    return rope_out


class My_MultiHeadSelfAttentionwithROPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V projections (each maps d_model -> d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Optional[Float[Tensor, "batch seq seq"]] = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        B, T, _ = x.shape
        H = self.num_heads
        d_h = self.d_head

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        def split_heads(t):
            return t.view(B, T, H, d_h).transpose(1, 2)  # (B, H, T, d_h)

        Q, K, V = map(split_heads, (Q, K, V))

        # 1. 构造位置编码
        device = x.device
        pos = torch.arange(T, device=device)  # (T,)
        pos = pos.unsqueeze(0).expand(B, T)  # (B, T) 或根据实际 batch 需求调整

        # 2. 应用 RoPE 到 Q, K
        Q = My_rope(d_h, theta=10000.0, max_seq_len=T,
                    in_query_or_key=Q, token_positions=pos)
        K = My_rope(d_h, theta=10000.0, max_seq_len=T,
                    in_query_or_key=K, token_positions=pos)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T, T)
        attn_out = my_scaled_dot_product_attention(Q, K, V, mask=mask)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(attn_out)


class My_transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
    ):
        super().__init__()
        self.embedding = My_embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                My_MultiHeadSelfAttentionwithROPE(d_model, num_heads),
                nn.LayerNorm(d_model),
                My_swiglu(d_model, d_ff),
                nn.LayerNorm(d_model),
            ]))
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        for attn, norm1, ff, norm2 in self.layers:
            # Self-attention + Add & Norm
            attn_out = attn(norm1(x), mask)
            x = x + attn_out
            # FFN + Add & Norm
            ff_out = ff(norm2(x))
            x = x + ff_out
        logits = self.out_proj(x)
        return logits
