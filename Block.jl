using Flux

"""
    Block(d_model::Int, d_hidden::Int, n_heads, p_drop::Float64)

Creates an Attention Block used by Encoder and Decoder, including Multi-Head-Attention, Layer Normalization and a Feed-Forward Network.
Feed-Forward Network consists of 2 Dense Layers, first one with ReLU activation, second one without activation.

# Arguments

- `d_model`: dimensions of model. input and output must be this size. Default 256
- `d_hidden`: dimensions of the hidden layer in the feed-forward network. Default 1024
- `n_heads`: number of attention heads. Must equally divide `d_model`. Default 2
- `p_drop`: probability for dropout. Default 0.1
"""

struct Block
    feed_forward1::Dense
    feed_forward2::Dense
    masked_mha::MultiHeadAttention
    mha::MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
    norm3::LayerNorm
    dropout::Dropout
end

Flux.@functor Block

function Block(
    d_model::Int=256,
    d_hidden::Int=1024,
    n_heads::Int=2,
    p_drop::Float64=0.1,
)

    Block(
        Dense(d_model => d_hidden, relu),
        Dense(d_hidden => d_model, identity),
        MultiHeadAttention(d_model, nheads=n_heads, dropout_prob=p_drop),
        MultiHeadAttention(d_model, nheads=n_heads, dropout_prob=p_drop),
        LayerNorm(d_model),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Dropout(p_drop)
    )
end

# Block for Decoder includes Masked Attention and Cross Attention
function (b::Block)(enc_context::A, context::A, mask::M=nothing) where {A<:AbstractArray,M<:Union{AbstractArray{Bool}, Nothing}}
    masked_attention = b.norm1(context)
    masked_attention, attention_score = b.masked_mha(masked_attention, mask=mask)
    masked_attention = masked_attention .+ context
    attention = b.norm2(masked_attention)
    attention, attention_score = b.mha(attention, attention, enc_context)
    attention = attention .+ masked_attention
    forward(attention, b)
end

# Block for Encoder includes Self Attention
function (b::Block)(context::A) where {A<:AbstractArray}
    attention = b.norm2(context)
    attention, attention_score = b.mha(attention)
    attention = attention .+ context
    forward(attention, b)
end

# Feed-Forward-Network
function forward(attention, b::Block)
    ff_output = b.norm3(attention)
    ff_output = b.feed_forward1(ff_output)
    ff_output = b.feed_forward2(ff_output)
    ff_output = b.dropout(ff_output)
    ff_output = ff_output .+ attention
end
