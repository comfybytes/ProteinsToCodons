using Flux

"""
Creates an Attention Block, including Layer Normalization and a 2 layer Feed Forward Network
# Arguments
- `d_model`: dimensions of model. input and output must be this size. Default 240
- `n_heads`: number of attention heads. Default 4
- (optional) `activation`: Activation Function for Feed-Forward Network. Default ReLU

`d_model` must be divisible by `n_heads`.
"""
struct Block
    feed_forward1::Dense
    feed_forward2::Dense
    masked_mha::MultiHeadAttention
    mha::MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
    dropout::Dropout
end

Flux.@functor Block

function Block(
    d_model::Int=16,
    d_hidden::Int=32,
    n_heads::Int=1,
    p_drop::Float64=0.1,
    activation=relu,
)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))

    Block(
        Dense(d_model => d_hidden, activation),
        Dense(d_hidden => d_model, identity),
        MultiHeadAttention(d_model, nheads=n_heads, dropout_prob=p_drop),
        MultiHeadAttention(d_model, nheads=n_heads, dropout_prob=p_drop),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Dropout(p_drop)
    )
end

function (b::Block)(enc_context::A, context::A, mask::M=nothing) where {A<:AbstractArray,M<:Union{AbstractArray{Bool}, Nothing}}
    masked_attention, attention_score = b.masked_mha(context, mask=mask)
    masked_attention = b.norm1(masked_attention + context)
    attention, attention_score = b.mha(masked_attention, enc_context, enc_context)
    attention = b.norm2(attention + masked_attention)
    forward(attention, b)
end

function (b::Block)(context::A) where {A<:AbstractArray}
    attention, attention_score = b.mha(context)
    attention = b.norm2(attention + context)
    forward(attention, b)
end

function forward(attention, b::Block)
    ff_output = b.feed_forward1(attention)
    ff_output = b.feed_forward2(ff_output)
    ff_output = b.dropout(ff_output)
    b.norm2(ff_output + attention)
end
