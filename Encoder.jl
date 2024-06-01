using Flux
#TODO add Dropout
include("Embedder.jl")
"""
Create an Encoder Layer
# Arguments
- `d_model`: dimensions of the entire d_model. Default 512
- `n_heads`: number of attention heads. Default 8
- (optional) `activation`: Activation Function for Feed-Forward Network

`d_model` must be divisible by `n_heads`.
"""
struct Encoder
    d_model::Int
    embedding::Embedding
    q_projection::Dense
    k_projection::Dense
    v_projection::Dense
    mha::MultiHeadAttention
    norm1::LayerNorm
    feed_forward1::Dense
    feed_forward2::Dense
    norm2::LayerNorm
end

function Encoder(
    d_model::Int=512,
    n_heads::Int=8,
    activation=relu)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))
    d_attention = convert(Int,(d_model/n_heads))
    return Encoder(
        d_model,
        Embedding(20 => d_model),
        Dense(d_model => d_attention,identity),
        Dense(d_model => d_attention,identity),
        Dense(d_model => d_attention,identity),
        MultiHeadAttention(d_attention => d_model => d_model, nheads=n_heads),
        LayerNorm(d_model),
        Dense(d_model => d_model*4,activation),
        Dense(d_model*4 => d_model,identity),
        LayerNorm(d_model)
    )
end

function (e::Encoder)(data) #TODO move embedding and positional_encoding to Embedder.jl
    len_data = length(data)
    data = e.embedding(data)
    data = data + positional_encoding(len_data,e.d_model)
    data = cat(data; dims = ndims(data) +1) #TODO remove when switching to actual data
    q = e.q_projection(data)
    k = e.k_projection(data)
    v = e.v_projection(data)
    attention_output, attention_score = e.mha(q,k,v)
    data = e.norm1(attention_output + data)
    data_ff = e.feed_forward1(data)
    data_ff = e.feed_forward2(data_ff)
    e.norm2(data_ff + data)
end
