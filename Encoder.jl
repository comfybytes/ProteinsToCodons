using Flux

#TODO add Dropout
"""
Create an Encoder Layer
# Arguments
- `d_model`: dimensions of the entire d_model. Default 512
- `n_heads`: number of attention heads. Default 8
- (optional) `activation`: Activation Function for Feed-Forward Network

`d_model` must be divisible by `n_heads`.
"""
struct Encoder
    feed_forward1::Dense
    feed_forward2::Dense
    mha::MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
end

function Encoder(
    d_model::Int=240,
    d_hidden::Int=480,
    n_heads::Int=4,
    activation=relu)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))

    return Encoder(
        Dense(d_model => d_hidden, activation),
        Dense(d_hidden => d_model, identity),
        MultiHeadAttention(d_model => d_model * n_heads => d_model, nheads=n_heads),
        LayerNorm(d_model),
        LayerNorm(d_model)
    )
end

function (encoder::Encoder)(data::Array{Float32,3})
    attention_output, attention_score = encoder.mha(data)
    data = encoder.norm1(attention_output .+ data)
    data_ff = encoder.feed_forward1(data)
    data_ff = encoder.feed_forward2(data_ff)
    encoder.norm2(data_ff + data)
end