using Flux

"""
Create an Encoder Layer
# Arguments
- `d_model`: dimensions of model. input and output must be this size. Default 240
- `n_heads`: number of attention heads. Default 4
- (optional) `activation`: Activation Function for Feed-Forward Network. Default ReLU

`d_model` must be divisible by `n_heads`.
"""
struct Encoder
    feed_forward1::Dense
    feed_forward2::Dense
    mha::MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
    drop_out::Dropout
end

function Encoder(
    d_model::Int=240,
    d_hidden::Int=480,
    n_heads::Int=4,
    p_drop::Float64=0.1,
    activation=relu,
    )

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))

    return Encoder(
        Dense(d_model => d_hidden, activation),
        Dense(d_hidden => d_model, identity),
        MultiHeadAttention(d_model => d_model * n_heads => d_model, nheads=n_heads, dropout_prob=p_drop),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Dropout(p_drop)
    )
end

function (e::Encoder)(data::Array{Float32,3})
    attention, attention_score = e.mha(data)
    attention = e.drop_out(attention)
    attention = e.norm1(attention .+ data)
    ff_output = e.feed_forward1(attention)
    ff_output = e.feed_forward2(ff_output)
    ff_output = e.drop_out(ff_output)
    e.norm2(ff_output + attention)
end