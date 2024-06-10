using Flux

struct Decoder
    feed_forward1::Dense
    feed_forward2::Dense
    masked_mha::MultiHeadAttention
    mha::MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
    norm3::LayerNorm
end

function Decoder(
    d_model::Int=240,
    d_inner::Int=480,
    n_heads::Int=4,
    activation=relu)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))

    return Decoder(
        Dense(d_model => d_inner, activation),
        Dense(d_inner => d_model, identity),
        MultiHeadAttention(d_model => d_model * n_heads => d_model, nheads=n_heads),
        MultiHeadAttention(d_model => d_model * n_heads => d_model, nheads=n_heads),
        LayerNorm(d_model),
        LayerNorm(d_model),
        LayerNorm(d_model)
    )
end

function (decoder::Decoder)(data::Array{Float32,3}, previous_output, mask=nothing)

end