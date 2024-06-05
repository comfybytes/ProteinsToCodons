using Flux

struct Decoder
    q_projection::Dense
    k_projection::Dense
    v_projection::Dense
    masked_q_projection::Dense
    masked_k_projection::Dense
    masked_v_projection::Dense
    feed_forward1::Dense
    feed_forward2::Dense
    masked_mha::MultiHeadAttention
    mha::MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
    norm3::LayerNorm
end

function Decoder(
    d_model::Int=512,
    d_inner::Int=2048,
    n_heads::Int=8,
    activation=relu)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))
    d_attention = convert(Int, (d_model / n_heads))

    return Decoder(
        Dense(d_model => d_attention, identity),
        Dense(d_model => d_attention, identity),
        Dense(d_model => d_attention, identity),
        Dense(d_model => d_attention, identity),
        Dense(d_model => d_attention, identity),
        Dense(d_model => d_attention, identity),
        Dense(d_model => d_inner, activation),
        Dense(d_inner => d_model, identity),
        MultiHeadAttention(d_attention => d_model => d_model, nheads=n_heads),
        MultiHeadAttention(d_attention => d_model => d_model, nheads=n_heads),
        LayerNorm(d_model),
        LayerNorm(d_model),
        LayerNorm(d_model)
    )
end

function (decoder::Decoder)(data::Array{Float32,3}, mask)

end