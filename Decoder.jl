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

function (decoder::Decoder)(encoder_data::Array{Float32,3}, previous, mask)
    masked_attention, attention_score = decoder.masked_mha(previous,mask=mask)
    norm_masked_attention = decoder.norm1(masked_attention .+ previous)
    attention, attention_score = decoder.mha(norm_masked_attention,encoder_data,encoder_data)
    norm_attention = decoder.norm2(attention .+ norm_masked_attention)
    ff_output = decoder.feed_forward1(norm_attention)
    ff_output = decoder.feed_forward2(ff_output)
    decoder.norm3(ff_output + norm_attention)
end