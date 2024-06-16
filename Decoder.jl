using Flux

struct Decoder
    feed_forward1::Dense
    feed_forward2::Dense
    masked_mha::MultiHeadAttention
    mha::MultiHeadAttention
    norm1::LayerNorm
    norm2::LayerNorm
    norm3::LayerNorm
    drop_out::Dropout
end

Flux.@functor Decoder

function Decoder(
    d_model::Int=240,
    d_inner::Int=480,
    n_heads::Int=4,
    p_drop::Float64=0.1,
    activation=relu
    )

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))

    Decoder(
        Dense(d_model => d_inner, activation),
        Dense(d_inner => d_model, identity),
        MultiHeadAttention(d_model => d_model * n_heads => d_model, nheads=n_heads, dropout_prob=p_drop),
        MultiHeadAttention(d_model => d_model * n_heads => d_model, nheads=n_heads, dropout_prob=p_drop),
        LayerNorm(d_model),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Dropout(p_drop)
    )
end

function (d::Decoder)(encoder_data, previous, mask)
    masked_attention, attention_score = d.masked_mha(previous, mask=mask)
    masked_attention = d.drop_out(masked_attention)
    norm_masked_attention = d.norm1(masked_attention + previous)
    attention, attention_score = d.mha(norm_masked_attention, encoder_data, encoder_data)
    attention = d.drop_out(attention)
    attention = d.norm2(attention + norm_masked_attention)
    ff_output = d.feed_forward1(attention)
    ff_output = d.feed_forward2(ff_output)
    ff_output = d.drop_out(ff_output)
    d.norm3(ff_output + attention)
end