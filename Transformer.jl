include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")
include("Decoder.jl")

using Flux, BioSequences, SeqDL, SeqDL.Data, SeqDL.Util

struct Transformer
    in_embedder::Embedding
    out_embedder::Embedding
    pos_encoder::PositionEncoding
    encoder::Encoder
    decoder::Decoder
    drop_out::Dropout
    linear::Dense
end

Flux.@functor Transformer

function Transformer(
    in_alphabet,
    out_alphabet,
    d_model::Int=240,
    d_hidden::Int=480,
    n_heads::Int=4,
    p_drop::Float64=0.1,
    max_len::Int=1000
)

    in_len = length(in_alphabet)
    out_len = length(out_alphabet)
    Transformer(
        Embedding(in_len => d_model),
        Embedding(out_len => d_model),
        PositionEncoding(d_model, max_len),
        Encoder(d_model, d_hidden, n_heads, p_drop),
        Decoder(d_model, d_hidden, n_heads, p_drop),
        Dropout(p_drop),
        Dense(d_model => out_len, identity)
    )
end

function (t::Transformer)(peptides::Matrix{Int64}, dna::Matrix{Int64}, n_layers::Int=1)
    input = t.in_embedder(peptides)
    input = input .+ t.pos_encoder(input)
    input = t.drop_out(input)

    for _ in 1:n_layers
        input = t.encoder(input)
    end

    output = t.out_embedder(dna)
    output = output .+ t.pos_encoder(output)
    output = t.drop_out(output)

    mask = make_causal_mask(output)

    for _ in 1:n_layers
        output = t.decoder(input, output, mask)
    end

    logits = t.linear(output)
    logits = softmax(logits)
    logits = map(x -> x[1], argmax(logits, dims=1))
    reshape(logits, (size(logits, 2), size(logits, 3)))
end