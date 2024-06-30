include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")
include("Decoder.jl")

using Flux, BioSequences

struct Transformer
    encoder::Encoder
    decoder::Decoder
    Linear::Dense
end

Flux.@functor Transformer

function Transformer(
    prot_alphabet,
    dna_alphabet,
    d_model::Int=240,
    d_hidden::Int=480,
    n_heads::Int=4,
    n_layers::Int=3,
    p_drop::Float64=0.1,
    mask::Bool=true,
    activation=relu,
    max_len::Int=1000,
)

    Transformer(
        Encoder(prot_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, activation, max_len),
        Decoder(prot_alphabet, dna_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, mask, activation, max_len),
        Dense(d_model => length(dna_alphabet))
    )
end

function (t::Transformer)(prots::Matrix{Int64}, dna::Matrix{Int64})
    enc_out = t.encoder(prots)
    dec_out = t.decoder(enc_out, dna)
    generate(dec_out, t.Linear)
end

function (t::Transformer)(prots::Matrix{Int64})
    enc_out = t.encoder(prots)
    dec_out = t.decoder(enc_out)
    generate(dec_out, t.Linear)
end
