include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")
include("Decoder.jl")

using Flux, BioSequences

struct Transformer
    d_model::Int
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
        d_model,
        Encoder(prot_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, activation, max_len),
        Decoder(prot_alphabet, dna_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, mask, activation, max_len),
        Dense(d_model => length(dna_alphabet))
    )
end

function (t::Transformer)(prots::Matrix{Int64}, dna::Matrix{Int64})
    enc_out = t.encoder(prots)
    dec_out = t.decoder(enc_out, dna)
    generate_logits(dec_out, t.Linear)
end

function (t::Transformer)(prots::Matrix{Int64})
    enc_out = t.encoder(prots)
    dec_out = t.decoder(enc_out)
    generate_logits(dec_out, t.Linear)
end

function generate(sequence::Vector{Int64}, model::Transformer)
    len_output = length(sequence) * 3
    sequence = reshape(sequence, size(sequence, 1), 1)
    enc_out = model.encoder(sequence)
    context = [5]
    output = []
    for i in 1:len_output
        context = model.decoder(enc_out, context)
        logit = generate_logits(context, model.Linear)
        push!(output,Int64(logit[1]))
    end
    output = reshape(output,size(output,1),1)
    convert(Matrix{Int64}, output)
end 