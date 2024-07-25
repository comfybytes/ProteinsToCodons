include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")
include("Decoder.jl")

using Flux, BioSequences, CUDA, cuDNN

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
    d_model::Int=16,
    d_hidden::Int=32,
    n_heads::Int=1,
    n_layers::Int=2,
    p_drop::Float64=0.1,
    mask::Bool=true,
    activation=relu,
    max_len::Int=1000,
)

    Transformer(
        d_model,
        Encoder(prot_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, activation, max_len),
        Decoder(prot_alphabet, dna_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, mask, activation, max_len),
        Dense(d_model => length(dna_alphabet), identity)
    )
end

function (t::Transformer)(prots::A, dna::A) where {A<:AbstractArray} # Function For Training
    enc_context = t.encoder(prots)
    dec_out = t.decoder(enc_context, dna, true)
    t.Linear(dec_out)
end

function generate(sequence::A, model::Transformer) where {A<:AbstractArray} # Function For Inference
    len_output = length(sequence) * 3
    sequence = reshape(sequence, size(sequence, 1), 1)
    output = [5]

    enc_out = model.encoder(sequence)
    for i in 1:len_output
        context = reshape(output,size(output,1),1)
        pre_logit = model.decoder(enc_out, context)
        logit = logits(pre_logit, model.Linear)
        push!(output,Int64(logit[1]))
    end
    popfirst!(output)
    output = reshape(output,size(output,1),1)
    convert(Matrix{Int64}, output) #change
end