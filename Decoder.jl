using Flux, BioSequences

include("Block.jl")

struct Decoder
    prot_embedder::Embedding
    dna_embedder::Embedding
    pos_encoder::PositionEncoding
    attention_block::Block
    linear::Dense
    n_layers::Int
    mask::Bool
    dropout::Dropout
end

Flux.@functor Decoder

function Decoder(
    prot_alphabet,
    dna_alphabet,
    d_model::Int=16,
    d_hidden::Int=32,
    n_heads::Int=1,
    n_layers::Int=2,
    p_drop::Float64=0.1,
    mask::Bool=true,
    activation=relu,
    max_len::Int=1000
)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))
    prot_len = length(prot_alphabet)
    dna_len = length(dna_alphabet)+1

    Decoder(
        Embedding(prot_len => d_model),
        Embedding(dna_len => d_model),
        PositionEncoding(d_model, max_len),
        Block(d_model, d_hidden, n_heads, p_drop, activation),
        Dense(d_model => dna_len),
        n_layers,
        mask,
        Dropout(p_drop)
    )
end

function (d::Decoder)(enc_context::A, context::M, mask::Bool) where {A<:AbstractArray,M<:AbstractMatrix} # Function For Training
    context = d.dna_embedder(context)
    context = context .+ d.pos_encoder(context)
    context = d.dropout(context)

    mask = make_causal_mask(context)
    input = context
    for _ in 1:d.n_layers
        input = context
        input = d.attention_block(enc_context, input, mask)
    end
    input
end

function (d::Decoder)(enc_context::A, context::A) where {A<:AbstractArray} # Function For Inference
    context = d.dna_embedder(context)
    context = context .+ d.pos_encoder(context)
    context = d.dropout(context)
    for _ in 1:d.n_layers
        context = d.attention_block(enc_context, context)
    end
    context
end