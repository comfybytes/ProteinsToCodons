using Flux, BioSequences

include("Block.jl")

struct Encoder
    prot_embedder::Embedding
    pos_encoder::PositionEncoding
    attention_block::Block
    n_layers::Int
    dropout::Dropout
end

Flux.@functor Encoder

function Encoder(
    prot_alphabet,
    d_model::Int=4,
    d_hidden::Int=16,
    n_heads::Int=1,
    n_layers::Int=2,
    p_drop::Float64=0.1,
    activation=relu,
    max_len::Int=1000
)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))
    prot_len = length(prot_alphabet)
    Encoder(
        Embedding(prot_len => d_model),
        PositionEncoding(d_model, max_len),
        Block(d_model, d_hidden, n_heads, p_drop, activation),
        n_layers,
        Dropout(p_drop)
    )
end

function (e::Encoder)(prots::A) where {A<:AbstractArray}
    input = e.prot_embedder(prots)
    input = input .+ e.pos_encoder(input)
    input = e.dropout(input)
    for _ in 1:e.n_layers
        input = e.attention_block(input)
    end
    input
end
