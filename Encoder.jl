using Flux, BioSequences

include("Block.jl")

"""
    Encoder(prot_alphabet, d_model::Int, d_hidden::Int, n_heads::Int, n_layers::Int, p_drop::Float64, max_len::Int)

Creates an Encoder which analyzes a matrix or a vector of tokenized amino acid sequences.
Performs Embedding into a High-Dimensional Vector Space.
Creates and processes multiple sequential Encoders by creating multiple Blocks.
Returns a context, which can be used by the Decoder or can be independetly used in the Encoder-only architecture.

# Arguments
- `prot_alphabet`: alphabet used for creating an Embedding Layer, usually amino acids.
- `d_model`: dimensions of model. input and output must be this size. Default 256
- `d_hidden`: dimensions of the hidden layer in the feed-forward network. Default 1024
- `n_heads`: number of attention heads. Must equally divide `d_model`. Default 2
- `n_layers`: number of sequential Encoders. Default 2
- `p_drop`: probability for dropout. Default 0.1
- `max_len`: maximum sequence length. Default 1000
"""

struct Encoder
    prot_embedder::Embedding
    pos_encoder::PositionEncoding
    attention_blocks::Vector{Block}
    n_layers::Int
    dropout::Dropout
end

Flux.@functor Encoder

function Encoder(
    prot_alphabet,
    d_model::Int=256,
    d_hidden::Int=1024,
    n_heads::Int=2,
    n_layers::Int=2,
    p_drop::Float64=0.1,
    max_len::Int=1000
)

    prot_len = length(prot_alphabet)
    blocks = fill(Block(d_model, d_hidden, n_heads, p_drop),n_layers)

    Encoder(
        Embedding(prot_len => d_model),
        PositionEncoding(d_model, max_len),
        blocks,
        n_layers,
        Dropout(p_drop)
    )
end

# Encoder 
function (e::Encoder)(prots::A) where {A<:AbstractArray}
    context = e.prot_embedder(prots)
    context = context .+ e.pos_encoder(context)
    context = e.dropout(context)
    for block in e.attention_blocks
        context = block(context)
    end
    context
end
