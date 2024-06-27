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
    d_model::Int=240,
    d_hidden::Int=480,
    n_heads::Int=4,
    n_layers::Int=3,
    p_drop::Float64=0.1,
    mask::Bool=false,
    activation=relu,
    max_len::Int=1000
)

    d_model % n_heads == 0 || throw(ArgumentError("d_model = $(d_model) should be divisible by nheads = $(n_heads)"))
    prot_len = length(prot_alphabet)
    dna_len = length(dna_alphabet)

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

function (d::Decoder)(input::Array{Float32, 3}, dna::Matrix{Int64})
    context = d.dna_embedder(dna)
    context = context .+ d.pos_encoder(context)
    context = d.dropout(context)

    mask = d.mask ? make_causal_mask(context) : nothing

    for _ in 1:d.n_layers
        context = d.attention_block(input, context, mask)
    end
    context
end

function (d::Decoder)(prots::Matrix{Int64})
    context = d.prot_embedder(prots)
    context = context .+ d.pos_encoder(context)
    context = d.dropout(context)

    mask = d.mask ? make_causal_mask(context) : nothing

    for _ in 1:d.n_layers
        context = d.attention_block(context, mask)
    end
    context
end

#function generate()
#    logits = t.linear(input)
#    logits = softmax(logits)
#    logits = map(x -> x[1], argmax(logits, dims=1))
#    reshape(logits, (size(logits, 2), size(logits, 3)))
#end