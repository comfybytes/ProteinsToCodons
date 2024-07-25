using BioSequences, Flux, LinearAlgebra
"""
Create a Tokenizer
# Arguments
- `alphabet`: a vector of symbols used for the Tokenizer
# Usage
WIP
"""
struct Tokenizer
    lookup_encode::Dict
    lookup_decode::Dict
    onehot_encode::Dict
end

function Tokenizer(alphabet)
    ints = collect(1:length(alphabet))
    identity = Matrix{Int}(I, length(alphabet), length(alphabet))
    lookup_encode = Dict(zip(alphabet, ints))
    lookup_decode = Dict(zip(ints, alphabet))
    onehot_encode = Dict(zip(alphabet,eachrow(identity)))

    Tokenizer(lookup_encode,lookup_decode,onehot_encode)
end

function (tok::Tokenizer)(matrix::T) where {T<:Union{Matrix{DNA},Matrix{AminoAcid}}}
    map(t -> get(tok.lookup_encode, t, 0), matrix)
end

function (tok::Tokenizer)(seq::LongSequence)
    map(t -> get(tok.lookup_encode, t, 0), collect(seq))
end

function (tok::Tokenizer)(vec::T) where {T<:Union{Vector{LongDNA},Vector{LongAA}}}
    matrix = hcat(map(t -> collect(t), vec)...)
    tok(matrix)
end

function (tok::Tokenizer)(matrix::Matrix{Int64})
    map(t -> get(tok.lookup_decode, t, 0), matrix)
end

function target(vec::Vector{LongDNA}, tok::Tokenizer)
    matrix = hcat(map(t -> collect(t), vec)...)
    dim_1 = size(matrix, 1)
    dim_2 = size(matrix, 2)
    matrix = map(t -> get(tok.onehot_encode, t, 0), matrix)
    matrix = hcat(matrix...)
    matrix = reshape(matrix, length(tok.onehot_encode), dim_1, dim_2)
end