using BioSequences
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
end

function Tokenizer(alphabet)

    ints = collect(1:length(alphabet))
    lookup_encode = Dict(zip(alphabet, ints))
    lookup_decode = Dict(zip(ints, alphabet))

    Tokenizer(lookup_encode,lookup_decode)
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
    map(t -> get(tok.lookup_decode, t, nothing), matrix)
end
