using BioSequences, Flux, LinearAlgebra

"""
    Tokenizer(alphabet)

Creates a Tokenizer, which transforms sequences of amino acids or DNA into matrices of positive integers.
Also provides a special CodonTokenizer with a predefined alphabet for codons.
The Tokenizer saves two dictionaries, one for encoding to positive integers and one for decoding to amino acids or DNA.
Additionally a dictionary for one hot encoding is saved, used for DNA during the model training process.

# Arguments
- `alphabet`: a vector of unique tokens within the expected sequences
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

function (tok::Tokenizer)(matrix::T) where {T<:Union{Matrix{DNA},Matrix{AminoAcid},Matrix{LongSequence{DNAAlphabet{4}}}}}
    map(t -> get(tok.lookup_encode, t, 0), matrix)
end

function (tok::Tokenizer)(seq::LongSequence)
    map(t -> get(tok.lookup_encode, t, 0), collect(seq))
end

function (tok::Tokenizer)(vec::T) where {T<:Union{Vector{LongDNA},Vector{LongAA},Vector{Vector{LongSequence{DNAAlphabet{4}}}}}}
    matrix = hcat(map(t -> collect(t), vec)...)
    tok(matrix)
end

function (tok::Tokenizer)(array::T) where {T<:Union{Matrix{Int64},Vector{Int64}}}
    map(t -> get(tok.lookup_decode, t, 0), array)
end

# creates a one hot encoding for DNA sequences used during training
function target(vec::T, tok::Tokenizer) where {T<:Union{Vector{LongDNA},Vector{Vector{LongSequence{DNAAlphabet{4}}}}}}
    matrix = hcat(map(t -> collect(t), vec)...)
    dim_1 = size(matrix, 1)
    dim_2 = size(matrix, 2)
    matrix = map(t -> get(tok.onehot_encode, t, 0), matrix)
    matrix = hcat(matrix...)
    matrix = reshape(matrix, length(tok.onehot_encode), dim_1, dim_2)
end

# CodonTokenizer used for Encoder-only Transformer
function CodonTokenizer()
    codons = [
        dna"TTT",dna"TTC",dna"TTA",dna"TTG",
        dna"TCT",dna"TCC",dna"TCA",dna"TCG",
        dna"TAT",dna"TAC",dna"TAA",dna"TAG",
        dna"TGT",dna"TGC",dna"TGA",dna"TGG",
        dna"CTT",dna"CTC",dna"CTA",dna"CTG",
        dna"CCT",dna"CCC",dna"CCA",dna"CCG",
        dna"CAT",dna"CAC",dna"CAA",dna"CAG",
        dna"CGT",dna"CGC",dna"CGA",dna"CGG",
        dna"ATT",dna"ATC",dna"ATA",dna"ATG",
        dna"ACT",dna"ACC",dna"ACA",dna"ACG",
        dna"AAT",dna"AAC",dna"AAA",dna"AAG",
        dna"AGT",dna"AGC",dna"AGA",dna"AGG",
        dna"GTT",dna"GTC",dna"GTA",dna"GTG",
        dna"GCT",dna"GCC",dna"GCA",dna"GCG",
        dna"GAT",dna"GAC",dna"GAA",dna"GAG",
        dna"GGT",dna"GGC",dna"GGA",dna"GGG"]
    
    Tokenizer(codons)
end

# seperates DNA sequences into groups of three to create codons
function seq_to_codons(dna::Vector{LongDNA})
    map(seq -> [seq[i:i+2] for i in 1:3:length(seq)],dna)
end

