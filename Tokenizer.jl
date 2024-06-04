using BioSequences
"""
Create a Tokenizer
# Arguments
- `cds_data`: CDSData Custom Type
"""
struct Tokenizer
    aa_alphabet::Vector{AminoAcid}
    lookup_encode::Dict{AminoAcid, Int64}
    lookup_decode::Dict{Int64,AminoAcid}
end

function Tokenizer(
    cds_data)
    len = length(cds_data.aa_alphabet)
    ints = collect(1:len)
    lookup_encode = Dict(zip(cds_data.aa_alphabet,ints))
    lookup_decode = Dict(zip(ints,cds_data.aa_alphabet))
    return Tokenizer(
        cds_data.aa_alphabet,
        lookup_encode,
        lookup_decode
    )
end

"""
Encodes a Vector{LongAA} into Matrix{AminoAcid}
# Arguments
- `t::Tokenizer`: 
- `aa`: cds_data.peptide
"""
function encode(t::Tokenizer,aa_matrix::Matrix{AminoAcid})
    map(x -> get(t.lookup_encode,x,0),aa_matrix)
end

function encode(t::Tokenizer,aa::LongAA)
    map(x -> get(t.lookup_encode,x,0),collect(aa))
end

function encode(t::Tokenizer,aa::Vector{LongAA})
    aa_matrix = hcat(map(seq -> collect(seq),aa)...)
    encode(t,aa_matrix)
end

function decode(t::Tokenizer,aa_matrix::Matrix{Int64})
    map(x -> get(t.lookup_decode,x,nothing),aa_matrix)
end

function positional_encoding(seq_len::d,seq_num::d,d_model::d) where {d<:Integer}
    encoding = fill(0.0,(d_model,seq_len,seq_num))
    encoding = convert(Array{Float32,3},encoding)
    for batch in 1:seq_num
        for pos in 1:seq_len
            for i in 1:2:(d_model-1)
                term = pos/10000^(2i/d_model)
                encoding[i,pos,batch] = sin(term)
                encoding[i+1,pos,batch] = cos(term)
            end
        end
    end
    return encoding
end