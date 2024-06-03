include("Encoder.jl")
include("Tokenizer.jl")

using Pkg; Pkg.develop(path= "/Users/julia/Studium/Thesis/seqdl/SeqDL")
using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences

cds_data = extractCDS("../datafiles/celegans.fasta")
d_model = 512
len_aa_alphabet = length(cds_data.aa_alphabet)
peptides = cds_data.peptide[1:3]

tok = Tokenizer(cds_data)
emb = Embedding(len_aa_alphabet => d_model)
enc = Encoder()

tokenized = encode(tok,peptides)
seq_len, seq_num = size(tokenized)
display(tokenized)
embedded = emb(tokenized)
pos_encoded = embedded + positional_encoding(seq_len,seq_num,d_model)
encoded = enc(pos_encoded)
println(size(encoded))