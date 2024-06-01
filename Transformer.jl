include("Encoder.jl")
include("Tokenizer.jl")
using Pkg; Pkg.develop(path= "/Users/julia/Studium/Thesis/seqdl/SeqDL")
using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences

cds_data = extractCDS("../datafiles/celegans.fasta")
tok = Tokenizer(cds_data)
enc = Encoder(cds_data)
peptides = cds_data.peptide[1:3]
encoded = encode(tok,peptides)
enc(encoded)