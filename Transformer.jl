include("Encoder.jl")
include("Tokenizer.jl")
using Pkg; Pkg.develop(path= "/Users/julia/Studium/Thesis/seqdl/SeqDL")
using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences

cds_data = extractCDS("../datafiles/celegans.fasta")
tok = Tokenizer(cds_data)
peptides = cds_data.peptide
encoded = encode(tok,peptides)
decoded = decode(tok,encoded)
