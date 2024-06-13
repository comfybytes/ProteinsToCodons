using Pkg;
Pkg.develop(path="/Users/julia/Studium/Thesis/seqdl/SeqDL");

using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences, Flux
include("Transformer.jl")

cds_data = extractCDS("../datafiles/celegans.fasta")



tf = Transformer(cds_data.aa_alphabet, cds_data.nt_alphabet, 240, 480, 4)

tf(cds_data.peptide[1:4], cds_data.dna[1:4], 1)