include("Transformer.jl")

using Pkg;
Pkg.develop(path="/Users/julia/Studium/Thesis/seqdl/SeqDL");

using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences, Flux


cds_data = extractCDS("../datafiles/celegans.fasta")

model = Transformer(cds_data.aa_alphabet, cds_data.nt_alphabet, 100, 200, 2, 6)
opt_state = Flux.setup(Adam(), model)

x_train = Tokenizer(cds_data.aa_alphabet)(cds_data.peptide[1:20])
y_train = Tokenizer(cds_data.nt_alphabet)(cds_data.dna[1:20])
train_set = Flux.DataLoader((x_train, y_train), batchsize=2)


for epoch in 1:20
    Flux.train!(model, train_set, opt_state) do m, x, y
        Flux.crossentropy(m(x, y), y)
    end
end