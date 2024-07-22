include("Transformer.jl")

using Pkg;
Pkg.develop(path="./SeqDL");

using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences, Flux, CUDA, cuDNN


cds_data = extractCDS("./datafiles/celegans.fasta")

model = Transformer(cds_data.aa_alphabet, cds_data.nt_alphabet, 4, 16, 1, 2) |> gpu

opt_state = Flux.setup(Adam(), model)


aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
dna_tokenizer = Tokenizer(cds_data.nt_alphabet)

x_train = aa_tokenizer(cds_data.peptide[1:4])
y_train = dna_tokenizer(cds_data.dna[1:4])
train_set = Flux.DataLoader((x_train, y_train), batchsize=2)

for epoch in 1:500
    println("Epoch: $(epoch)")
    Flux.train!(model, train_set, opt_state) do m, x, y
        Flux.crossentropy(m(x, y), y)
    end
end

sequence = aa_tokenizer(cds_data.peptide[17])
out = generate(sequence, model)
out = dna_tokenizer(out)
display(out)