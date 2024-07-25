include("Transformer.jl")

using Pkg;
Pkg.develop(path="./SeqDL");

using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences, Flux, CUDA, cuDNN, ProgressMeter


cds_data = extractCDS("./datafiles/celegans.fasta")

model = Transformer(cds_data.aa_alphabet, cds_data.nt_alphabet, 16, 32, 1, 6) |> gpu

opt_state = Flux.setup(Adam(0.0001,(0.9, 0.98)), model) |> gpu


aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
dna_tokenizer = Tokenizer(cds_data.nt_alphabet)

x_train = aa_tokenizer(cds_data.peptide[1:4096])
y_train = dna_tokenizer(cds_data.dna[1:4096])
y_target = target(cds_data.dna[1:4096],dna_tokenizer)

train_set = Flux.DataLoader((x_train, y_train, y_target) |> gpu, batchsize=64)

@showprogress desc="Training Model..." for epoch in 1:100
    Flux.train!(model, train_set, opt_state) do m, x, y, z
        Flux.logitcrossentropy(m(x, y), z)
    end
end

#sequence = aa_tokenizer(cds_data.peptide[17])
#out = generate(sequence, model)
#out = dna_tokenizer(out)
#display(out)