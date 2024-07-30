include("Transformer.jl")

using Pkg;
Pkg.develop(path="./SeqDL");

using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences, Flux, CUDA, cuDNN, ProgressMeter


cds_data = extractCDS("./datafiles/celegans.fasta")

model = Transformer(cds_data.aa_alphabet, cds_data.nt_alphabet, 16, 32, 1, 6) |> gpu

opt_state = Flux.setup(Adam(0.001,(0.9, 0.98)), model) |> gpu

aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
dna_tokenizer = Tokenizer(cds_data.nt_alphabet)

peptides = aa_tokenizer(cds_data.peptide[1:5000])
x1_train, x1_test = Flux.splitobs(peptides, at=0.8)

dna = dna_tokenizer(cds_data.dna[1:5000])
x2_train, x2_test = Flux.splitobs(dna, at=0.8)

targets = target(cds_data.dna[1:5000], dna_tokenizer)
y_train, y_test = Flux.splitobs(targets, at=0.8)
x1_test = Array(x1_test) |> gpu
x2_test = Array(x2_test) |> gpu
y_test = Array(y_test) |> gpu

train_set = Flux.DataLoader((x1_train, x2_train, y_train) |> gpu, batchsize=64, shuffle=true)


es = Flux.early_stopping(Flux.logitcrossentropy, 4 , init_score = 1)

@showprogress desc="Training Model..." for epoch in 1:100
    Flux.train!(model, train_set, opt_state) do m, x1, x2, y
        Flux.logitcrossentropy(m(x1, x2), y)
    end
    if es(model(x1_test, x2_test), y_test)
        @info "Early Stopping at Epoch: $epoch"
        break
    end
end

#sequence = aa_tokenizer(cds_data.peptide[17])
#out = generate(sequence, model)
#out = dna_tokenizer(out)
#display(out)