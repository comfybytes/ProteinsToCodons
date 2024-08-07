include("Transformer.jl")
include("TransformerClassifier.jl")
using SeqDL, SeqDL.Data, SeqDL.Util
using BioSequences, Flux, ProgressMeter, JLD2, Dates
using CUDA, cuDNN

cds_data = read_cds("ecoli")
#display(size(cds_data.peptide[1]))

#model = Transformer(cds_data, 64, 128, 2, 6)
#model = train_model(model, cds_data, 70, true)
#save_model(model, "ecoli")


#model = load_model("ecoli_2024-08-05T18:08:39.869_32_64_2_4.jld2", cds_data)
#prediction = generate(cds_data.peptide[17], model, cds_data)
#display(cds_data.peptide[17])
#display(cds_data.dna[17])
#display(prediction)

model = TransformerClassifier(cds_data, 64, 128, 2, 6)
train_model(model, cds_data, 70, true)