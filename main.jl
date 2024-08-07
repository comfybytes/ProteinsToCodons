include("Transformer.jl")
include("TransformerClassifier.jl")
using SeqDL, SeqDL.Data, SeqDL.Util
using BioSequences, Flux, ProgressMeter, JLD2, Dates
using CUDA, cuDNN

cds_data = get_cds("ecoli")

#model = Transformer(cds_data, 16, 32, 1, 1)
#model = train_model(model, cds_data, 1, true)
#save_model(model, "ecoli")


#model = load_model("ecoli_2024-08-05T18:08:39.869_32_64_2_4.jld2", cds_data)
#prediction = generate(cds_data.peptide[17], model, cds_data)
#display(cds_data.peptide[17])
#display(cds_data.dna[17])
#display(prediction)


model = TransformerClassifier(cds_data, 256, 512, 1, 6)
model = train_model(model, cds_data, 70, true)
prediction = generate(cds_data.peptide[17:19], model, cds_data)
#display(cds_data.peptide[3900])
display(cds_data.dna[17:19])
display(prediction)

#prediction = generate(cds_data.peptide[17], model, cds_data)
#display(prediction)

#prediction = generate(cds_data.peptide[18], model, cds_data)
#display(prediction)

#prediction = generate(cds_data.peptide[19], model, cds_data)
#display(prediction)