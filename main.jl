include("Transformer.jl")
using SeqDL, SeqDL.Data, SeqDL.Util
using BioSequences, Flux, ProgressMeter, JLD2, Dates
using CUDA, cuDNN
using Pkg;
Pkg.develop(path="./SeqDL");

cds_data = read_cds("ecoli")
model = Transformer(cds_data, 16, 32, 1, 4)
model = train_model(model, cds_data, 100, true)
#save_model(model, "ecoli")

prediction = generate(cds_data.peptide[17], model, cds_data)
display(prediction)
display(cds_data.dna[17])
#model = load_model("celegans_2024-07-31T20:24:30.485_8_16_1_1", cds_data)