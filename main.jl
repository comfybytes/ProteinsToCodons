include("Transformer.jl")
include("TransformerClassifier.jl")
using SeqDL, SeqDL.Data, SeqDL.Util, SeqDL.DL
using BioSequences, Flux, ProgressMeter, JLD2, Dates, DataFrames
using CUDA, cuDNN

cds_data = get_cds("celegans")

#model = Transformer(cds_data, 16, 32, 1, 1)
#model = train_model(model, cds_data, 1, true)
#save_model(model, "ecoli")


#model = load_model("ecoli_2024-08-05T18:08:39.869_32_64_2_4.jld2", cds_data)
#prediction = generate(cds_data.peptide[17], model, cds_data)                                                                                                                                                                                               
#display(cds_data.peptide[17])
#display(cds_data.dna[17])
#display(prediction)

#=
model = TransformerClassifier(cds_data, 256, 1024, 4, 1)
model = train_model(model, cds_data, 300, true)
#prediction = generate(cds_data.peptide[17:19], model, cds_data)


split_index = Int(ceil(0.8 * length(cds_data.peptide)))

range = 1:length(cds_data.peptide)
sids = collect(range[split_index+1:end])


struct ModelStruct
    model::TransformerClassifier
    generate::Function
end

model_struct = ModelStruct(model,generate)
data = SeqDL.DL.TransformerData(cds_data)
#, e = SeqDL.DL.getOutput(sids,cds_data,model_struct)
df = SeqDL.DL.collectCodons(sids[1:3],data, model_struct)
#df = DataFrame(t = t, e = e)
rt = SeqDL.Analysis.ExtResultTable(df)
#display(rt)
s = SeqDL.Analysis.FullSummary(rt)
display(s)


#SeqDL.DL.showResults(sids[1:3], data, model_struct)
=#