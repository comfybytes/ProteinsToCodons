#include("Transformer.jl")
include("TransformerClassifier.jl")
include("Tokenizer.jl")
using SeqDL, SeqDL.Data, SeqDL.Util, SeqDL.DL
using BioSequences, Flux, ProgressMeter, JLD2, Dates, DataFrames, LatexPrint
using CUDA, cuDNN

struct ModelStruct
    model::TransformerClassifier
    generate::Function
end

struct Predi
    train_sids::Vector
    test_sids::Vector
    data
    model::ModelStruct
end


df = DataFrame()
for species in ["hsapiens"]
    cds_data = get_cds(species)
    split_index = Int(ceil(0.8 * length(cds_data.peptide)))

    range = 1:length(cds_data.peptide)
    test_sids = collect(range[split_index+1:end])
    train_sids = collect(range[1:split_index])
    
    data = SeqDL.DL.TransformerData(cds_data)
    for i in 1:10
        model = TransformerClassifier(cds_data, 20, 60, 2, 1)
        model = train_model(model, cds_data, 100, true)
        #save_model(model, "$(i)-$(species)")

        model_struct = ModelStruct(model,generate)
        pred = Predi(train_sids,test_sids,data,model_struct)
        row = SeqDL.Analysis.format(SeqDL.Analysis.ComparisonSummary(species,pred))
        push!(df, row)
        display(df)
    end
end

#=
df2 = DataFrame()
for species in ["athaliana","celegans","drerio","ecoli","hsapiens","scerevisiae"]
    cds_data = get_cds(species)
    split_index = Int(ceil(0.8 * length(cds_data.peptide)))

    range = 1:length(cds_data.peptide)
    test_sids = collect(range[split_index+1:end])
    train_sids = collect(range[1:split_index])
    
    data = SeqDL.DL.TransformerData(cds_data)
        model = load_model("$(species).jld2", cds_data)
        #model = train_model(model, cds_data, 100, true)
        #save_model(model, "$(i)-$(species)")

        model_struct = ModelStruct(model,generate)
        df = SeqDL.DL.collectCodons(test_sids, data, model_struct)
        rt = SeqDL.Analysis.ExtResultTable(df)
        s = SeqDL.Analysis.FullSummary(rt)
        pred = Predi(train_sids,test_sids,data,model_struct)
        row = SeqDL.Analysis.format(SeqDL.Analysis.ComparisonSummary(species,pred))
        push!(df2, row)
        display(s)
end
display(df2)
#ldf = df
#colnames = ["Species", "Type", "CDS",
#"\$a_{tr} \$", "\$e_{tr} \$", "\$a_{te} \$", "\$e_{te} \$", "\$\\Delta a_{te}\$"]
#rename!(ldf, colnames)
#println(tabular(ldf))


#model = load_model("ecoli_2024-08-05T18:08:39.869_32_64_2_4.jld2", cds_data)
#prediction = generate(cds_data.peptide[17], model, cds_data)                                                                                                                                                                                               
#display(cds_data.peptide[17])
#display(cds_data.dna[17])
#display(prediction)



#model = train_model(model, cds_data, 100, true)
#prediction = generate(cds_data.peptide[17:19], model, cds_data)






#, e = SeqDL.DL.getOutput(sids,cds_data,model_struct)
#df = SeqDL.DL.collectCodons(test_sids, data, model_struct)
#df = DataFrame(t = t, e = e)
#rt = SeqDL.Analysis.ExtResultTable(df)
#display(rt)
#s = SeqDL.Analysis.FullSummary(rt)
#display(s)


#SeqDL.DL.showResults(sids[1:3], data, model_struct)
=#