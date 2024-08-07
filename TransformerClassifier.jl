include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")

using Flux, BioSequences, CUDA, cuDNN, Dates, ProgressMeter
using SeqDL, SeqDL.Data, SeqDL.Util

struct TransformerClassifier
    d_model::Int
    d_hidden::Int
    n_heads::Int
    n_layers::Int
    p_drop::Float64
    max_len::Int
    encoder::Encoder
    linear::Dense
end

Flux.@functor TransformerClassifier

function TransformerClassifier(
    cds_data::CDSData,
    d_model::Int=16,
    d_hidden::Int=32,
    n_heads::Int=1,
    n_layers::Int=2,
    p_drop::Float64=0.1,
    max_len::Int=1000
)

    TransformerClassifier(
        d_model,
        d_hidden,
        n_heads,
        n_layers,
        p_drop,
        max_len,
        Encoder(cds_data.aa_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, max_len),
        Dense(d_model => 64, identity)
    )
end

function (t::TransformerClassifier)(prots::A) where {A<:AbstractArray} # Function For Training
    context = t.encoder(prots)
    t.linear(context)
end

function train_model(model::TransformerClassifier, cds_data::CDSData, epochs::Int=100, usegpu::Bool=true)
    @info "Preparing model for training"
    device = usegpu ? gpu : cpu
    CUDA.device_reset!()

    model = model |> device
    opt_state = Flux.setup(Adam(0.001,(0.9, 0.98)), model) |> device

    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    cd_tokenizer = CodonTokenizer()

    peptides = aa_tokenizer(cds_data.peptide)
    x_train, x_test = Flux.splitobs(peptides, at=0.8)
    
    targets = target(seq_to_codons(cds_data.dna), cd_tokenizer)
    y_train, y_test = Flux.splitobs(targets, at=0.8)

    train_set = Flux.DataLoader((x_train, y_train) |> device, batchsize=128, shuffle=true)
    x1_test = Array(x_test) |> device
    y_test = Array(y_test) |> device

    #es = Flux.early_stopping(Flux.logitcrossentropy, 7 , init_score = 10)

    @showprogress desc="Training Model..." for epoch in 1:epochs
        Flux.train!(model, train_set, opt_state) do m, x, y
            loss = Flux.logitcrossentropy(m(x), y)
        end
        #if es(model(x_test), y_test)
        #    @info "Stopped training earlier at Epoch: $epoch out of $epochs due to increasing Loss on test set"
        #    break
        #end
    end
    @info "Finished training model"
    return model
end

function seq_to_codons(dna::Vector{LongDNA})
    map(seq -> [seq[i:i+2] for i in 1:3:length(seq)],dna)
end
