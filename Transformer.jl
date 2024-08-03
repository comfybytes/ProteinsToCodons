include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")
include("Decoder.jl")

using Flux, BioSequences, CUDA, cuDNN, Dates, ProgressMeter
using SeqDL, SeqDL.Data, SeqDL.Util

struct Transformer
    d_model::Int
    d_hidden::Int
    n_heads::Int
    n_layers::Int
    p_drop::Float64
    max_len::Int
    encoder::Encoder
    decoder::Decoder
    linear::Dense
end

Flux.@functor Transformer

function Transformer(
    cds_data::CDSData,
    d_model::Int=16,
    d_hidden::Int=32,
    n_heads::Int=1,
    n_layers::Int=2,
    p_drop::Float64=0.1,
    max_len::Int=1000
)

    Transformer(
        d_model,
        d_hidden,
        n_heads,
        n_layers,
        p_drop,
        max_len,
        Encoder(cds_data.aa_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, max_len),
        Decoder(cds_data.aa_alphabet, cds_data.nt_alphabet, d_model, d_hidden, n_heads, n_layers, p_drop, max_len),
        Dense(d_model => length(cds_data.nt_alphabet), identity)
    )
end

function (t::Transformer)(prots::A, dna::A) where {A<:AbstractArray} # Function For Training
    enc_context = t.encoder(prots)
    dec_out = t.decoder(enc_context, dna, true)
    t.linear(dec_out)
end

function generate(sequence::LongAA, model::Transformer, cds_data::CDSData, usegpu::Bool=true) # Function For Inference
    device = usegpu ? gpu : cpu

    model = model |> device
    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    dna_tokenizer = Tokenizer(cds_data.nt_alphabet)

    sequence = aa_tokenizer(sequence)
    len_output = length(sequence) * 3
    sequence = reshape(sequence, size(sequence, 1), 1)
    enc_out = model.encoder(sequence)
    context = [5]
    for i in 1:len_output
        logits = reshape(context, size(context, 1), 1)
        logits = model.decoder(enc_out, logits)
        logits = model.linear(logits) |> cpu
        logits = softmax(logits)
        token = argmax(logits)
        context = vcat(context, token[1])
        if i == 1
            popfirst!(context)
        end
    end
    context = dna_tokenizer(context)
    context = LongDNA{4}(context)
end

function train_model(model::Transformer, cds_data::CDSData, epochs::Int=100, usegpu::Bool=true)
    @info "Preparing model for training"
    device = usegpu ? gpu : cpu
    CUDA.device_reset!()

    model = model |> device
    opt_state = Flux.setup(Adam(0.001,(0.9, 0.98)), model) |> device

    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    dna_tokenizer = Tokenizer(cds_data.nt_alphabet)
    
    peptides = aa_tokenizer(cds_data.peptide)
    x1_train, x1_test = Flux.splitobs(peptides, at=0.8)

    dna = dna_tokenizer(cds_data.dna)
    x2_train, x2_test = Flux.splitobs(dna, at=0.8)

    targets = target(cds_data.dna, dna_tokenizer)
    y_train, y_test = Flux.splitobs(targets, at=0.8)

    train_set = Flux.DataLoader((x1_train, x2_train, y_train) |> device, batchsize=128, shuffle=true)
    x1_test = Array(x1_test) |> device
    x2_test = Array(x2_test) |> device
    y_test = Array(y_test) |> device

    es = Flux.early_stopping(Flux.logitcrossentropy, 4 , init_score = 10)

    @showprogress desc="Training Model..." for epoch in 1:epochs
        Flux.train!(model, train_set, opt_state) do m, x1, x2, y
            Flux.logitcrossentropy(m(x1, x2), y)
        end

        if es(model(x1_test, x2_test), y_test)
            @info "Stopped training earlier at Epoch: $epoch out of $epochs due to increasing Loss on test set"
            break
        end
    end
    @info "Finished training model"
    return model
end

function save_model(model::Transformer, name::String, path::String="models/")
    if !isdir(path)
        mkdir(path)
    end

    model = cpu(model)
    model_state = Flux.state(model)
    d_model = model.d_model
    d_hidden = model.d_hidden
    n_heads = model.n_heads
    n_layers = model.n_layers

    model_name = "$(name)_$(now())_$(d_model)_$(d_hidden)_$(n_heads)_$(n_layers)"
    jldsave("$(path)$(model_name).jld2"; model_state)
    @info "Sucessfully saved model: $model_name"
end

function load_model(name::String, cds_data::CDSData, path::String="models/")
    name = endswith(name, ".jld2") ? name : name * ".jld2"
    model_state = JLD2.load("$(path)$(name)", "model_state")
    model = Transformer(
        cds_data,
        model_state.d_model,
        model_state.d_hidden,
        model_state.n_heads,
        model_state.n_layers,
        model_state.p_drop,
        model_state.max_len)
    Flux.loadmodel!(model, model_state)
end

function read_cds(species::String)
    
    fastaFiles = Dict(
    "creinhardtii" => "ena-ch-reinhardtii.fasta",
    "celegans" => "celegans.fasta",
    "ecoli" => "Escherichia_coli.HUSEC2011CHR1.cds.all.fa",
    "athaliana" => "Arabidopsis_thaliana.TAIR10.cds.all.fa",
    "drerio" => "Danio_rerio.GRCz11.cds.all.fa",
    "hsapiens" => "CCDS_nucleotide.current.fna",
    "scerevisiae" => "orf_genomic_yeast.fasta",
    )

    haskey(fastaFiles, species) || throw(ArgumentError("No fasta file available for $species, check spelling"))
    file = fastaFiles[species]
    extractCDS("./datafiles/$(file)")
end