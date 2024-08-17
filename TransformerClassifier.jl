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

function generate(sequence::LongAA, model::TransformerClassifier, cds_data::CDSData, usegpu::Bool=true) # Function For Inference
    device = usegpu ? gpu : cpu
    model = model |> device

    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    cd_tokenizer = CodonTokenizer()

    sequence = aa_tokenizer(sequence)
    sequence = reshape(sequence, size(sequence, 1), 1)
    context = model(sequence)
    logits = softmax(context) |> cpu
    logits = map(x -> x[1], argmax(logits, dims=1))
    logits = reshape(logits, size(logits, 2))
    output = cd_tokenizer(logits)
    output = join(output)
    output = LongDNA{4}(output)
end

function generate(sequences::Vector{LongAA}, model::TransformerClassifier, cds_data::CDSData, usegpu::Bool=true) # Function For Inference
    device = usegpu ? gpu : cpu
    model = model |> device

    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    cd_tokenizer = CodonTokenizer()

    sequences = aa_tokenizer(sequences)

    context = model(sequences)
    logits = softmax(context) |> cpu
    logits = map(x -> x[1], argmax(logits, dims=1))
    logits = reshape(logits, size(logits, 2), size(logits, 3))
    output = cd_tokenizer(logits)
    output = [join(collect(col)) for col in eachcol(output)]
    output = map(LongDNA{4}, output)
end

function train_model(model::TransformerClassifier, cds_data::CDSData, epochs::Int=100, usegpu::Bool=true)
    @info "Preparing model for training"
    device = usegpu ? gpu : cpu
    CUDA.device_reset!()

    model = model |> device
    opt_state = Flux.setup(Adam(), model) |> device

    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    cd_tokenizer = CodonTokenizer()

    peptides = aa_tokenizer(cds_data.peptide)
    x_train, x_test = Flux.splitobs(peptides, at=0.8)
    
    targets = target(seq_to_codons(cds_data.dna), cd_tokenizer)
    y_train, y_test = Flux.splitobs(targets, at=0.8)

    train_set = Flux.DataLoader((x_train, y_train) |> device, batchsize=128, shuffle=true)
    x_test = Array(x_test) |> device
    y_test = Array(y_test) |> device

    es = Flux.early_stopping(Flux.logitcrossentropy, 3 , init_score = 5)

    @showprogress desc="Training Model..." for epoch in 1:epochs
        Flux.train!(model, train_set, opt_state) do m, x, y
            smoothed = Flux.label_smoothing(y, 0.1f0)
            loss = Flux.logitcrossentropy(m(x), smoothed)
        end
        #if es(model(x_test), y_test)
        #    @info "Stopped training earlier at Epoch: $epoch out of $epochs due to increasing loss on test set"
        #    break
        #end
    end
    @info "Finished training model"
    return model
end

function seq_to_codons(dna::Vector{LongDNA})
    map(seq -> [seq[i:i+2] for i in 1:3:length(seq)],dna)
end

function get_cds(species::String)
    @info "Reading CDS Data"
    cds_files = Dict(
        "athaliana" => "athaliana-cds-d0002.jls",
        "celegans" => "celegans-cds-d0002.jls",
        "creinhardtii" => "creinhardtii-cds-d0002.jls",
        "drerio" => "drerio-cds-d0002.jls",
        "ecoli" => "ecoli-cds-d0002.jls",
        "hsapiens" => "hsapiens-cds-d0002.jls",
        "scerevisiae" => "scerevisiae-cds-d0002.jls"
    )
    haskey(cds_files, species) || throw(ArgumentError("No fasta file available for $species, check spelling"))
    file = cds_files[species]
    deserialize("./datafiles/$(file)")
end