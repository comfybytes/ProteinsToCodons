include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")

using Flux, BioSequences, CUDA, cuDNN, Dates, ProgressMeter, Serialization
using SeqDL, SeqDL.Data, SeqDL.Util

"""
    TransformerClassifier(cds_data::CDSData, d_model::Int, d_hidden::Int, n_heads::Int, n_layers::Int, p_drop::Float64, max_len::Int)

Creates an Encoder-onl Transformer. 
Can be trained using the `train_model` function. 
Can classify one or more sequences of amino acids with `generate` function.

# Arguments
- `cds_data`: CDSData containing data for an organism
- `d_model`: dimensions of model. input and output must be this size. Default 256
- `d_hidden`: dimensions of the hidden layer in the feed-forward network. Default 1024
- `n_heads`: number of attention heads. Must equally divide `d_model`. Default 2
- `n_layers`: number of sequential Encoders. Default 2
- `p_drop`: probability for dropout. Default 0.1
- `max_len`: maximum sequence length. Default 1000
"""

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
    d_model::Int=256,
    d_hidden::Int=1024,
    n_heads::Int=2,
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

# Function for Training. Functor links together encoder and linear transformation
function (t::TransformerClassifier)(prots::A) where {A<:AbstractArray}
    context = t.encoder(prots)
    t.linear(context)
end

"""
    generate(sequence::LongAA, model::Transformer, cds_data::CDSData, usegpu::Bool)

Function for classification of one or more amino acid sequences.
Outputs a DNA sequence by assigning a codon to each amino acid.
Returns the predicted DNA sequence.

# Arguments
- `sequence`: sequence of amino acids
- `model`: The created Transformer model
- `cds_data`: CDSData containing data for an organism
- `usegpu`: Whether or not to use NVIDIA GPU during inference
"""

function generate(sequences::Vector{LongAA}, model::TransformerClassifier, cds_data::CDSData, usegpu::Bool=true)
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

"""
    train_model(model::TransformerClassifier, cds_data::CDSData, epochs::Int, usegpu::Bool)

Function for model training.
Creates Tokenizers, one-hot encodes the labeled data, randomly shuffles the order of sequences and trains in batches.
Uses Early Stopping and logit cross-entropy.
Returns a trained model.

# Arguments
- `model`: The created TransformerClassifier model
- `cds_data`: CDSData containing data for an organism
- `epochs`: Number of epochs to train the model for
- `usegpu`: Whether or not to use NVIDIA GPU during inference
"""

function train_model(model::TransformerClassifier, cds_data::CDSData, epochs::Int=100, usegpu::Bool=true)
    @info "Preparing model for training"
    device = usegpu ? gpu : cpu
    CUDA.device_reset!()

    model = model |> device
    opt_state = Flux.setup(Adam(0.001,(0.9,0.98)), model) |> device

    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    cd_tokenizer = CodonTokenizer()

    peptides = aa_tokenizer(cds_data.peptide)
    x_train, x_test = Flux.splitobs(peptides, at=0.8)
    
    targets = target(seq_to_codons(cds_data.dna), cd_tokenizer)
    y_train, y_test = Flux.splitobs(targets, at=0.8)

    train_set = Flux.DataLoader((x_train, y_train) |> device, batchsize=128, shuffle=true)
    x_test = Array(x_test) |> device
    y_test = Array(y_test) |> device

    es = Flux.early_stopping(Flux.logitcrossentropy, 1 , init_score = 5)

    @showprogress desc="Training Model..." for epoch in 1:epochs
        Flux.train!(model, train_set, opt_state) do m, x, y
            smoothed = Flux.label_smoothing(y, 0.3f0)
            loss = Flux.logitcrossentropy(m(x), smoothed)
        end
        if epoch > 100
            if es(model(x_test), y_test)
                @info "Stopped training earlier at Epoch: $epoch out of $epochs due to increasing loss on test set"
                break
            end
        end
    end
    @info "Finished training model"
    return model
end