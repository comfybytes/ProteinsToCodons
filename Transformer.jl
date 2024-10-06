include("Tokenizer.jl")
include("PosEncoding.jl")
include("Encoder.jl")
include("Decoder.jl")

using Flux, BioSequences, CUDA, cuDNN, Dates, ProgressMeter, Serialization
using SeqDL, SeqDL.Data, SeqDL.Util

"""
    Transformer(cds_data::CDSData, d_model::Int, d_hidden::Int, n_heads::Int, n_layers::Int, p_drop::Float64, max_len::Int)

Creates an Encoder-Decoder Transformer. 
Can be trained using the `train_model` function. 
Can generate an output for a sequence with `generate` function.

# Arguments
- `cds_data`: CDSData containing data for an organism
- `d_model`: dimensions of model. input and output must be this size. Default 256
- `d_hidden`: dimensions of the hidden layer in the feed-forward network. Default 1024
- `n_heads`: number of attention heads. Must equally divide `d_model`. Default 2
- `n_layers`: number of sequential Encoders. Default 2
- `p_drop`: probability for dropout. Default 0.1
- `max_len`: maximum sequence length. Default 1000
"""

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
    d_model::Int=256,
    d_hidden::Int=1024,
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

# Function for Training. Functor links together encoder and decoder followed by a linear transformation
function (t::Transformer)(prots::A, dna::A) where {A<:AbstractArray}
    enc_context = t.encoder(prots)
    dec_out = t.decoder(enc_context, dna, true)
    t.linear(dec_out)
end

"""
    generate(sequence::LongAA, model::Transformer, cds_data::CDSData, usegpu::Bool)

Function for auto-regressive token-by-token generation of DNA for a given sequence of amino acids.
Generates DNA tokens until 3-times the length of the amino acid sequence is reached.
Uses an empty context to start ouput generation.
Consumes the previously generated context to generate the next context.
Returns the predicted DNA sequence.

# Arguments
- `sequence`: sequence of amino acids
- `model`: The created Transformer model
- `cds_data`: CDSData containing data for an organism
- `usegpu`: Whether or not to use NVIDIA GPU during inference
"""

function generate(sequence::LongAA, model::Transformer, cds_data::CDSData, usegpu::Bool=true)
    device = usegpu ? gpu : cpu
    model = model |> device

    aa_tokenizer = Tokenizer(cds_data.aa_alphabet)
    dna_tokenizer = Tokenizer(cds_data.nt_alphabet)

    sequence = aa_tokenizer(sequence)
    len_output = length(sequence) * 3
    sequence = reshape(sequence, size(sequence, 1), 1)

    enc_out = model.encoder(sequence)
    context = reshape([5], 1, 1)
    output = Vector{Int64}()
    
    for i in 1:len_output
        context = model.decoder(enc_out, context)
        logits = model.linear(context) |> cpu
        logits = softmax(logits)
        token = argmax(logits)
        push!(output, token[1])
    end
    output = dna_tokenizer(output)
    output = LongDNA{4}(output)
end

"""
    train_model(model::Transformer, cds_data::CDSData, epochs::Int, usegpu::Bool)

Function for model training.
Creates Tokenizers, one-hot encodes the labeled data, randomly shuffles the order of sequences and trains in batches.
Uses Early Stopping and logit cross-entropy.
Returns a trained model.

# Arguments
- `model`: The created Transformer model
- `cds_data`: CDSData containing data for an organism
- `epochs`: Number of epochs to train the model for
- `usegpu`: Whether or not to use NVIDIA GPU during inference
"""

function train_model(model::Transformer, cds_data::CDSData, epochs::Int=100, usegpu::Bool=true)
    @info "Preparing model for training"
    device = usegpu ? gpu : cpu
    CUDA.device_reset!()

    model = model |> device
    opt_state = Flux.setup(Adam(), model) |> device

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

    #es = Flux.early_stopping(Flux.logitcrossentropy, 7 , init_score = 10)

    @showprogress desc="Training Model..." for epoch in 1:epochs
        Flux.train!(model, train_set, opt_state) do m, x1, x2, y
            loss = Flux.logitcrossentropy(m(x1, x2), y)
        end
        #if es(model(x1_test, x2_test), y_test)
        #    @info "Stopped training earlier at Epoch: $epoch out of $epochs due to increasing Loss on test set"
        #    break
        #end
    end
    @info "Finished training model"
    return model
end

