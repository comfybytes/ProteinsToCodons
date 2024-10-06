include("TransformerClassifier.jl")
include("Transformer.jl")

# save Transformer or TransformerClassifier
function save_model(model::M, name::String, path::String="models/") where {M:<Union{Transformer, TransformerClassifier}}
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

# Load TransformerClassifier
function load_model(name::String, cds_data::CDSData, path::String="models/")
    name = endswith(name, ".jld2") ? name : name * ".jld2"
    model_state = JLD2.load("$(path)$(name)", "model_state")
    model = TransformerClassifier(
        cds_data,
        model_state.d_model,
        model_state.d_hidden,
        model_state.n_heads,
        model_state.n_layers,
        model_state.p_drop,
        model_state.max_len)
    Flux.loadmodel!(model, model_state)
end

# get CDSData from serialized organism files
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