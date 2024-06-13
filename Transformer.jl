include("Encoder.jl")
include("Tokenizer.jl")
include("Decoder.jl")

using SeqDL, SeqDL.Data, SeqDL.Util, BioSequences, Flux


struct Transformer
    d_model::Int
    in_tokenizer::Tokenizer
    out_tokenizer::Tokenizer
    in_embedder::Embedding
    out_embedder::Embedding
    encoder::Encoder
    decoder::Decoder
    drop_out::Dropout
end

function Transformer(
    in_alphabet,
    out_alphabet,
    d_model::Int=240,
    d_hidden::Int=480,
    n_heads::Int=4,
    p_drop::Float64=0.1)

    in_len = length(in_alphabet)
    out_len = length(out_alphabet)

    return Transformer(
        d_model,
        Tokenizer(in_alphabet),
        Tokenizer(out_alphabet),
        Embedding(in_len => d_model),
        Embedding(out_len => d_model),
        Encoder(d_model, d_hidden, n_heads, p_drop),
        Decoder(d_model, d_hidden, n_heads, p_drop),
        Dropout(p_drop)
    )
end

function (t::Transformer)(peptides, dna, n_layers)
    in_tokens = t.in_tokenizer(peptides)
    in_seq_len = size(in_tokens, 1)
    in_seq_num = size(in_tokens, 2)
    input = t.in_embedder(in_tokens) + positional_encoding(in_seq_len, in_seq_num, t.d_model)
    input = t.drop_out(input)
    for _ in 1:n_layers
        input = t.encoder(input)
    end
    out_tokens = t.out_tokenizer(dna)
    out_seq_len = size(out_tokens, 1)
    out_seq_num = size(out_tokens, 2)
    output = t.out_embedder(out_tokens) + positional_encoding(out_seq_len, out_seq_num, t.d_model)
    output = t.drop_out(output)
    mask = make_causal_mask(output)

    for _ in 1:n_layers
        output = t.decoder(input, output, mask)
    end
    return output
end
