using Flux

"""
    PositionEncoding(d_model::Int, max_len::Int)

Creates a matrix with the same values each time, which contains the positional encoding applied after embedding.

# Arguments
- `d_model`: dimensions of model. input and output must be this size. Default 256
- `max_len`: maximum sequence length. Default 1000
"""

struct PositionEncoding{E<:AbstractArray}
    encoding::E
end

Flux.@functor PositionEncoding
Flux.trainable(m::PositionEncoding) = (;) # tell Flux this Layer is not trainable

# used for object creation to use the functor
function PositionEncoding(d_model::Int, max_len::Int=1000)
    pos_enc = positional_encoding(d_model, max_len)
    PositionEncoding(pos_enc)
end

# creates the positional encoding values in a matrix with the same dimension as d_model and specifies a maximum sequence length
function positional_encoding(d_model::Int, max_len::Int=1000)
    encoding = Matrix{Float32}(undef, d_model, max_len)
    for pos in 1:max_len
        for dim in 1:2:(d_model-1)
            term = pos / 10000^(dim / d_model)
            encoding[dim, pos] = cos(term)
            encoding[dim+1, pos] = sin(term)
        end
    end
    encoding
end

# returns a subset of the positional encoding matrix or vector with the same dimensions as the input matrix or vector
function (p::PositionEncoding)(x::AbstractArray)
    seq_len = size(x, 2)
    max_len = size(p.encoding, 2)
    max_len >= seq_len || throw(ArgumentError("seq_len = $(seq_len) exceeds maximum length of $(max_len)"))

    p.encoding[:, 1:seq_len, 1]
end