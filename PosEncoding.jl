using Flux

struct PositionEncoding{E<:AbstractArray}
    encoding::E
end

Flux.@functor PositionEncoding
Flux.trainable(m::PositionEncoding) = (;)

function PositionEncoding(d_model::Int, max_len::Int=1000)
    pos_enc = positional_encoding(d_model, max_len)
    PositionEncoding(pos_enc)
end

function positional_encoding(d_model::Int, max_len::Int=1000)
    encoding = Matrix{Float32}(undef, d_model, max_len)
    for pos in 1:max_len
        for dim in 1:2:(d_model-1)
            term = pos / 10000^(2 * dim / d_model)
            encoding[dim, pos] = sin(term)
            encoding[dim+1, pos] = cos(term)
        end
    end
    encoding
end

function (p::PositionEncoding)(x::AbstractArray)
    seq_len = size(x, 2)
    max_len = size(p.encoding, 2)
    max_len >= seq_len || throw(ArgumentError("seq_len = $(seq_len) exceeds maximum position encoding length of $(max_len)"))

    p.encoding[:, 1:seq_len]
end