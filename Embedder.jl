
function positional_encoding(seq_length,d_model::dimension) where {dimension<:Integer}
    encoding = convert(Matrix{Float32},fill(0,(d_model,seq_length)))
    for pos in 1:seq_length
        for i in 1:2:(d_model-1)
            term = pos/10000^(2i/d_model)
            encoding[i,pos] = sin(term)
            encoding[i+1,pos] = cos(term)
        end
    end
    return encoding
end

