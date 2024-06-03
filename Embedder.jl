using BioSequences

function positional_encoding(seq_length,seq_num,d_model::dimension) where {dimension<:Integer}
    encoding = fill(0.0,(d_model,seq_length,seq_num))
    encoding = convert(Array{Float32, seq_num},encoding)
    for batch in 1:seq_num
        for pos in batch
            for i in 1:2:(d_model-1)
                term = pos/10000^(2i/d_model)
                encoding[i,pos,batch] = sin(term)
                encoding[i+1,pos,batch] = cos(term)
            end
        end
    end
    return encoding
end
