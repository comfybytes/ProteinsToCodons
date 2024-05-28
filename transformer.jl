using Flux


function positional_encoding(seq_length,d_model)
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


d_model = 512 # model dimensionality
example = [5,3,1]

encoder_embedding = Embedding(20 => d_model)
embedded = encoder_embedding(example)
encoded = embedded + positional_encoding(length(example),d_model)

# add batch size for example data
encoded = cat(encoded; dims = ndims(encoded) +1) 

# Create 3 different learned linear projections to apply to positionally encoded data
# Reduces dimensionality from 512 to 64 in preparation for Multi-Head-Attention
q_projection = Dense(512 => 64)
k_projection = Dense(512 => 64)
v_projection = Dense(512 => 64)

q = q_projection(encoded)
k = k_projection(encoded)
v = v_projection(encoded)

# create MultiHeadAttention, 8 heads. 512/heads = 64. 
# input dimensionality of 64 with output dimensionality of 512
mha = MultiHeadAttention(64 => 512 => 512, nheads = 8)

