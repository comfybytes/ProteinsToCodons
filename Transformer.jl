include("Encoder.jl")

example = [5,3,1]


encoder = Encoder()
encoder_example = encoder(example)
println(size(encoder_example))
