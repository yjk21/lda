module LDA

using ArrayViews
import Calculus

export lda, VBposterior , runTests, LDAoptions, VBparams, VBposterior

include("src/types.jl")
include("src/utils.jl")
include("src/LDA.jl")
include("src/LDAs.jl")
include("src/tests.jl")

end

