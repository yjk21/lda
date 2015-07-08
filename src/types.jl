type VBposterior
    λ::Matrix{Float64}
    lλ::Matrix{Float64}
    ϕ::Array{Float64,3}
    lϕ::Array{Float64,3}
    γ::Matrix{Float64}

    function VBposterior(Y, K)
        V,M = size(Y)
        ltemp = rand(V,K) + 1
        ltemp = nmzLambda(ltemp)
        ptemp = rand(K,V,M) + 1
        ptemp = nmzPhi(ptemp)
        gtemp = ones(K,M) 
        this = new(ltemp,log(ltemp), ptemp,zeros(K,V,M), gtemp)
        return this
    end
end

function init!(Q,Y, α, η)
    V,M = size(Y)
    K = length(α)
    docTotal = sum(Y,1)
    for d in 1:M
        for k in 1:K
            Q.γ[k, d] = α[k] + docTotal[d] / K
            println(Q.γ[k, d] )
            for j in 1:V
                Q.ϕ[k, j, d ] = 1.0/K
                Q.lϕ[k, j, d ] =  log(1.0/K)
            end
        end
    end
    Q.λ = nmzLambda(rand(V,K) + η)
    Q.lλ = nmzLambda(rand(V,K) + η)
end


type VBparams

    α::Vector{Float64}
    η::Float64

    function VBparams(K)
        new(ones(K), 0.0)
    end

end


type LDAoptions
    maxItVB::Int64
    maxItEM::Int64
    tolAlpha::Float64
    tolEM::Float64
    verbosity::Int64
    LDAoptions() = new(0,0,0.0,0.0,0)
end
