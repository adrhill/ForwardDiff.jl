####################
# value extraction #
####################

@inline extract_value!(::Type{T}, out::DiffResult, ydual) where {T} =
    DiffResults.value!(d -> value(T,d), out, ydual)
@inline extract_value!(::Type{T}, out, ydual) where {T} = out # ???

@inline function extract_value!(::Type{T}, out, y, ydual) where {T}
    map!(d -> value(T,d), y, ydual)
    copy_value!(out, y)
end

@inline copy_value!(out::DiffResult, y) = DiffResults.value!(out, y)
@inline copy_value!(out, y) = out

###################################
# vector mode function evaluation #
###################################

function vector_mode_dual_eval!(f::F, cfg::Union{JacobianConfig,GradientConfig,JVPConfig}, x) where {F}
    xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    return f(xdual)
end

function vector_mode_dual_eval!(f!::F, cfg::Union{JacobianConfig,JVPConfig}, y, x) where {F}
    ydual, xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    seed!(ydual, y)
    f!(ydual, xdual)
    return ydual
end

##################################
# seed construction/manipulation #
##################################

@generated function construct_seeds(::Type{Partials{N,V}}) where {N,V}
    return Expr(:tuple, [:(single_seed(Partials{N,V}, Val{$i}())) for i in 1:N]...)
end

function construct_jvp_seeds(::Type{Partials{N,V}}, dx::AbstractArray{V}) where {N,V}
    return tuple([single_jvp_seed(Partials{N,V}, mod1(i, N), x) for (i, x) in enumerate(dx)]...)
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    duals .= Dual{T,V,N}.(x, Ref(seed))
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    dual_inds = 1:N
    duals[dual_inds] .= Dual{T,V,N}.(view(x,dual_inds), seeds)
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    offset = index - 1
    dual_inds = (1:N) .+ offset
    duals[dual_inds] .= Dual{T,V,N}.(view(x, dual_inds), Ref(seed))
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    seed_inds = 1:chunksize
    dual_inds = seed_inds .+ offset
    duals[dual_inds] .= Dual{T,V,N}.(view(x, dual_inds), getindex.(Ref(seeds), seed_inds))
    return duals
end

seed_jvp!(duals, x) = seed!(duals, x)
seed_jvp!(duals, x, index) = seed!(duals, x, index)
function seed_jvp!(duals::AbstractArray{Dual{T,V,N}}, x, index,
                   seeds::NTuple{NP,Partials{N,V}}, chunksize = N) where {T,V,NP,N}
    offset = index - 1
    dual_inds = (1:chunksize) .+ offset
    @views duals[dual_inds] .= Dual{T,V,N}.(x[dual_inds], seeds[dual_inds])
    return duals
end
