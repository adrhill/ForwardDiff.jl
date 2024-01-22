###############
# API methods #
###############

# Related issues: #319, #428, #487

"""
    ForwardDiff.jvp(f, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig = JVPConfig(f, x, dx), check=Val{true}())

Return `J(f) ⋅ dx` (where `J(f)` is evaluated at `x`), assuming `f` is called as `f(x)`.
Multidimensional arrays are flattened in iteration order: the array
`J(f)` has shape `length(f(x)) × length(x)`, and its elements are
`J(f)[j,k] = ∂f(x)[j]/∂x[k]`.
`J(f) ⋅ dx` therefore has shape `length(f(x))`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jvp(f::F, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig{T} = JVPConfig(f, x, dx), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    require_one_based_indexing(x, dx)
    CHK && checktag(T, f, x)
    if chunksize(cfg) == length(x)
        return vector_mode_jvp(f, x, cfg)
    else
        return chunk_mode_jvp(f, x, cfg)
    end
end

"""
    ForwardDiff.jvp(f!, y::AbstractArray, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig = JVPConfig(f!, y, x, dx), check=Val{true}())

Return `J(f!) ⋅ dx` (where `J(f!)` is evaluated at `x`),  assuming `f!` is called as `f!(y, x)` where the result is
stored in `y`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jvp(f!::F, y::AbstractArray, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig{T} = JVPConfig(f!, y, x, dx), ::Val{CHK}=Val{true}()) where {F,T, CHK}
    require_one_based_indexing(y, x, dx)
    CHK && checktag(T, f!, x)
    if chunksize(cfg) == length(x)
        return vector_mode_jvp(f!, y, x, cfg)
    else
        return chunk_mode_jvp(f!, y, x, cfg)
    end
end


"""
    ForwardDiff.jvp!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig = JVPConfig(f, x, dx), check=Val{true}())

Compute `J(f) ⋅ dx` (where `J(f)` is evaluated at `x`) and store the result(s) in `result`, assuming `f` is called
as `f(x)`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jvp!(result::Union{AbstractArray,DiffResult}, f::F, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig{T} = JVPConfig(f, x, dx), ::Val{CHK}=Val{true}()) where {F,T, CHK}
    result isa DiffResult ? require_one_based_indexing(x, dx) : require_one_based_indexing(result, x, dx)
    CHK && checktag(T, f, x)
    if chunksize(cfg) == length(x)
        vector_mode_jvp!(result, f, x, cfg)
    else
        chunk_mode_jvp!(result, f, x, cfg)
    end
    return result
end

"""
    ForwardDiff.jvp!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig = JVPConfig(f!, y, x, dx), check=Val{true}())

Compute `J(f) ⋅ dx` (where `J(f)` is evaluated at `x`) and store the result(s) in `result`, assuming `f!` is
called as `f!(y, x)` where the result is stored in `y`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jvp!(result::Union{AbstractArray,DiffResult}, f!::F, y::AbstractArray, x::AbstractArray, dx::AbstractArray, cfg::JVPConfig{T} = JVPConfig(f!, y, x, dx), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    result isa DiffResult ? require_one_based_indexing(y, x, dx) : require_one_based_indexing(result, y, x, dx)
    CHK && checktag(T, f!, x)
    if chunksize(cfg) == length(x)
        vector_mode_jvp!(result, f!, y, x, cfg)
    else
        chunk_mode_jvp!(result, f!, y, x, cfg)
    end
    return result
end

# TODO: scalar inputs
jvp(f, x::Real, dx::Real) = throw(DimensionMismatch("jvp(f, x, dx) expects that x is an array. Perhaps you meant `derivative(f, x) * dx`?"))

#####################
# result extraction #
#####################



function extract_jvp!(::Type{T}, result::AbstractArray, ydual::AbstractArray) where {T}
    result .= sum.(partials.(T, ydual))
    return result
end

function extract_jvp!(::Type{T}, result::MutableDiffResult, ydual::AbstractArray) where {T}
    extract_jvp!(T, DiffResults.jvp(result), ydual) # TODO: support DiffResults
    return result
end

function extract_jvp_chunk!(::Type{T}, result, ydual) where {T}
    result .+= sum.(partials.(T, ydual))
    return result
end

###############
# vector mode #
###############

function vector_mode_jvp(f::F, x, cfg::JVPConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    ydual isa AbstractArray || throw(JVP_ERROR) # TODO: relax, also allow Reals
    result = similar(ydual, valtype(eltype(ydual)))
    extract_jvp!(T, result, ydual)
    return result
end

function vector_mode_jvp(f!::F, y, x, cfg::JVPConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    map!(d -> value(T,d), y, ydual)
    result = similar(y)
    extract_jvp!(T, result, ydual)
    return result
end

function vector_mode_jvp!(result, f::F, x, cfg::JVPConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    extract_jvp!(T, result, ydual)
    return result
end

function vector_mode_jvp!(result, f!::F, y, x, cfg::JVPConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    map!(d -> value(T,d), y, ydual)
    extract_jvp!(T, result, ydual)
    return result
end

# TODO: scalar outputs
const JVP_ERROR = DimensionMismatch("jvp(f, x, dx) expects that f(x) is an array. Perhaps you meant `derivative(f, x) * dx`?")

##############
# chunk mode #
##############

function jvp_chunk_mode_expr(work_array_definition::Expr, compute_ydual::Expr,
                                  result_definition::Expr)
    return quote
        @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

        # precalculate loop bounds
        xlen = length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        middlechunks = 2:div(xlen - lastchunksize, N)

        # seed work arrays
        $(work_array_definition)
        seeds = cfg.seeds

        # do first chunk manually to calculate output type
        seed_jvp!(xdual, x, 1, seeds)
        $(compute_ydual)
        ydual isa AbstractArray || throw(JVP_ERROR)  # TODO: relax, also allow Reals
        $(result_definition)
        fill!(result, zero(V))
        extract_jvp_chunk!(T, result, ydual)
        seed_jvp!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed_jvp!(xdual, x, i, seeds)
            $(compute_ydual)
            extract_jvp_chunk!(T, result, ydual)
            seed_jvp!(xdual, x, i)
        end

        # do final chunk
        seed_jvp!(xdual, x, lastchunkindex, seeds, lastchunksize)
        $(compute_ydual)
        extract_jvp_chunk!(T, result, ydual)

        return result
    end
end

@eval function chunk_mode_jvp(f::F, x, cfg::JVPConfig{T,V,NP,N}) where {F,T,V,NP,N}
    $(jvp_chunk_mode_expr(quote
                              xdual = cfg.duals
                              seed_jvp!(xdual, x)
                          end,
                          :(ydual = f(xdual)),
                          :(result = similar(ydual, valtype(eltype(ydual))))))
end

@eval function chunk_mode_jvp(f!::F, y, x, cfg::JVPConfig{T,V,NP,N}) where {F,T,V,NP,N}
    $(jvp_chunk_mode_expr(quote
                              ydual, xdual = cfg.duals
                              seed_jvp!(xdual, x)
                          end,
                          :(f!(seed_jvp!(ydual, y), xdual)),
                          :(result = similar(y))))
end

@eval function chunk_mode_jvp!(result, f::F, x, cfg::JVPConfig{T,V,NP,N}) where {F,T,V,NP,N}
    $(jvp_chunk_mode_expr(quote
                              xdual = cfg.duals
                              seed_jvp!(xdual, x)
                          end,
                          :(ydual = f(xdual)),
                          :())) # defines `result` through argument
end

@eval function chunk_mode_jvp!(result, f!::F, y, x, cfg::JVPConfig{T,V,NP,N}) where {F,T,V,NP,N}
    $(jvp_chunk_mode_expr(quote
                              ydual, xdual = cfg.duals
                              seed_jvp!(xdual, x)
                          end,
                          :(f!(ydual, xdual)),
                          :())) # defines `result` through argument
end
