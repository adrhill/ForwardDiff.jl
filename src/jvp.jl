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
    require_one_based_indexing(x)
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
    require_one_based_indexing(y, x)
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
    result isa DiffResult ? require_one_based_indexing(x) : require_one_based_indexing(result, x)
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
    result isa DiffResult ? require_one_based_indexing(y, x) : require_one_based_indexing(result, y, x)
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

function extract_jvp!(::Type{T}, result::AbstractArray, ydual::AbstractArray, n) where {T}
    out_reshaped = reshape(result, length(ydual), n)
    ydual_reshaped = vec(ydual)
    # Use closure to avoid GPU broadcasting with Type
    partials_wrap(ydual, nrange) = partials(T, ydual, nrange)
    out_reshaped .= partials_wrap.(ydual_reshaped, transpose(1:n))
    return result
end

function extract_jvp!(::Type{T}, result::MutableDiffResult, ydual::AbstractArray, n) where {T}
    extract_jvp!(T, DiffResults.jvp(result), ydual, n)
    return result
end

function extract_jvp_chunk!(::Type{T}, result, ydual, index, chunksize) where {T}
    ydual_reshaped = vec(ydual)
    offset = index - 1
    irange = 1:chunksize
    col = irange .+ offset
    # Use closure to avoid GPU broadcasting with Type
    partials_wrap(ydual, nrange) = partials(T, ydual, nrange)
    result[:, col] .= partials_wrap.(ydual_reshaped, transpose(irange))
    return result
end

reshape_jvp(result, ydual, xdual) = reshape(result, length(ydual), length(xdual))
reshape_jvp(result::DiffResult, ydual, xdual) = reshape_jvp(DiffResults.jvp(result), ydual, xdual)

###############
# vector mode #
###############

function vector_mode_jvp(f::F, x, cfg::JVPConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f, cfg, x)
    ydual isa AbstractArray || throw(JVP_ERROR)
    result = similar(ydual, valtype(eltype(ydual)), length(ydual), N)
    extract_jvp!(T, result, ydual, N)
    extract_value!(T, result, ydual)
    return result
end

function vector_mode_jvp(f!::F, y, x, cfg::JVPConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    map!(d -> value(T,d), y, ydual)
    result = similar(y, length(y), N)
    extract_jvp!(T, result, ydual, N)
    map!(d -> value(T,d), y, ydual)
    return result
end

function vector_mode_jvp!(result, f::F, x, cfg::JVPConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f, cfg, x)
    extract_jvp!(T, result, ydual, N)
    extract_value!(T, result, ydual)
    return result
end

function vector_mode_jvp!(result, f!::F, y, x, cfg::JVPConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    map!(d -> value(T,d), y, ydual)
    extract_jvp!(T, result, ydual, N)
    extract_value!(T, result, y, ydual)
    return result
end

# TODO: scalar outputs
const JVP_ERROR = DimensionMismatch("jvp(f, x, dx) expects that f(x) is an array. Perhaps you meant gradient(f, x)?")

# chunk mode #
#------------#

function jvp_chunk_mode_expr(work_array_definition::Expr, compute_ydual::Expr,
                                  result_definition::Expr, y_definition::Expr)
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
        seed!(xdual, x, 1, seeds)
        $(compute_ydual)
        ydual isa AbstractArray || throw(JVP_ERROR)
        $(result_definition)
        out_reshaped = reshape_jvp(result, ydual, xdual)
        extract_jvp_chunk!(T, out_reshaped, ydual, 1, N)
        seed!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            $(compute_ydual)
            extract_jvp_chunk!(T, out_reshaped, ydual, i, N)
            seed!(xdual, x, i)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        $(compute_ydual)
        extract_jvp_chunk!(T, out_reshaped, ydual, lastchunkindex, lastchunksize)

        $(y_definition)

        return result
    end
end

@eval function chunk_mode_jvp(f::F, x, cfg::JVPConfig{T,V,N}) where {F,T,V,N}
    $(jvp_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(result = similar(ydual, valtype(eltype(ydual)), length(ydual), xlen)),
                               :()))
end

@eval function chunk_mode_jvp(f!::F, y, x, cfg::JVPConfig{T,V,N}) where {F,T,V,N}
    $(jvp_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(result = similar(y, length(y), xlen)),
                               :(map!(d -> value(T,d), y, ydual))))
end

@eval function chunk_mode_jvp!(result, f::F, x, cfg::JVPConfig{T,V,N}) where {F,T,V,N}
    $(jvp_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(),
                               :(extract_value!(T, result, ydual))))
end

@eval function chunk_mode_jvp!(result, f!::F, y, x, cfg::JVPConfig{T,V,N}) where {F,T,V,N}
    $(jvp_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(),
                               :(extract_value!(T, result, y, ydual))))
end
