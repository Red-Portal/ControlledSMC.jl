
function update_schedule(
    schedule::AbstractVector,
    stats::AbstractVector{<:NamedTuple},
    n_steps_next::Int,
)
    divergences = map(enumerate(stats)) do (t, stat)
        if t == 1
            zero(eltype(schedule))
        else
            ℓG1, ℓG2 = stat.log_potential_moments
            ℓG2 - 2*ℓG1
        end
    end
    local_barrier        = sqrt.(divergences)
    local_barrier_cum    = cumsum(local_barrier)
    global_barrier       = last(local_barrier_cum)
    local_barrier_invmap = CubicSpline(schedule, local_barrier_cum)
    target_local_barrier = global_barrier*(1:n_steps_next)/n_steps_next
    local_barrier_invmap(target_local_barrier)
end
