
function update_schedule(
    schedule::AbstractVector,
    stats::AbstractVector{<:NamedTuple},
    n_steps_next::Int
)
    divergences = map(enumerate(stats)) do (t, stat)
        if t == 1
            zero(eltype(schedule))
        else
            ℓG1, ℓG2 = stat.log_potential_moments
            ℓG2 - 2*ℓG1
        end
    end
    local_barrier          = sqrt.(divergences)
    barrier                = cumsum(local_barrier)
    global_barrier         = last(barrier)
    barrier_interp         = CubicSpline(barrier, schedule; extrapolate=true)
    local_barrier_invmap   = CubicSpline(schedule, barrier)
    target_barrier_profile = global_barrier*range(0, 1; length=n_steps_next)
    schedule_next          = local_barrier_invmap(target_barrier_profile)
    local_barrier          = DataInterpolations.derivative.(Ref(barrier_interp), schedule)
    schedule_next, local_barrier, global_barrier
end
