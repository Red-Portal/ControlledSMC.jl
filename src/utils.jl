
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end
