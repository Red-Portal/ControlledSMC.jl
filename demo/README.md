
# Controlled Sequential Monte Carlo 
## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.update()
```

## Project Structure
* `sample.jl`: The generic sampling routines.
* `common.jl`: Commonly shared utilities.
* `twist.jl`: Twisting-specific routines.

* `smc_ula.jl`: SMC with an ULA kernel implementation.
* `smc_klmc.jl`: SMC with a KLMC kernel implementation.
* `csmc_ula.jl`: Controlled SMC with an ULA kernel implementation.
* `csmc_klmc.jl`: Controlled SMC with a KLMC kernel implementation.

All that each implementation of SMC are doing is defining the following methods:
```julia
rand_initial_with_potential(rng, sampler, n_particles)

mutate_with_potential(rng, sampler, timestep, particles, logtarget)
```
The type of `sampler` dispatches the implementation corresponding to each SMC implementation.
See `sample.jl` first to see how these methods are used. 
`rand_initial_with_potential` has a default implementation which is automatically used if it is not specialized.


## Examples
### Controlled SMC-ULA

```julia
include("csmc_ula.jl")
main()
```

![csmcula](https://github.com/Red-Portal/ControlledAIS/blob/main/demo/figures/csmc_ula.png)

 ### Controlled SMC-KLMC

```julia
include("csmc_klmc.jl")
main()
```

![csmcklmc](https://github.com/Red-Portal/ControlledAIS/blob/main/demo/figures/csmc_klmc.png)

