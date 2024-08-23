
# Controlled Sequential Monte Carlo 
## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.update()
```

## Project Structure
* `sample.jl`: The generic sampling routines.
* `commonl.jl`: Commonly shared utilities.
* `twist.jl`: Twisting-specific routines.

* `smc_ula.jl`: SMC with an ULA kernel implementation.
* `smc_klmc.jl`: SMC with a KLMC kernel implementation.
* `csmc_ula.jl`: Controlled SMC with an ULA kernel implementation.
* `csmc_klmc.jl`: Controlled SMC with a KLMC kernel implementation.

## Examples
### Controlled SMC-ULA

```julia
include("csmc_ula.jl")
main()
```

![csmcula](https://github.com/Red-Portal/ControlledAIS/blob/main/demo/figures/csmc_ula.png)

 SMC-ULA

```julia
include("csmc_klmc.jl")
main()
```

![csmcklmc](https://github.com/Red-Portal/ControlledAIS/blob/main/demo/figures/csmc_klmc.png)

