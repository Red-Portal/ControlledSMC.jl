var documenterSearchIndex = {"docs":
[{"location":"stan/#Stan-Models","page":"Stan Models","title":"Stan Models","text":"","category":"section"},{"location":"stan/","page":"Stan Models","title":"Stan Models","text":"using ControlledSMC\nusing Distributions\nusing FillArrays\nusing LinearAlgebra\nusing LogDensityProblems, StanLogDensityProblems\nusing Plots, StatsPlots\nusing PosteriorDB\nusing ProgressMeter\nusing Random, Random123\n\nfunction experiment_smcuhmc(rng, path, d, n_particles, n_reps, ylims)\n    stepsizes = 10.0.^range(-4, -1; length=8)\n    dampings  = [0.05, 0.1, 0.15, 0.2]\n\n    for α in dampings\n        ℓZs = @showprogress map(stepsizes) do ϵ\n            sampler = SMCUHMC(ϵ, α, Eye(d))\n            mean(1:n_reps) do k\n                try \n                    rng_local = deepcopy(rng)\n                    set_counter!(rng_local, k)\n                    _, _, _, stats = ControlledSMC.sample(\n                        rng_local, sampler, path, n_particles, 0.5; show_progress=false\n                    )\n                    last(stats).log_normalizer\n                catch e\n                    if occursin(\"log_density\", e.msg)\n                        -10^10\n                    else\n                        throw(e)\n                    end\n                end\n            end\n        end\n        Plots.plot!(\n            stepsizes, ℓZs;\n            label=\"uMHC α = $(α)\",\n            xscale=:log10,\n            ylims,\n            xlabel=\"stepsizes\",\n            ylabel=\"logZ\",\n        ) |> display\n    end\nend\n\nfunction experiment_smcklmc(rng, path, d, n_particles, n_reps, ylims)\n    stepsizes = 10.0.^range(-5, -2; length=8)\n    dampings  = [100., 500., 1000., 5000]\n\n    for γ in dampings\n        ℓZs = @showprogress map(stepsizes) do h\n            sampler = SMCKLMC(γ*h, γ)\n            mean(1:n_reps) do k\n                try \n                    rng_local = deepcopy(rng)\n                    set_counter!(rng_local, k)\n                    _, _, _, stats = ControlledSMC.sample(\n                        rng_local, sampler, path, n_particles, 0.5; show_progress=false\n                    )\n                    last(stats).log_normalizer\n                catch e\n                    if occursin(\"log_density\", e.msg)\n                        -10^10\n                    else\n                        throw(e)\n                    end\n                end\n            end\n        end\n        Plots.plot!(\n            stepsizes, ℓZs;\n            label=\"KLMC γ = $(γ)\",\n            xscale=:log10,\n            ylims,\n            xlabel=\"stepsizes\",\n            ylabel=\"logZ\",\n        ) |> display\n    end\nend\n\nfunction experiment_smcula(rng, path, d, n_particles, n_reps, ylims)\n    ula_stepsizes = 10.0.^range(-5, -2; length=4)\n\n    ℓZs = @showprogress map(ula_stepsizes) do h\n        sampler = SMCULA(h, h, TimeCorrectForwardKernel(), Eye(d), path)\n        mean(1:n_reps) do k\n            try \n                rng_local = deepcopy(rng)\n                set_counter!(rng_local, k)\n                _, _, _, stats = ControlledSMC.sample(\n                    rng_local, sampler, path, n_particles, 0.5; show_progress=false\n                )\n                last(stats).log_normalizer\n            catch e\n                if occursin(\"log_density\", e.msg)\n                    -10^10\n                else\n                    throw(e)\n                end\n            end\n        end\n    end\n    Plots.plot!(\n        ula_stepsizes, ℓZs;\n        label=\"ULA\",\n        xscale=:log10,\n        ylims,\n        xlabel=\"ULA\",\n        ylabel=\"logZ\",\n    ) |> display\nend\n\nfunction run_stanmodel(name, ylims)\n    pdb  = PosteriorDB.database()\n    post = PosteriorDB.posterior(pdb, name)\n    prob = StanProblem(post, \".stan/\"; force=true)\n    d    = LogDensityProblems.dimension(prob)\n\n    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)\n    rng  = Philox4x(UInt64, seed, 8)\n\n    n_reps      = 8\n    n_iters     = 32\n    proposal    = MvNormal(Zeros(d), I)\n    schedule    = range(0, 1; length=n_iters) .^ 4\n    path        = GeometricAnnealingPath(schedule, proposal, prob)\n    n_particles = 256\n\n    Plots.plot() |> display\n    experiment_smcula( rng, path, d, n_particles, n_reps, ylims)\n    experiment_smcuhmc(rng, path, d, n_particles, n_reps, ylims)\n    experiment_smcklmc(rng, path, d, n_particles, n_reps, ylims)\nend\n\nif !isdir(\".stan\")\n    mkdir(\".stan\")\nend","category":"page"},{"location":"stan/","page":"Stan Models","title":"Stan Models","text":"run_stanmodel(\"radon_mn-radon_hierarchical_intercept_noncentered\", (-3000, Inf))\nPlots.savefig(\"vanilla_smc_radon.svg\")","category":"page"},{"location":"stan/","page":"Stan Models","title":"Stan Models","text":"(Image: )","category":"page"},{"location":"stan/","page":"Stan Models","title":"Stan Models","text":"run_stanmodel(\"dogs-dogs\", (-300, Inf))\nPlots.savefig(\"vanilla_smc_dogs-dogs.svg\")","category":"page"},{"location":"stan/","page":"Stan Models","title":"Stan Models","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ControlledSMC","category":"page"},{"location":"#ControlledSMC","page":"Home","title":"ControlledSMC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ControlledSMC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ControlledSMC]","category":"page"}]
}
