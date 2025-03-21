
using PosteriorDB, StanLogDensityProblems
using ProgressMeter

function main()
    if !isdir(".stan")
        mkdir(".stan")
    end
    pdb  = PosteriorDB.database()

    @showprogress for name in PosteriorDB.posterior_names(pdb)
        post = PosteriorDB.posterior(pdb, name)
        StanProblem(
            post, ".stan/";
            force=true,
            nan_on_error=true,
            make_args=["STAN_THREADS=true"],
        )
    end
end
