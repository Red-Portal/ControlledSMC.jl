using ControlledSMC
using Documenter

DocMeta.setdocmeta!(ControlledSMC, :DocTestSetup, :(using ControlledSMC); recursive=true)

makedocs(;
    modules=[ControlledSMC],
    authors="Kyurae Kim <kyrkim@seas.upenn.edu> and contributors",
    sitename="ControlledSMC.jl",
    format=Documenter.HTML(;
        canonical="https://Red-Portal.github.io/ControlledSMC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md", "Kernel Adaptation" => "kernel_adaptation.md"],
)

deploydocs(; repo="github.com/Red-Portal/ControlledSMC.jl", devbranch="main")
