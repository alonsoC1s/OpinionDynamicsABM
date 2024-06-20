using OpinionDynamicsABM
using Documenter

DocMeta.setdocmeta!(OpinionDynamicsABM, :DocTestSetup, :(using OpinionDynamicsABM); recursive=true)

makedocs(;
    modules=[OpinionDynamicsABM],
    authors="Alonso Martínez Cisneros",
    sitename="OpinionDynamicsABM.jl",
    format=Documenter.HTML(;
        canonical="https://computationalhumanities.pages.zib.de/OpinionDynamicsABM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    repo = Remotes.GitLab("ComputationalHumanities", "OpinionDynamicsABM.jl"),
    checkdocs = :exports
)
