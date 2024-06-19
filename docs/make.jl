using OpinionDynamicsABM
using Documenter

DocMeta.setdocmeta!(OpinionDynamicsABM, :DocTestSetup, :(using OpinionDynamicsABM); recursive=true)

makedocs(;
    modules=[OpinionDynamicsABM],
    authors="Alonso Martínez Cisneros",
    sitename="OpinionDynamicsABM.jl",
    format=Documenter.HTML(;
        canonical="https://amartine.gitlab.io/OpinionDynamicsABM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
