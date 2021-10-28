# Introduction
 Bayesian Trend Filtering via Proximal MCMC

# Installation
ProxBTF.jl is dependent on Gurobi.jl. Before installing ProxBTF.jl, please see [Gurobi.jl](https://github.com/jump-dev/Gurobi.jl) for instructions on how to install Gurobi.jl. Once Gurobi.jl is installed, the following code can be used to install ProxBTF.jl from github. 
````
julia> using Pkg
julia> Pkg.add("https://github.com/qhengncsu/ProxBTF.jl")
````
Installing ProxBTF.jl from github will intruct Julia to install its dependencies at their latest releases, which might stop working as the dependencies get updated. To ensure reproducibility and avoid dependency hell, you can also clone this repository, cd the project directory and press ] in Julia REPL to enter Pkg mode. Then the following code and help you reproduce the exact same environment as that of the author.
````
pkg> activate .
pkg> instantiate
````
Once ProxBTF.jl is successfully set up using one of the two approaches mentioned above, 
````
julia> using ProxBTF
````
will help you bring the elements of the export list into the surrounding global namespace.

# Usage
Use function pbtf to fit a proximal Bayesian trend filtering model. For examples, see [Introduction to ProxBTF.jl](https://github.com/qhengncsu/ProxBTF.jl/blob/main/vignettes/Introduction%20to%20ProxBTF.jl.html). If you wish to run pbsrtf (proximal Bayesian shape-restricted trend filtering), you need a valid Gurobi license in your system. You can obtain an academic license for free from [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/).

# Issues
Feel free to contact the author at <qheng@ncsu.edu> for questions, issues and bugs.
 
