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
will help you bring the elements of the export list into the surrounding global namespace, namely pbtf and pbsrtf.

# Using Jupyternotebook
To be able to use Jupyternotebook with a Julia kernel, you'll need to install IJulia.
````
Pkg.add("IJulia")
````

# A light-weight alternative
In consideration of the fact that ProxBTF.jl is heavy-weight due to its dependence on Gurobi.jl and Convex.jl, we provide an alternative way of using our implementation. Everything is packed into a single, self-contained script. See [minimum_reproduce.jl](https://github.com/qhengncsu/ProxBTF.jl/blob/main/vignettes/minimum_reproduce.jl). However, the script does not contain code for Proximal Bayesian Shape-Restriced Trend Filtering. If you wish to use pbsrtf, then it is necessary to install ProxBTF.jl per the instructions above.

# Tutorials
For a tutorial of using the light-weight implementation, see [Introduction to pbtf](https://github.com/qhengncsu/ProxBTF.jl/blob/main/vignettes/Introduction_to_pbtf.ipynb). For a tutorial of using the whole package ProxBTF.jl, see [Introduction to ProxBTF.jl](https://github.com/qhengncsu/ProxBTF.jl/blob/main/vignettes/Introduction_to_ProxBTF.jl.ipynb). Notice that if you wish to run pbsrtf, you need a valid Gurobi license on your machine. You can obtain an academic license for free from [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/).

# Issues
Feel free to contact the author at <qheng@ncsu.edu> for questions, issues and comments.
 
