function visualize(result_quantile::DataFrame,problem::Union{pbtfProblem,pbsrtfProblem};legend_position::Symbol=:topleft)
    plt=plot(size=(800,600))
    scatter!(problem.x,problem.y,markersize=5,markercolor=:green,label="data",legendfontsize=20,
             xtickfontsize=20,ytickfontsize=20,markerstrokewidth=0,legend=legend_position)
    plot!(subplot=1,problem.xgrid,result_quantile[1:problem.n, Symbol("50.0%")],linewidth=5,label="posterior",c=:blue,
          ribbon=(result_quantile[1:problem.n, Symbol("50.0%")] .- result_quantile[1:problem.n, Symbol("2.5%")],
          result_quantile[1:problem.n, Symbol("97.5%")] .- result_quantile[1:problem.n, Symbol("50.0%")]),fillalpha=0.3)
end
