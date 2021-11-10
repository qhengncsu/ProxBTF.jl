using ProxBTF
σ   = 3# noise s.d.
x   = [1:1.:100;]
fx  = 13 .* sin.((4*π/100).*x)
y = fx + σ .* randn(100)
result_summary,result_quantile,problem,elapse_time=pbtf(y,x,2,nsample=3000)
visualize(result_quantile,problem,legend_position=:top)
σ   = 1.0# noise s.d.
x   = [0.1:0.1:10;]
fx  = x .+ sin.(x)
y = fx + σ .* randn(100)
result_summary,result_quantile,problem,elapse_time=pbsrtf(y,x,2,"increasing")
visualize(result_quantile,problem)
fx = map(xi -> 0<= xi <=2 ? 10-5xi : 2<xi <= 8 ? 0 : 5xi-40,x)
y = fx + σ .* randn(100)
result_summary,result_quantile,problem,elapse_time=pbsrtf(y,x,1,"convex")
visualize(result_quantile,problem)
fx = map(xi -> 0<= xi <= 8 ? 0 : (xi-8)^3,x)
y = fx + σ .* randn(100)
result_summary,result_quantile,problem,elapse_time=pbsrtf(y,x,2,"inc-convex")
visualize(result_quantile,problem)
