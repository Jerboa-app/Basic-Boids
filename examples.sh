# mileage may vary, but these arre generally quite interesting parameter sets

julia-1.5 -O3 render.jl -N 100 -D 0.1 -v 6.0 --alignmentDistance 3.0 --attractionDistance 18.0 --blindAngle 45.0 --name ex1 --upscaling 2 --watch false -T 7200 
julia-1.5 -O3 render.jl -N 100 -D 0.1 -v 6.0 --alignmentDistance 12.0 --attractionDistance 18.0 --blindAngle 45.0 --name ex2 --upscaling 2 --watch false -T 7200 
julia-1.5 -O3 render.jl -N 100 -D 0.1 -v 10.0 --alignmentDistance 3.0 --attractionDistance 18.0 --blindAngle 45.0 --repulsionStrength 2.0 --alignmentStrength 2.0 --attractionStrength 2.0 --name ex3 --upscaling 2 --watch false -T 7200 
julia-1.5 -O3 render.jl -N 500 -D 0.1 -v 10.0 --alignmentDistance 10.0 --attractionDistance 50.0 --blindAngle 45.0 --repulsionStrength 2.0 --alignmentStrength 2.0 --attractionStrength 2.0 --name ex4 --upscaling 2 --watch false -T 7200 
julia-1.5 -O3 render.jl -N 500 -D 0.1 -v 10.0 --alignmentDistance 3.0 --attractionDistance 50.0 --blindAngle 90.0 --repulsionStrength 2.0 --alignmentStrength 2.0 --attractionStrength 2.0 --name ex5 --upscaling 2 --watch false -T 7200 
julia-1.5 -O3 render.jl -N 500 -D 0.3 -v 10.0 --alignmentDistance 3.0 --attractionDistance 50.0 --blindAngle 90.0 --repulsionStrength 2.0 --alignmentStrength 2.0 --attractionStrength 2.0 --name ex6 --upscaling 2 --watch false -T 7200
