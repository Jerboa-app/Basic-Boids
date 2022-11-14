using GLMakie, ProgressMeter, LinearAlgebra, DelimitedFiles, ArgParse, Statistics, Distances
include("boids3DCUDA.jl")
set_theme!(font="OpenDyslexic")

s = ArgParseSettings()

@add_arg_table! s begin

    "--name"
    help = "movie name"
    arg_type = String
    default = "boids"

    "--upscaling"
	help = "x times increase to 1920x1080 output resolution"
	arg_type = Int
	default = 1

    "-N"
    help = "number of particles"
    arg_type = Int
    default = 500

    "-T"
    help = "number of frames"
    arg_type = Int
    default = 60*30

    "-L"
    help = "box size  (multiples of particle size)"
    arg_type = Float64
    default = 0.0

    "--speed","-v"
    help = "speed (multiples of particle size / second)"
    arg_type = Float64
    default = 6.0

    "--noise","-D"
    help = "rotational noise"
    arg_type = Float64
    default = 0.1

    "--attractionStrength"
    help = "attraction turning rate radian/sec"
    arg_type = Float64
    default = 1.0

    "--attractionDistance"
    help = "radius of attraction (multiples of particle size)"
    arg_type = Float64
    default = 10.0

    "--alignmentStrength"
    help = "alignment turning rate radian/sec"
    arg_type = Float64
    default = 1.0

    "--alignmentDistance"
    help = "radius of alignment (multiples of particle size)"
    arg_type = Float64
    default = 5.0

    "--repulsionStrength"
    help = "repulsion turning rate radian/sec"
    arg_type = Float64
    default = 1.0

    "--repulsionDistance"
    help = "radius of repulsion (multiples of particle size)"
    arg_type = Float64
    default = 1.0

    "--blindAngle"
    help = "angle behind boid it cannot see (degrees)"
    arg_type = Float64
    default = 45.0

    "--watch"
    help = "run in interactive mode"
    arg_type = Bool
    default = true

    "--showAxes"
    help = "show xyz axes"
    arg_type = Bool
    default = true

    "--rotateCamera"
    help = "continuously rotate camera"
    arg_type = Bool
    default = true
end

args = parse_args(s)

@info args

N = args["N"]
upscale = args["upscaling"]
watch = args["watch"]

if !watch
    # this will tell GLMakie to go as fast as possible
    GLMakie.set_window_config!(framerate = Inf, vsync = false)
    # this will suppress the plot window = faster plots + does not takeover your pc
    # trying to plot and display!
    GLMakie.inline!(true)
end

if upscale > 1 && watch
    @warn "if your monitor is not $(1920*upscale)x$(1080*upscale), actual resolution may be limited"
    @info "run with --watch false, to correct this"
end
rStr = args["repulsionStrength"]
rR = args["repulsionDistance"]
alStr = args["alignmentStrength"]
alR = args["alignmentDistance"]
atStr = args["attractionStrength"]
atR = args["attractionDistance"]
name = args["name"]
blindAngle = args["blindAngle"]
camRotate = args["rotateCamera"]

function order(X)
    v = zeros(size(X,2))
    for i in 1:size(X,1)
        v += X[i,:]/norm(X[i,:])
    end
    return norm(v./size(X,1))
end

T = args["T"]
L = args["L"]
Dr = args["noise"]
v0 = args["speed"]

X = zeros(N,6)
p = 0.1

rho = 100/35^3
l = (N/rho)^(1/3)
for i in 1:N
    X[i,1:3] = rand(3)*(l-4).+2
    X[i,4:6] = rand(3)
    X[i,4:6] ./= norm(X[i,4:6])
end

# burn in intial collisions
@showprogress for k in 1:10
    global X = GPUSteps(
        X,
        1,
        L=Float32.([L,L,L]),
        v0=v0,
        Dr=Dr,
        repStr=0.0,
        alStr=0.0,
        atStr=0.0,
        periodic= L>0 ? true : false
    )[end,:,:]
end

Traj = zeros(T,size(X,1),size(X,2))
Traj[1,:,:] = X;
@info "Simulating"
@showprogress for t in 1:(T-1)
    Traj[t+1,:,:] = GPUSteps(
        Traj[t,:,:],
        10,
        dt=1.0/60.0/10.0,
        k=0.0,
        L=Float32.([L,L,L]),
        v0=v0,
        Dr=Dr,
        repStr=rStr,
        alStr=alStr,
        atStr=atStr,
        rrep=rR,
        ral=alR,
        rat=atR,
        blindAngle=deg2rad(blindAngle),
        periodic= L>0 ? true : false
    )[end,:,:]
end

fig = Figure(resolution=(1920*upscale,1080*upscale))
p=Node([Point3f0(Traj[1,i,1:3]...) for i in 1:N])
n=Node([Point3f0(Traj[1,i,4:6]...) for i in 1:N]);
ax = Axis3(fig[1,1],limits=(0,L,0,L,0,L),viewmode=:fit,
    titlesize=40*upscale,xlabelsize=40*upscale,
    ylabelsize=40*upscale,zlabelsize=40*upscale,xticklabelsize=16*upscale,yticklabelsize=16*upscale,
    zticklabelsize=16*upscale
)
ax.aspect = (1,1,1)
ax.elevation[] = 0.5
L=20

if (!args["showAxes"])
    hidedecorations!(ax)
    hidespines!(ax)
end


prog = Progress(size(Traj,1)-1)
arrows!(ax,p,n,markersize=(2,2,2),markerspace=SceneSpace)
@info "Rendering"
GLMakie.record(fig, "$name.mp4", collect(2:1:size(Traj,1)), framerate=60) do i
    p[] = [Point3f0(Traj[i,j,1:3]...) for j in 1:size(Traj,2)]
    n[] = [Point3f0(Traj[i,j,4:6]...) for j in 1:size(Traj,2)]
    comx = mean(Traj[i,:,1])
    comy = mean(Traj[i,:,2])
    comz = mean(Traj[i,:,3])
    ax.limits[] = (comx-L/2.0,comx+L/2.0,comy-L/2.0,comy+L/2.0,comz-L/2.0,comz+L/2.0)
    if camRotate
        ax.azimuth[] = ax.azimuth[] + 1.0/60.0 * pi/8
    end
    next!(prog)
end