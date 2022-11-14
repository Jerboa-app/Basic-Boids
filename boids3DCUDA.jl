using CUDA

"""
    Map the pairwise force to a 2d kernel so at best we get one gpu thread
    calculating the force (or the abscence of it) between one pair of particles,
    with all pairs being done simultaneously!
"""
function ForceKernel(Y,F,T,R,k,N,t,rrep,ral,rat,L,blindAngle)
    index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x # starts at 1 for blockidx !
    stride_x = gridDim().x*blockDim().x

    index_y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    stride_y = gridDim().y*blockDim().y

    for i in index_x:stride_x:N
        for j in index_y:stride_y:N
            if (i == j)
                continue
            end
            r = 2*R # min dist by radii
            @inbounds rx = Y[t-1,j,1]-Y[t-1,i,1] # sep vector x
            @inbounds ry = Y[t-1,j,2]-Y[t-1,i,2] # sep vector y
            @inbounds rz = Y[t-1,j,3]-Y[t-1,i,3]
            if (L[1] > 0)
                # periodic use ``nearest image''
                if (rx > L[1]*0.5)
                    rx -= L[1]
                elseif (rx <= -L[1]*0.5)
                    rx += L[1]
                end

                if (ry > L[2]*0.5)
                    ry -= L[2]
                elseif (ry <= -L[2]*0.5)
                    ry += L[2]
                end

                if (rz > L[3]*0.5)
                    rz -= L[3]
                elseif (ry <= -L[3]*0.5)
                    rz += L[3]
                end
            end
            d2 = rx*rx+ry*ry+rz*rz
            if (d2 < r*r) # distance test without sqrt
                # apply a force to handle overlap
                d = CUDA.sqrt(d2) # need it now
                mag = -k*(r-d)

                @inbounds F[i,1] += mag*rx/d
                @inbounds F[i,2] += mag*ry/d
                @inbounds F[i,3] += mag*rz/d
            end

            d = CUDA.sqrt(d2)
            v = CUDA.sqrt(Y[t-1,j,4]*Y[t-1,j,4]+Y[t-1,j,5]*Y[t-1,j,5]+Y[t-1,j,6]*Y[t-1,j,6])
            angle = CUDA.acos((-rx*Y[t-1,j,4]-ry*Y[t-1,j,5]-rz*Y[t-1,j,6])/(v*d))
            if (angle < blindAngle)
                return
            end

            repel = d2 > 0 && d2 < rrep*rrep
            align = !repel && d2 > 0 && d2 < ral*ral
            attract = !repel && !align && d2 > 0 && d2 < rat*rat

            if (repel)
                d = CUDA.sqrt(d2)
                nx = -rx/d
                ny = -ry/d
                nz = -rz/d
                @inbounds T[i,1] += nx
                @inbounds T[i,2] += ny
                @inbounds T[i,3] += nz
            end

            if (align)
                # alignment
                v = CUDA.sqrt(Y[t-1,j,4]*Y[t-1,j,4]+Y[t-1,j,5]*Y[t-1,j,5]+Y[t-1,j,6]*Y[t-1,j,6])
                @inbounds T[i,4] += Y[t-1,j,4]/v
                @inbounds T[i,5] += Y[t-1,j,5]/v
                @inbounds T[i,6] += Y[t-1,j,6]/v
            end

            if (attract)
                # attraction
                d = CUDA.sqrt(d2)
                # cross product of i's veloctiy with ij seperation vector
                nx = rx/d
                ny = ry/d
                nz = rz/d
                @inbounds T[i,7] += nx
                @inbounds T[i,8] += ny
                @inbounds T[i,9] += nz
            end

        end
    end
end

"""
    Do an Euler-Maruyama step with the calculated forces and constants, 1d kernel
    so at best one thread for each particle.
"""
function StepKernel(Y,F,T,V,v0,Dr,Dt,repStr,alStr,atStr,dt,RAND,N,t)
    index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x # starts at 1 for blockidx !
    stride_x = gridDim().x*blockDim().x

    for i in index_x:stride_x:N
        # angular diffusion vector
        @inbounds dx = RAND[t-1,i,4]
        @inbounds dy = RAND[t-1,i,5]
        @inbounds dz = RAND[t-1,i,6]

        @inbounds crossX = Y[t-1,i,5]*dz-Y[t-1,i,6]*dy
        @inbounds crossY = Y[t-1,i,6]*dx-Y[t-1,i,4]*dz
        @inbounds crossZ = Y[t-1,i,4]*dy-Y[t-1,i,5]*dx

        @inbounds repMagnitude = T[i,1]*T[i,1]+T[i,2]*T[i,2]+T[i,3]*T[i,3]
        @inbounds alignMagnitude = T[i,4]*T[i,4]+T[i,5]*T[i,5]+T[i,6]*T[i,6]
        @inbounds attractMagnitude = T[i,7]*T[i,7]+T[i,8]*T[i,8]+T[i,9]*T[i,9]

        D = CUDA.sqrt(2.0*Dr*dt)

        @inbounds nx = Y[t-1,i,4]
        @inbounds ny = Y[t-1,i,5]
        @inbounds nz = Y[t-1,i,6]

        rotationStrength = 0.0

        # desired direction
        if (repMagnitude > 0)
            # priority to repel
            d = CUDA.sqrt(repMagnitude)
            @inbounds nx = T[i,1]/d
            @inbounds ny = T[i,2]/d
            @inbounds nz = T[i,3]/d
            rotationStrength = repStr
        else
            if (alignMagnitude > 0 && attractMagnitude > 0)
                al = CUDA.sqrt(alignMagnitude)
                at = CUDA.sqrt(attractMagnitude)
                @inbounds nx = (T[i,4]+T[i,7])/(al+at)
                @inbounds ny = (T[i,5]+T[i,8])/(al+at)
                @inbounds nz = (T[i,6]+T[i,9])/(al+at)
                rotationStrength = (alStr+atStr)/2.0
            elseif (alignMagnitude > 0 && attractMagnitude == 0)
                al = CUDA.sqrt(alignMagnitude)
                @inbounds nx = (T[i,4])/(al)
                @inbounds ny = (T[i,5])/(al)
                @inbounds nz = (T[i,6])/(al)
                rotationStrength = alStr
            elseif (alignMagnitude == 0 && attractMagnitude > 0)
                rotationStrength = atStr
                at = CUDA.sqrt(attractMagnitude)
                @inbounds nx = (T[i,7])/(at)
                @inbounds ny = (T[i,8])/(at)
                @inbounds nz = (T[i,9])/(at)
            end
        end

        # either fully rotate or partially to n
        # cross product
        vnorm = CUDA.sqrt(Y[t-1,i,4]*Y[t-1,i,4]+Y[t-1,i,5]*Y[t-1,i,5]+Y[t-1,i,6]*Y[t-1,i,6])
        @inbounds kx = Y[t-1,i,5]*nz-Y[t-1,i,6]*ny
        @inbounds ky = Y[t-1,i,6]*nx-Y[t-1,i,4]*nz
        @inbounds kz = Y[t-1,i,4]*ny-Y[t-1,i,5]*nx
        normK = CUDA.sqrt(kx*kx+ky*ky+kz*kz)
        kx /= normK
        ky /= normK
        kz /= normK
        # angle between
        cosTheta = (Y[t-1,i,4]*nx+Y[t-1,i,5]*ny+Y[t-1,i,6]*nz)/vnorm
        theta = CUDA.acos(cosTheta)

        if (theta <= rotationStrength*dt || rotationStrength == 0.0)
            # can fully rotate this time step, so just choose n (or not rotating at all)
        else
            # cannot fully rotate in one timestep, apply rodrigue's formula
            # to rotate about cross product, by Str*dt

            # use rodrigues' rotation formula

            @inbounds vx = Y[t-1,i,4]
            @inbounds vy = Y[t-1,i,5]
            @inbounds vz = Y[t-1,i,6]

            kCrossVx = ky*vz-kz*vy
            kCrossVy = kz*vx-kx*vz
            kCrossVz = kx*vy-ky*vx

            kDOTv = kx*vx+ky*vy+kz*vz

            kTimesKDotVx = kx*kDOTv
            kTimesKDotVy = ky*kDOTv
            kTimesKDotVz = kz*kDOTv

            cosTheta = CUDA.cos(rotationStrength*dt)
            sinTheta = CUDA.sin(rotationStrength*dt)

            nx = vx*cosTheta+kCrossVx*sinTheta+kTimesKDotVx*(1.0-cosTheta)
            ny = vy*cosTheta+kCrossVy*sinTheta+kTimesKDotVy*(1.0-cosTheta)
            nz = vz*cosTheta+kCrossVz*sinTheta+kTimesKDotVz*(1.0-cosTheta)
        end

        @inbounds Y[t,i,4] = nx + D*crossX
        @inbounds Y[t,i,5] = ny + D*crossY 
        @inbounds Y[t,i,6] = nz + D*crossZ

        @inbounds norm = Y[t,i,4]*Y[t,i,4]+Y[t,i,5]*Y[t,i,5]+Y[t,i,6]*Y[t,i,6]
        norm = CUDA.sqrt(norm)
        @inbounds Y[t,i,4] /= norm
        @inbounds Y[t,i,5] /= norm
        @inbounds Y[t,i,6] /= norm
        T[i,1]=0.0
        T[i,2]=0.0
        T[i,3]=0.0
        T[i,4]=0.0
        T[i,5]=0.0
        T[i,6]=0.0
        T[i,7]=0.0
        T[i,8]=0.0
        T[i,9]=0.0
        DT = CUDA.sqrt(2.0*Dt*dt)

        # save increments for boundary kernel
        @inbounds V[t,i,1] = dt * (v0*Y[t,i,4] + F[i,1]) + RAND[t-1,i,1]*DT
        @inbounds V[t,i,2] = dt * (v0*Y[t,i,5] + F[i,2]) + RAND[t-1,i,2]*DT
        @inbounds V[t,i,3] = dt * (v0*Y[t,i,6] + F[i,3]) + RAND[t-1,i,3]*DT
        # update (subject to boundary kernel)
        @inbounds Y[t,i,1] = Y[t-1,i,1] + V[t,i,1]
        @inbounds Y[t,i,2] = Y[t-1,i,2] + V[t,i,2]
        @inbounds Y[t,i,3] = Y[t-1,i,3] + V[t,i,3]

        # reset force while we are here
        F[i,1] = 0.0
        F[i,2] = 0.0
        F[i,3] = 0.0
    end
end

"""
    Keep particles in the box [0,L]x[0,L]x[0,L] by ``elastic collision''
"""
function BoundsKernel(Y,F,R,V,L,N,t,periodic=false)
    index_x = (blockIdx().x-1)*blockDim().x + threadIdx().x # starts at 1 for blockidx !
    stride_x = gridDim().x*blockDim().x

    for i in index_x:stride_x:N
        if (periodic)
            if (Y[t,i,1] < 0)
                Y[t,i,1] += L[1]
            elseif (Y[t,i,1] > L[1])
                Y[t,i,1] -= L[1]
            end

            if (Y[t,i,2] < 0)
                Y[t,i,2] += L[2]
            elseif (Y[t,i,2] > L[2])
                Y[t,i,2] -= L[2]
            end

            if (Y[t,i,3] < 0)
                Y[t,i,3] += L[3]
            elseif (Y[t,i,3] > L[3])
                Y[t,i,3] -= L[3]
            end
        else
            ux = 0. |> Float32 # be careful this is needed  for atan2...
            uy = 0. |> Float32
            uz = 0. |> Float32
            flag = false
            @inbounds vx = V[t,i,1]
            @inbounds vy = V[t,i,2]
            @inbounds vz = V[t,i,3]

            @inbounds OUTX = Y[t,i,1] < R || Y[t,i,1] > L[1]-R
            @inbounds OUTY = Y[t,i,2] < R || Y[t,i,2] > L[2]-R
            @inbounds OUTZ = Y[t,i,3] < R || Y[t,i,3] > L[3]-R

            if (OUTX)
                ux = -1.0*vx |> Float32
                flag = true
            end

            if (OUTY)
                uy = -1.0*vy |> Float32
                flag = true
            end

            if (OUTZ)
                uz = -1.0*vy |> Float32
            end

            if (flag)
                @inbounds Y[t,i,1] += ux
                @inbounds Y[t,i,2] += uy
                @inbounds Y[t,i,3] += uz
            end
        end
    end
end

"""
    Encapsulate GPU kernels in a function, computes data in batches to
        balance host/device memory shunting with memory on device.
"""
function GPUSteps(X,steps::Int=1;
    R=1.0,v0=1.0,dt=0.0166,Dt=0.0,Dr=0.005,L::Vector{Float32}=[10,10,10],k=300.,
    α=0.1,repStr=1.0,alStr=1.0,atStr=1.0,ral=10*R,rat=20*R,rrep=R*2,threads=256,periodic=false,
    blindAngle=0.0)
    Y = zeros(1+steps,size(X,1),size(X,2))
    Y[1,:,:] = X
    Y = Y|>cu
    V = zeros(size(Y))|>cu
    # forces
    F = zeros(size(Y,2),3)|>cu
    # torques
    T = zeros(size(Y,2),3*3)|>cu
    applyBounds = sum(L)>0
    L = L |> cu

    # curand not quite here yet?? https://discourse.julialang.org/t/how-to-generate-a-random-number-in-cuda-kernel-function/50364
    # will have to precompute random variables
    RAND = zeros(steps,size(X,1),8)
    if Dt > 0
        RAND[:,:,1:3] = randn(steps,size(X,1),3)
    end

    if Dr > 0
        RAND[:,:,4:6] = randn(steps,size(X,1),3)
    end

    RAND=RAND|>cu

    for i in 2:steps+1

        CUDA.@sync begin
            @cuda threads=threads blocks=ceil(Int,size(X,1)/threads) ForceKernel(Y,F,T,R,k,size(X,1),i,rrep,ral,rat,periodic*L,blindAngle)
        end

        CUDA.@sync begin
            @cuda threads=threads blocks=ceil(Int,size(X,1)/threads) StepKernel(Y,F,T,V,v0,Dr,Dt,repStr,alStr,atStr,dt,RAND,size(X,1),i)
        end

        if (applyBounds)
            CUDA.@sync begin
                @cuda threads=threads blocks=ceil(Int,size(X,1)/threads) BoundsKernel(Y,F,R,V,L,size(X,1),i,periodic)
            end
        end

    end

    return Y[2:end,:,:] |> Array
end
