module Kalman

using Utils, StaticArrays

@withkw mutable struct KalmanFilter{N, M, T}
    x::SVector{N, T} = zeros(N) # state mean vector
    P::SMatrix{N, N, T} = eye(N) # state covariance matrix
    F::SMatrix{N, N, T} = eye(N) # transition matrix
    u::SVector{N, T} = zeros(N) # transition offset vector
    Q::SMatrix{N, N, T} = eye(N) # transition covariance matrix
    H::SMatrix{M, N, T} = eye(M, N)  # observation matrix
    R::SMatrix{M, M, T} = eye(M, M) # observation covariance matrix
end

@inline @replace function measure_update!(kf::KalmanFilter, z)
    y = z - H * x
    PH = P * H.'
    S = R + H * PH
    U = chol(S)
    M = PH / U
    v = U.' \ y
    x += M * v
    P -= M * M.'
    return x, P
end

@inline @replace function predict_update!(kf::KalmanFilter)
    x = F * x + u
    P = F * P * F.' + Q
    return x, P
end

function kalman_filter!(kf::KalmanFilter, zs)
    xs = similar(zs, (length(kf.x), size(zs, 2)))
    for i in 1:size(zs, 2)
        xs[:, i], ~ = measure_update!(kf, view(zs, :, i))
        predict_update!(kf)
    end
    return xs
end

kalman_filter(kf::KalmanFilter, zs) = kalman_filter!(deepcopy(kf), zs)

measure_update!(kf::KalmanFilter{N, 1, T}, z) where {N, T} =  measure_update1!(kf, z)

@inline @replace function measure_update1!(kf::KalmanFilter, z)
    y = z - H * x
    PH = P * H.'
    S = R + H * PH
    U = sqrt(S[1])
    M = PH / U
    v = y / U
    x += M * v
    P -= M * M.'
    return x, P
end

function kalman_filter!(kf::KalmanFilter{1, 1, T}, zs) where T
    xs = similar(zs, (length(kf.x), size(zs, 2)))
    x, P, F, u, Q, H, R = kf.x[1], kf.P[1], kf.F[1], kf.u[1], kf.Q[1], kf.H[1], kf.R[1]
    for i in 1:size(zs, 2)
        z = zs[1, i]
        y = z - H * x
        S = R + H * P * H
        K = P * H / S
        x = x + K * y
        P = P - K * H * P
        xs[1, i] = x
        x = F * x + u
        P = F * P * F + Q
    end
    kf.x, kf.P = T[x], T[P]
    return xs
end

function kalman_filter!(kf::KalmanFilter{2, 1, T}, zs) where T
    xs = similar(zs, (length(kf.x), size(zs, 2)))
    x1, x2 = kf.x
    u1, u2 = kf.u
    H1, H2 = kf.H
    R, = kf.R
    P11, P21, P12, P22 = kf.P
    F11, F21, F12, F22 = kf.F
    Q11, Q21, Q12, Q22 = kf.Q
    for i in 1:size(zs, 2)
        z = zs[1, i]
        y = z - H1 * x1 + H2 * x2
        HP1 = H1 * P11 + H2 * P21
        HP2 = H1 * P12 + H2 * P22
        PH1 = P11 * H1 + P12 * H2
        PH2 = P21 * H1 + P22 * H2
        S = R + HP1 * H1 + HP2 * H2
        K1 = PH1 / S
        K2 = PH2 / S
        x1 = x1 + K1 * y
        x2 = x2 + K2 * y
        P11 -= K1 * HP1
        P21 -= K2 * HP1
        P12 -= K1 * HP2
        P22 -= K2 * HP2
        xs[1, i] = x1
        xs[2, i] = x2
        x1 = F11 * x1 + F12 * x2 + u1
        x2 = F21 * x1 + F22 * x2 + u2
        PF11 = P11 * F11 + P12 *F12
        PF21 = P21 * F11 + P22 * F12
        PF12 = P11 * F21 + P12 * F22
        PF22 = P21 * F21 + P22 * F22
        P11 = F11 * PF11 + F12 * PF12 + Q11
        P21 = F21 * PF11 + F22 * PF12 + Q21
        P12 = F11 * PF21 + F12 * PF22 + Q12
        P22 = F21 * PF21 + F22 * PF22 + Q22
    end
    kf.x, kf.P = T[x1, x2], T[P11 P12; P21 P22]
    return xs
end

end