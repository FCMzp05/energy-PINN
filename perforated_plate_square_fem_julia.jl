using LinearAlgebra
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Printf
using SparseArrays
using Statistics

Base.@kwdef mutable struct Config
    hardening_mode::String = "isotropic"
    width::Float64 = 200.0
    height::Float64 = 200.0
    radius::Float64 = 50.0
    thickness::Float64 = 100.0
    young::Float64 = 7.0e4
    poisson::Float64 = 0.20
    yield_stress::Float64 = 250.0
    tangent_modulus::Float64 = 2171.0
    iso_q1::Float64 = -216.9135
    iso_b1::Float64 = 213.9273
    kin_c1::Float64 = 58791.656
    kin_gamma1::Float64 = 147.7362
    kin_c2::Float64 = 1803.7759
    kin_gamma2::Float64 = 0.0
    top_displacement::Float64 = 2.0
    load_steps::Int = 20
    ntheta::Int = 80
    nradial::Int = 32
    radial_bias::Float64 = 1.8
    newton_tol::Float64 = 1.0e-10
    newton_max_iter::Int = 80
    tangent_eps::Float64 = 1.0e-8
    return_tol::Float64 = 1.0e-10
    return_max_iter::Int = 80
    print_every::Int = 5
    save_plots::Int = 1
    plot_dpi::Int = 160
    output_dir::String = ""
end

Base.@kwdef mutable struct MaterialState
    eps_p::Matrix{Float64} = zeros(3, 3)
    p_eq::Float64 = 0.0
    x1::Matrix{Float64} = zeros(3, 3)
    x2::Matrix{Float64} = zeros(3, 3)
end

struct MeshData
    nodes::Matrix{Float64}
    triangles::Matrix{Int}
    left_nodes::Vector{Int}
    bottom_nodes::Vector{Int}
    top_nodes::Vector{Int}
    right_nodes::Vector{Int}
    hole_nodes::Vector{Int}
    top_edges::Matrix{Int}
end

function parse_cli!(cfg::Config)
    i = 1
    while i <= length(ARGS)
        key = ARGS[i]
        !startswith(key, "--") && error("Unknown argument: $key")
        i == length(ARGS) && error("Missing value for $key")
        val = ARGS[i + 1]
        name = Symbol(replace(key[3:end], "-" => "_"))
        hasproperty(cfg, name) || error("Unsupported option: $key")
        old = getproperty(cfg, name)
        if old isa Int
            setproperty!(cfg, name, parse(Int, val))
        elseif old isa Float64
            setproperty!(cfg, name, parse(Float64, val))
        elseif old isa String
            setproperty!(cfg, name, val)
        else
            error("Unsupported option type for $key")
        end
        i += 2
    end
    if isempty(cfg.output_dir)
        cfg.output_dir = joinpath(@__DIR__, "outputs", "perforated_plate_square_fem")
    end
    return cfg
end

function hardening_modulus(cfg::Config)
    ce = cfg.young
    ct = cfg.tangent_modulus
    abs(ce - ct) < 1.0e-12 && return 0.0
    return ct * ce / (ce - ct)
end

function elastic_matrix(cfg::Config)
    lam = cfg.young * cfg.poisson / ((1.0 + cfg.poisson) * (1.0 - 2.0 * cfg.poisson))
    mu = cfg.young / (2.0 * (1.0 + cfg.poisson))
    return [
        lam + 2.0 * mu lam lam 0.0
        lam lam + 2.0 * mu lam 0.0
        lam lam lam + 2.0 * mu 0.0
        0.0 0.0 0.0 2.0 * mu
    ]
end

function shear_modulus(cfg::Config)
    return cfg.young / (2.0 * (1.0 + cfg.poisson))
end

function deviatoric(sig::AbstractVector)
    mean_sig = (sig[1] + sig[2] + sig[3]) / 3.0
    return [sig[1] - mean_sig, sig[2] - mean_sig, sig[3] - mean_sig, sig[4]]
end

function von_mises(sig::AbstractVector)
    sx, sy, sz, sxy = sig
    return sqrt(max(0.5 * ((sx - sy)^2 + (sy - sz)^2 + (sz - sx)^2) + 3.0 * sxy^2, 0.0))
end

function outer_radius(cfg::Config, theta::Float64)
    cx = cos(theta)
    sy = sin(theta)
    rx = abs(cx) < 1.0e-12 ? Inf : cfg.width / cx
    ry = abs(sy) < 1.0e-12 ? Inf : cfg.height / sy
    return min(rx, ry)
end

node_id(ir::Int, it::Int, nradial::Int) = ir + 1 + it * (nradial + 1)

function build_polar_mesh(cfg::Config)
    nnode = (cfg.nradial + 1) * (cfg.ntheta + 1)
    nodes = zeros(Float64, nnode, 2)
    thetas = range(0.0, 0.5 * pi, length = cfg.ntheta + 1)
    for (jt, theta) in enumerate(thetas)
        rmax = outer_radius(cfg, theta)
        for ir in 0:cfg.nradial
            alpha = (ir / cfg.nradial)^cfg.radial_bias
            radius = cfg.radius + alpha * (rmax - cfg.radius)
            nid = node_id(ir, jt - 1, cfg.nradial)
            nodes[nid, 1] = radius * cos(theta)
            nodes[nid, 2] = radius * sin(theta)
        end
    end

    triangles = zeros(Int, 2 * cfg.nradial * cfg.ntheta, 3)
    tid = 1
    for jt in 0:(cfg.ntheta - 1)
        for ir in 0:(cfg.nradial - 1)
            n0 = node_id(ir, jt, cfg.nradial)
            n1 = node_id(ir + 1, jt, cfg.nradial)
            n2 = node_id(ir, jt + 1, cfg.nradial)
            n3 = node_id(ir + 1, jt + 1, cfg.nradial)
            triangles[tid, :] .= (n0, n1, n3)
            triangles[tid + 1, :] .= (n0, n3, n2)
            tid += 2
        end
    end

    tol = 1.0e-8
    left_nodes = Int[]
    bottom_nodes = Int[]
    top_nodes = Int[]
    right_nodes = Int[]
    hole_nodes = Int[]
    outer_nodes = [node_id(cfg.nradial, jt, cfg.nradial) for jt in 0:cfg.ntheta]
    for nid in 1:size(nodes, 1)
        x = nodes[nid, 1]
        y = nodes[nid, 2]
        r = hypot(x, y)
        abs(x) < tol && push!(left_nodes, nid)
        abs(y) < tol && push!(bottom_nodes, nid)
        abs(y - cfg.height) < tol && push!(top_nodes, nid)
        abs(x - cfg.width) < tol && push!(right_nodes, nid)
        abs(r - cfg.radius) < 5.0e-7 && push!(hole_nodes, nid)
    end

    edge_buf = Vector{NTuple{2, Int}}()
    for k in 1:(length(outer_nodes) - 1)
        n1 = outer_nodes[k]
        n2 = outer_nodes[k + 1]
        if abs(nodes[n1, 2] - cfg.height) < tol && abs(nodes[n2, 2] - cfg.height) < tol
            push!(edge_buf, (n1, n2))
        end
    end
    top_edges = zeros(Int, length(edge_buf), 2)
    for (i, edge) in enumerate(edge_buf)
        top_edges[i, 1] = edge[1]
        top_edges[i, 2] = edge[2]
    end

    return MeshData(
        nodes,
        triangles,
        unique(sort(left_nodes)),
        unique(sort(bottom_nodes)),
        unique(sort(top_nodes)),
        unique(sort(right_nodes)),
        unique(sort(hole_nodes)),
        top_edges,
    )
end

function build_tri_operators(mesh::MeshData)
    ne = size(mesh.triangles, 1)
    b_mats = zeros(Float64, ne, 4, 6)
    areas = zeros(Float64, ne)
    for e in 1:ne
        tri = mesh.triangles[e, :]
        xy = mesh.nodes[tri, :]
        x1, y1 = xy[1, 1], xy[1, 2]
        x2, y2 = xy[2, 1], xy[2, 2]
        x3, y3 = xy[3, 1], xy[3, 2]
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(area2)
        beta = [y2 - y3, y3 - y1, y1 - y2]
        gamma = [x3 - x2, x1 - x3, x2 - x1]
        dndx = beta ./ area2
        dndy = gamma ./ area2
        b = zeros(Float64, 4, 6)
        for a in 1:3
            b[1, 2 * a - 1] = dndx[a]
            b[2, 2 * a] = dndy[a]
            b[4, 2 * a - 1] = 0.5 * dndy[a]
            b[4, 2 * a] = 0.5 * dndx[a]
        end
        b_mats[e, :, :] .= b
        areas[e] = area
    end
    return b_mats, areas
end

function build_element_dofs(mesh::MeshData)
    ne = size(mesh.triangles, 1)
    edof = zeros(Int, ne, 6)
    for e in 1:ne
        tri = mesh.triangles[e, :]
        edof[e, :] .= (
            2 * tri[1] - 1, 2 * tri[1],
            2 * tri[2] - 1, 2 * tri[2],
            2 * tri[3] - 1, 2 * tri[3],
        )
    end
    return edof
end

function build_constraints(mesh::MeshData, target_uy::Float64)
    bc = Dict{Int, Float64}()
    for nid in mesh.left_nodes
        bc[2 * nid - 1] = 0.0
    end
    for nid in mesh.bottom_nodes
        bc[2 * nid] = 0.0
    end
    for nid in mesh.top_nodes
        bc[2 * nid] = target_uy
    end
    return bc
end

function gather_sets(ndof::Int, bc::Dict{Int, Float64})
    fixed = sort!(collect(keys(bc)))
    fixed_vec = Vector{Int}(fixed)
    free = setdiff(collect(1:ndof), fixed_vec)
    return fixed_vec, free
end

function strain_vec_to_tensor(epsv::AbstractVector)
    eps = zeros(Float64, 3, 3)
    eps[1, 1] = epsv[1]
    eps[2, 2] = epsv[2]
    eps[3, 3] = epsv[3]
    eps[1, 2] = epsv[4]
    eps[2, 1] = epsv[4]
    return eps
end

function tensor_to_vec(sig::AbstractMatrix)
    return [sig[1, 1], sig[2, 2], sig[3, 3], sig[1, 2]]
end

function deviatoric(a::Matrix{Float64})
    return a .- (tr(a) / 3.0) .* Matrix{Float64}(I, 3, 3)
end

function j2_norm(a::Matrix{Float64})
    return sqrt(max(1.5 * sum(a .* a), 0.0))
end

function elastic_stress(cfg::Config, eps::Matrix{Float64}, eps_p::Matrix{Float64})
    lam = cfg.young * cfg.poisson / ((1.0 + cfg.poisson) * (1.0 - 2.0 * cfg.poisson))
    mu = cfg.young / (2.0 * (1.0 + cfg.poisson))
    ee = eps - eps_p
    return lam * tr(ee) .* Matrix{Float64}(I, 3, 3) .+ 2.0 * mu .* ee
end

iso_radius(cfg::Config, p_eq::Float64) = cfg.iso_q1 * (1.0 - exp(-cfg.iso_b1 * p_eq))
yield_radius(cfg::Config, p_eq::Float64) = cfg.yield_stress + iso_radius(cfg, p_eq)

function isotropic_update(total_strain::Vector{Float64}, state::MaterialState, cfg::Config, cmat::Matrix{Float64}, mu::Float64, hmod::Float64)
    eps_p_prev = tensor_to_vec(state.eps_p)
    sigma_trial = cmat * (total_strain - eps_p_prev)
    s_trial = deviatoric(sigma_trial)
    seq_trial = von_mises(sigma_trial)
    fy = seq_trial - (cfg.yield_stress + hmod * state.p_eq)
    if fy <= 0.0 || seq_trial <= 1.0e-12
        return sigma_trial, MaterialState(copy(state.eps_p), state.p_eq, copy(state.x1), copy(state.x2))
    end
    dgamma = fy / (3.0 * mu + hmod)
    flow = (1.5 / seq_trial) .* s_trial
    eps_p_new_vec = eps_p_prev .+ dgamma .* flow
    p_eq_new = state.p_eq + dgamma
    mean_trial = (sigma_trial[1] + sigma_trial[2] + sigma_trial[3]) / 3.0
    factor = 1.0 - 3.0 * mu * dgamma / seq_trial
    s_new = factor .* s_trial
    sigma_new = [s_new[1] + mean_trial, s_new[2] + mean_trial, s_new[3] + mean_trial, s_new[4]]
    return sigma_new, MaterialState(strain_vec_to_tensor(eps_p_new_vec), p_eq_new, copy(state.x1), copy(state.x2))
end

function mixed_hardening_update(total_strain::Vector{Float64}, state::MaterialState, cfg::Config)
    mu = cfg.young / (2.0 * (1.0 + cfg.poisson))
    eps = strain_vec_to_tensor(total_strain)
    sigma_trial = elastic_stress(cfg, eps, state.eps_p)
    s_trial = deviatoric(sigma_trial)
    x_old = state.x1 + state.x2
    f_trial = j2_norm(s_trial - x_old) - yield_radius(cfg, state.p_eq)
    if f_trial <= cfg.return_tol
        return tensor_to_vec(sigma_trial), MaterialState(copy(state.eps_p), state.p_eq, copy(state.x1), copy(state.x2))
    end

    function residual(dpeq::Float64)
        p_new = state.p_eq + dpeq
        radius = yield_radius(cfg, p_new)
        w1 = 1.0 / (1.0 + cfg.kin_gamma1 * dpeq)
        w2 = 1.0 / (1.0 + cfg.kin_gamma2 * dpeq)
        z = s_trial .- w1 .* state.x1 .- w2 .* state.x2
        alpha = 1.0 + (3.0 * mu + w1 * cfg.kin_c1 + w2 * cfg.kin_c2) * dpeq / radius
        return radius * alpha - j2_norm(z)
    end

    dpeq = max(f_trial / (3.0 * mu + cfg.kin_c1 + cfg.kin_c2 + abs(cfg.iso_q1 * cfg.iso_b1)), 0.0)
    converged = false
    for _ in 1:cfg.return_max_iter
        f_val = residual(dpeq)
        abs(f_val) < cfg.return_tol && (converged = true; break)
        h = max(1.0e-8, 1.0e-6 * max(1.0, abs(dpeq)))
        df = (residual(dpeq + h) - f_val) / h
        abs(df) < 1.0e-14 && error("Return-mapping derivative vanished")
        dpeq = max(dpeq - f_val / df, 0.0)
    end
    converged || error("Mixed-hardening return mapping failed")

    p_new = state.p_eq + dpeq
    radius = yield_radius(cfg, p_new)
    w1 = 1.0 / (1.0 + cfg.kin_gamma1 * dpeq)
    w2 = 1.0 / (1.0 + cfg.kin_gamma2 * dpeq)
    z = s_trial .- w1 .* state.x1 .- w2 .* state.x2
    z_norm = j2_norm(z)
    z_norm > 0.0 || error("Returned Z vanished unexpectedly")
    nbar = z ./ z_norm
    alpha = 1.0 + (3.0 * mu + w1 * cfg.kin_c1 + w2 * cfg.kin_c2) * dpeq / radius
    x1 = w1 .* (state.x1 .+ cfg.kin_c1 * dpeq .* nbar)
    x2 = w2 .* (state.x2 .+ cfg.kin_c2 * dpeq .* nbar)
    s_new = x1 .+ x2 .+ z ./ alpha
    hydro = tr(sigma_trial) / 3.0
    sigma_new = s_new .+ hydro .* Matrix{Float64}(I, 3, 3)
    eps_p_new = state.eps_p .+ 1.5 * dpeq .* nbar
    return tensor_to_vec(sigma_new), MaterialState(eps_p_new, p_new, x1, x2)
end

function constitutive_update(total_strain::Vector{Float64}, state::MaterialState, cfg::Config, cmat::Matrix{Float64}, mu::Float64, hmod::Float64)
    if cfg.hardening_mode == "kinematic"
        return mixed_hardening_update(total_strain, state, cfg)
    end
    return isotropic_update(total_strain, state, cfg, cmat, mu, hmod)
end

function numerical_tangent(cfg::Config, total_strain::Vector{Float64}, state::MaterialState, cmat::Matrix{Float64}, mu::Float64, hmod::Float64)
    sigma_base, _ = constitutive_update(total_strain, state, cfg, cmat, mu, hmod)
    c_alg = zeros(Float64, 4, 4)
    for j in 1:4
        strain_pert = copy(total_strain)
        strain_pert[j] += cfg.tangent_eps
        sigma_pert, _ = constitutive_update(strain_pert, state, cfg, cmat, mu, hmod)
        c_alg[:, j] .= (sigma_pert .- sigma_base) ./ cfg.tangent_eps
    end
    return 0.5 .* (c_alg .+ transpose(c_alg))
end

function assemble_system(mesh::MeshData, b_mats::Array{Float64, 3}, areas::Vector{Float64}, edof::Matrix{Int}, cfg::Config, u::Vector{Float64}, state_ref::Vector{MaterialState}, build_stiffness::Bool)
    ne = size(mesh.triangles, 1)
    ndof = 2 * size(mesh.nodes, 1)
    cmat = elastic_matrix(cfg)
    mu = shear_modulus(cfg)
    hmod = hardening_modulus(cfg)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    fint = zeros(Float64, ndof)
    sigma_trial = zeros(Float64, ne, 4)
    strain_trial = zeros(Float64, ne, 4)
    state_trial = [MaterialState() for _ in 1:ne]

    for e in 1:ne
        ids = edof[e, :]
        ue = u[ids]
        b = Matrix(view(b_mats, e, :, :))
        total_strain = vec(b * ue)
        sigma_g, state_new = constitutive_update(total_strain, state_ref[e], cfg, cmat, mu, hmod)
        sigma_trial[e, :] .= sigma_g
        strain_trial[e, :] .= total_strain
        state_trial[e] = state_new
        fe = cfg.thickness * areas[e] .* (transpose(b) * sigma_g)
        fint[ids] .+= fe
        if build_stiffness
            c_gp = numerical_tangent(cfg, total_strain, state_ref[e], cmat, mu, hmod)
            ke = cfg.thickness * areas[e] .* (transpose(b) * c_gp * b)
            for a in 1:6, bidx in 1:6
                push!(rows, ids[a])
                push!(cols, ids[bidx])
                push!(vals, ke[a, bidx])
            end
        end
    end
    stiff = build_stiffness ? sparse(rows, cols, vals, ndof, ndof) : spzeros(ndof, ndof)
    return stiff, fint, sigma_trial, strain_trial, state_trial
end

function solve_fem_reference(cfg::Config, mesh::MeshData)
    @printf("Running Julia von Mises FEM reference solver for perforated plate...\n")
    b_mats, areas = build_tri_operators(mesh)
    edof = build_element_dofs(mesh)
    ndof = 2 * size(mesh.nodes, 1)
    ne = size(mesh.triangles, 1)
    u = zeros(Float64, ndof)
    sigma_gp = zeros(Float64, ne, 4)
    strain_gp = zeros(Float64, ne, 4)
    state_gp = [MaterialState() for _ in 1:ne]
    history = Matrix{Float64}(undef, 0, 3)
    load_curve = Matrix{Float64}(undef, 0, 3)
    top_y_dofs = 2 .* mesh.top_nodes

    for step in 1:cfg.load_steps
        target_uy = cfg.top_displacement * step / cfg.load_steps
        @printf("  Load step %02d/%02d | top_uy %.6f\n", step, cfg.load_steps, target_uy)
        bc = build_constraints(mesh, target_uy)
        fixed, free = gather_sets(ndof, bc)
        for (dof, val) in bc
            u[dof] = val
        end
        step_converged = false
        last_rel_res = Inf
        ref_norm = 1.0
        for it in 1:cfg.newton_max_iter
            stiff, fint, sigma_trial, strain_trial, state_trial = assemble_system(mesh, b_mats, areas, edof, cfg, u, state_gp, true)
            residual = -fint
            ref_norm = it == 1 ? max(norm(residual[free]), 1.0) : ref_norm
            rel_res = norm(residual[free]) / ref_norm
            last_rel_res = rel_res
            history = vcat(history, reshape([step, it, rel_res], 1, 3))
            should_print = (it == 1) || (it == cfg.newton_max_iter) || (mod(it, cfg.print_every) == 0)
            should_print && @printf("    Newton iter %02d: rel_res=%.3e\n", it, rel_res)
            if rel_res < cfg.newton_tol
                sigma_gp .= sigma_trial
                strain_gp .= strain_trial
                state_gp = [MaterialState(copy(s.eps_p), s.p_eq, copy(s.x1), copy(s.x2)) for s in state_trial]
                step_converged = true
                break
            end
            kff = Matrix(stiff[free, free])
            du = kff \ residual[free]
            damping = 1.0
            best_rel = rel_res
            best_u = copy(u)
            best_sigma = copy(sigma_trial)
            best_strain = copy(strain_trial)
            best_state = [MaterialState(copy(s.eps_p), s.p_eq, copy(s.x1), copy(s.x2)) for s in state_trial]
            for _ in 1:8
                u_trial = copy(u)
                u_trial[free] .+= damping .* du
                _, fint_trial, sigma_ls, strain_ls, state_ls = assemble_system(mesh, b_mats, areas, edof, cfg, u_trial, state_gp, false)
                rel_trial = norm((-fint_trial)[free]) / ref_norm
                if rel_trial < best_rel
                    best_rel = rel_trial
                    best_u = u_trial
                    best_sigma = sigma_ls
                    best_strain = strain_ls
                    best_state = [MaterialState(copy(s.eps_p), s.p_eq, copy(s.x1), copy(s.x2)) for s in state_ls]
                end
                rel_trial < rel_res && break
                damping *= 0.5
            end
            u .= best_u
            sigma_trial = best_sigma
            strain_trial = best_strain
            state_trial = best_state
            if it == cfg.newton_max_iter
                sigma_gp .= sigma_trial
                strain_gp .= strain_trial
                state_gp = [MaterialState(copy(s.eps_p), s.p_eq, copy(s.x1), copy(s.x2)) for s in state_trial]
            end
        end
        step_converged || error(@sprintf("Load step %d failed to converge: last_rel_res=%.3e, tol=%.3e, newton_max_iter=%d", step, last_rel_res, cfg.newton_tol, cfg.newton_max_iter))
        _, fint_final, sigma_final, strain_final, state_final = assemble_system(mesh, b_mats, areas, edof, cfg, u, state_gp, false)
        sigma_gp .= sigma_final
        strain_gp .= strain_final
        state_gp = [MaterialState(copy(s.eps_p), s.p_eq, copy(s.x1), copy(s.x2)) for s in state_final]
        reaction_y = sum(fint_final[top_y_dofs])
        load_curve = vcat(load_curve, reshape([step, target_uy, reaction_y], 1, 3))
    end

    nn = size(mesh.nodes, 1)
    sigma_node = zeros(Float64, nn, 4)
    eps_p_node = zeros(Float64, nn, 4)
    alpha_node = zeros(Float64, nn)
    counts = zeros(Float64, nn)
    for e in 1:ne
        sig_e = sigma_gp[e, :]
        epsp_e = tensor_to_vec(state_gp[e].eps_p)
        alpha_e = state_gp[e].p_eq
        for nid in mesh.triangles[e, :]
            sigma_node[nid, :] .+= sig_e
            eps_p_node[nid, :] .+= epsp_e
            alpha_node[nid] += alpha_e
            counts[nid] += 1.0
        end
    end
    for nid in 1:nn
        if counts[nid] > 0.0
            sigma_node[nid, :] ./= counts[nid]
            eps_p_node[nid, :] ./= counts[nid]
            alpha_node[nid] /= counts[nid]
        end
    end
    sigma_vm = [von_mises(view(sigma_node, i, :)) for i in 1:nn]

    return Dict(
        "u" => u,
        "history" => history,
        "load_curve" => load_curve,
        "ux" => u[1:2:end],
        "uy" => u[2:2:end],
        "sigma_xx" => sigma_node[:, 1],
        "sigma_yy" => sigma_node[:, 2],
        "sigma_zz" => sigma_node[:, 3],
        "sigma_xy" => sigma_node[:, 4],
        "sigma_vm" => sigma_vm,
        "eps_p_xx" => eps_p_node[:, 1],
        "eps_p_yy" => eps_p_node[:, 2],
        "eps_p_zz" => eps_p_node[:, 3],
        "eps_p_xy" => eps_p_node[:, 4],
        "eps_p_eq" => alpha_node,
    )
end

function save_fields_csv(path::String, mesh::MeshData, fields::AbstractDict{String, <:Any})
    open(path, "w") do io
        println(io, "node_id,x,y,ux,uy,sigma_xx,sigma_yy,sigma_zz,sigma_xy,sigma_vm,eps_p_xx,eps_p_yy,eps_p_zz,eps_p_xy,eps_p_eq")
        for nid in 1:size(mesh.nodes, 1)
            @printf(
                io,
                "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                nid,
                mesh.nodes[nid, 1],
                mesh.nodes[nid, 2],
                fields["ux"][nid],
                fields["uy"][nid],
                fields["sigma_xx"][nid],
                fields["sigma_yy"][nid],
                fields["sigma_zz"][nid],
                fields["sigma_xy"][nid],
                fields["sigma_vm"][nid],
                fields["eps_p_xx"][nid],
                fields["eps_p_yy"][nid],
                fields["eps_p_zz"][nid],
                fields["eps_p_xy"][nid],
                fields["eps_p_eq"][nid],
            )
        end
    end
end

function save_elements_csv(path::String, mesh::MeshData)
    open(path, "w") do io
        println(io, "element_id,n1,n2,n3")
        for eid in 1:size(mesh.triangles, 1)
            tri = mesh.triangles[eid, :]
            @printf(io, "%d,%d,%d,%d\n", eid, tri[1], tri[2], tri[3])
        end
    end
end

function save_boundary_csv(path::String, ids::Vector{Int})
    open(path, "w") do io
        println(io, "node_id")
        for nid in ids
            println(io, nid)
        end
    end
end

function save_top_edges_csv(path::String, mesh::MeshData)
    open(path, "w") do io
        println(io, "edge_id,n1,n2")
        for eid in 1:size(mesh.top_edges, 1)
            n1, n2 = mesh.top_edges[eid, 1], mesh.top_edges[eid, 2]
            @printf(io, "%d,%d,%d\n", eid, n1, n2)
        end
    end
end

function save_history_csv(path::String, history::Matrix{Float64})
    open(path, "w") do io
        println(io, "load_step,newton_iter,rel_residual")
        for i in 1:size(history, 1)
            @printf(io, "%.0f,%.0f,%.12e\n", history[i, 1], history[i, 2], history[i, 3])
        end
    end
end

function save_load_curve_csv(path::String, load_curve::Matrix{Float64})
    open(path, "w") do io
        println(io, "load_step,top_uy,reaction_y")
        for i in 1:size(load_curve, 1)
            @printf(io, "%.0f,%.12e,%.12e\n", load_curve[i, 1], load_curve[i, 2], load_curve[i, 3])
        end
    end
end

function write_run_config(path::String, cfg::Config)
    open(path, "w") do io
        for name in fieldnames(Config)
            println(io, string(name), "=", getfield(cfg, name))
        end
    end
end

function interpolate_field_to_grid(cfg::Config, mesh::MeshData, field::AbstractVector{<:Real}; nx::Int = 220, ny::Int = 220, nneigh::Int = 6)
    xs = collect(range(0.0, cfg.width, length = nx))
    ys = collect(range(0.0, cfg.height, length = ny))
    z = fill(NaN, ny, nx)
    k = min(nneigh, size(mesh.nodes, 1))
    for (iy, y) in enumerate(ys)
        for (ix, x) in enumerate(xs)
            if hypot(x, y) < cfg.radius - 1.0e-10
                continue
            end
            dist2 = (mesh.nodes[:, 1] .- x) .^ 2 .+ (mesh.nodes[:, 2] .- y) .^ 2
            idx = partialsortperm(dist2, 1:k)
            dsel = dist2[idx]
            if dsel[1] < 1.0e-20
                z[iy, ix] = float(field[idx[1]])
            else
                w = 1.0 ./ dsel
                z[iy, ix] = sum(w .* field[idx]) / sum(w)
            end
        end
    end
    return xs, ys, z
end

function ensure_plots_loaded()
    try
        Base.invokelatest(Core.eval, Main, :(using Plots))
        return Base.invokelatest(getfield, Main, :Plots)
    catch
        error("Plots not found. Run with --save-plots 0 to skip plotting, or install it by: julia -e 'import Pkg; Pkg.add(\"Plots\")'")
    end
end

function save_plots_if_needed(cfg::Config, mesh::MeshData, fem::AbstractDict{String, <:Any}, output_dir::String)
    cfg.save_plots == 0 && return
    plotmod = ensure_plots_loaded()
    Base.invokelatest(plotmod.default; dpi = cfg.plot_dpi)
    labels = [
        ("ux", "U_x"),
        ("uy", "U_y"),
        ("sigma_xx", "Sigma_xx"),
        ("sigma_yy", "Sigma_yy"),
        ("sigma_vm", "Von Mises"),
        ("eps_p_eq", "Eq. plastic strain"),
    ]
    panel_plots = Any[]
    for (idx, (key, title)) in enumerate(labels)
        xs, ys, z = interpolate_field_to_grid(cfg, mesh, fem[key])
        field_fig = Base.invokelatest(
            plotmod.contourf,
            xs,
            ys,
            z,
            title = title,
            xlabel = "X Coordinate",
            ylabel = "Y Coordinate",
            aspect_ratio = :equal,
            colorbar = true,
            levels = 24,
            c = :jet,
        )
        push!(panel_plots, field_fig)
    end
    fig = Base.invokelatest(plotmod.plot, panel_plots...; layout = (2, 3), size = (1380, 820))
    Base.invokelatest(plotmod.savefig, fig, joinpath(output_dir, "perforated_plate_square_fem_fields.png"))

    xs_uy, ys_uy, z_uy = interpolate_field_to_grid(cfg, mesh, fem["uy"])
    uy_fig = Base.invokelatest(plotmod.contourf, xs_uy, ys_uy, z_uy; title = "U_y", xlabel = "X Coordinate", ylabel = "Y Coordinate", aspect_ratio = :equal, colorbar = true, size = (620, 480), levels = 24, c = :jet)
    Base.invokelatest(plotmod.savefig, uy_fig, joinpath(output_dir, "perforated_plate_square_fem_uy.png"))

    hist = fem["history"]
    hist_fig = Base.invokelatest(plotmod.plot, hist[:, 2], hist[:, 3]; marker = :circle, yscale = :log10, xlabel = "Newton Iteration", ylabel = "Relative Residual", title = "Newton Residual History", legend = false, size = (620, 420))
    Base.invokelatest(plotmod.savefig, hist_fig, joinpath(output_dir, "perforated_plate_square_fem_history.png"))

    curve = fem["load_curve"]
    load_fig = Base.invokelatest(plotmod.plot, curve[:, 2], curve[:, 3]; marker = :circle, xlabel = "Top displacement", ylabel = "Reaction Y", title = "Load-displacement curve", legend = false, size = (620, 420))
    Base.invokelatest(plotmod.savefig, load_fig, joinpath(output_dir, "perforated_plate_square_fem_load.png"))
end

function main()
    cfg = parse_cli!(Config())
    mkpath(cfg.output_dir)
    @printf("Julia perforated-plate von Mises FEM benchmark | ntheta=%d nradial=%d load_steps=%d\n", cfg.ntheta, cfg.nradial, cfg.load_steps)
    mesh = build_polar_mesh(cfg)
    fem = solve_fem_reference(cfg, mesh)
    save_fields_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_fields.csv"), mesh, fem)
    save_elements_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_elements.csv"), mesh)
    save_boundary_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_boundary_left.csv"), mesh.left_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_boundary_bottom.csv"), mesh.bottom_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_boundary_top.csv"), mesh.top_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_boundary_right.csv"), mesh.right_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_boundary_hole.csv"), mesh.hole_nodes)
    save_top_edges_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_top_edges.csv"), mesh)
    save_history_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_history.csv"), fem["history"])
    save_load_curve_csv(joinpath(cfg.output_dir, "perforated_plate_square_fem_load_curve.csv"), fem["load_curve"])
    write_run_config(joinpath(cfg.output_dir, "perforated_plate_square_fem_run_config.txt"), cfg)
    save_plots_if_needed(cfg, mesh, fem, cfg.output_dir)
    @printf("Saved outputs to: %s\n", cfg.output_dir)
end

main()
