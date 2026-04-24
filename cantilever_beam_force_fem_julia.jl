using LinearAlgebra
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
using Printf
using SparseArrays
using Statistics

Base.@kwdef mutable struct Config
    length::Float64 = 1.0
    height::Float64 = 0.2
    young::Float64 = 2.0e5
    poisson::Float64 = 0.3
    yield_stress::Float64 = 200.0
    hardening::Float64 = 5.0e4
    load_start::Float64 = 3.0
    load_end::Float64 = 5.0
    load_steps::Int = 20
    load_case::String = "right_mid_point"
    fem_nx::Int = 63
    fem_ny::Int = 13
    newton_tol::Float64 = 1.0e-6
    newton_max_iter::Int = 400
    print_every::Int = 10
    save_plots::Int = 1
    plot_dpi::Int = 120
    output_dir::String = ""
end

struct MeshData
    nodes::Matrix{Float64}
    elements::Matrix{Int}
    triangles::Matrix{Int}
    left_nodes::Vector{Int}
    right_nodes::Vector{Int}
    top_nodes::Vector{Int}
    bottom_nodes::Vector{Int}
    top_edges::Matrix{Int}
    tip_node::Int
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
        cfg.output_dir = joinpath(@__DIR__, "outputs", "cantilever_beam_force_fem")
    end
    return cfg
end

function lame_parameters(cfg::Config)
    mu = cfg.young / (2.0 * (1.0 + cfg.poisson))
    lam = cfg.young * cfg.poisson / ((1.0 + cfg.poisson) * (1.0 - 2.0 * cfg.poisson))
    return lam, mu
end

function elastic_matrix(cfg::Config)
    lam, mu = lame_parameters(cfg)
    return [
        lam + 2.0 * mu lam lam 0.0
        lam lam + 2.0 * mu lam 0.0
        lam lam lam + 2.0 * mu 0.0
        0.0 0.0 0.0 2.0 * mu
    ]
end

function deviatoric(sig::AbstractVector)
    mean_sig = (sig[1] + sig[2] + sig[3]) / 3.0
    return [sig[1] - mean_sig, sig[2] - mean_sig, sig[3] - mean_sig, sig[4]]
end

function von_mises(sig::AbstractVector)
    s = deviatoric(sig)
    return sqrt(max(1.5 * (s[1]^2 + s[2]^2 + s[3]^2 + 2.0 * s[4]^2), 0.0))
end

function build_structured_mesh(cfg::Config)
    x = collect(range(0.0, cfg.length, length = cfg.fem_nx + 1))
    y = collect(range(0.0, cfg.height, length = cfg.fem_ny + 1))
    nx1 = cfg.fem_nx + 1
    nodes = zeros(Float64, (cfg.fem_nx + 1) * (cfg.fem_ny + 1), 2)
    for j in 0:cfg.fem_ny, i in 0:cfg.fem_nx
        nid = j * nx1 + i + 1
        nodes[nid, 1] = x[i + 1]
        nodes[nid, 2] = y[j + 1]
    end
    elements = zeros(Int, cfg.fem_nx * cfg.fem_ny, 4)
    triangles = zeros(Int, 2 * cfg.fem_nx * cfg.fem_ny, 3)
    eid = 1
    tid = 1
    for j in 0:(cfg.fem_ny - 1), i in 0:(cfg.fem_nx - 1)
        n0 = j * nx1 + i + 1
        n1 = n0 + 1
        n2 = n0 + nx1
        n3 = n2 + 1
        elements[eid, :] .= (n0, n1, n3, n2)
        triangles[tid, :] .= (n0, n1, n3)
        triangles[tid + 1, :] .= (n0, n3, n2)
        eid += 1
        tid += 2
    end
    left_nodes = collect(1:nx1:(cfg.fem_ny * nx1 + 1))
    right_nodes = collect(nx1:nx1:((cfg.fem_ny + 1) * nx1))
    bottom_nodes = collect(1:nx1)
    top_nodes = collect(cfg.fem_ny * nx1 + 1:(cfg.fem_ny + 1) * nx1)
    top_edges = zeros(Int, cfg.fem_nx, 2)
    for i in 1:cfg.fem_nx
        top_edges[i, :] .= (top_nodes[i], top_nodes[i + 1])
    end
    tip_local = argmin(abs.(nodes[right_nodes, 2] .- 0.5 * cfg.height))
    tip_node = right_nodes[tip_local]
    return MeshData(nodes, elements, triangles, left_nodes, right_nodes, top_nodes, bottom_nodes, top_edges, tip_node)
end

function q4_reference_matrices(cfg::Config)
    dx = cfg.length / cfg.fem_nx
    dy = cfg.height / cfg.fem_ny
    g = 1.0 / sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    b_all = zeros(Float64, 4, 4, 8)
    for (idx, (xi, eta)) in enumerate(gauss)
        dndxi = 0.25 .* [-(1.0 - eta), +(1.0 - eta), +(1.0 + eta), -(1.0 + eta)]
        dndeta = 0.25 .* [-(1.0 - xi), -(1.0 + xi), +(1.0 + xi), +(1.0 - xi)]
        dndx = 2.0 .* dndxi ./ dx
        dndy = 2.0 .* dndeta ./ dy
        b = zeros(Float64, 4, 8)
        for a in 1:4
            col = 2 * a - 1
            b[1, col] = dndx[a]
            b[2, col + 1] = dndy[a]
            b[4, col] = 0.5 * dndy[a]
            b[4, col + 1] = 0.5 * dndx[a]
        end
        b_all[idx, :, :] .= b
    end
    det_j = dx * dy / 4.0
    return b_all, det_j
end

function build_free_dofs(mesh::MeshData)
    ndof = 2 * size(mesh.nodes, 1)
    constrained = falses(ndof)
    for nid in mesh.left_nodes
        constrained[2 * nid - 1] = true
        constrained[2 * nid] = true
    end
    free = findall(.!constrained)
    return free
end

function build_top_load_vector(mesh::MeshData)
    ndof = 2 * size(mesh.nodes, 1)
    force = zeros(Float64, ndof)
    force[2 * mesh.tip_node] = -1.0
    return force
end

function radial_return(total_strain::Vector{Float64}, eps_p_prev::Vector{Float64}, alpha_prev::Float64, cfg::Config, cmat::Matrix{Float64}, mu::Float64)
    sigma_trial = cmat * (total_strain - eps_p_prev)
    s_trial = deviatoric(sigma_trial)
    seq_trial = von_mises(sigma_trial)
    fy = seq_trial - (cfg.yield_stress + cfg.hardening * alpha_prev)
    if fy <= 0.0 || seq_trial <= 1.0e-12
        return sigma_trial, copy(eps_p_prev), alpha_prev, total_strain - eps_p_prev
    end
    dgamma = fy / (3.0 * mu + cfg.hardening)
    flow = (1.5 / seq_trial) .* s_trial
    eps_p_new = eps_p_prev .+ dgamma .* flow
    alpha_new = alpha_prev + dgamma
    mean_trial = (sigma_trial[1] + sigma_trial[2] + sigma_trial[3]) / 3.0
    factor = 1.0 - 3.0 * mu * dgamma / seq_trial
    s_new = factor .* s_trial
    sigma_new = [s_new[1] + mean_trial, s_new[2] + mean_trial, s_new[3] + mean_trial, s_new[4]]
    return sigma_new, eps_p_new, alpha_new, total_strain - eps_p_new
end

function compute_fem_internal_force(mesh::MeshData, b_mats::Array{Float64, 3}, det_j::Float64, u_full::Vector{Float64}, eps_p_prev::Array{Float64, 3}, alpha_prev::Matrix{Float64}, cfg::Config)
    cmat = elastic_matrix(cfg)
    _, mu = lame_parameters(cfg)
    ndof = length(u_full)
    ne = size(mesh.elements, 1)
    fint = zeros(Float64, ndof)
    sigma_gp = zeros(Float64, ne, 4, 4)
    eps_p_new = zeros(Float64, ne, 4, 4)
    alpha_new = zeros(Float64, ne, 4)
    eps_e_new = zeros(Float64, ne, 4, 4)
    for e in 1:ne
        elem = mesh.elements[e, :]
        ids = Int[]
        for nid in elem
            push!(ids, 2 * nid - 1)
            push!(ids, 2 * nid)
        end
        ue = u_full[ids]
        fe = zeros(Float64, 8)
        for g in 1:4
            b = Matrix(view(b_mats, g, :, :))
            strain = vec(b * ue)
            sigma_g, epsp_g, alpha_g, epse_g = radial_return(strain, vec(eps_p_prev[e, g, :]), alpha_prev[e, g], cfg, cmat, mu)
            sigma_gp[e, g, :] .= sigma_g
            eps_p_new[e, g, :] .= epsp_g
            alpha_new[e, g] = alpha_g
            eps_e_new[e, g, :] .= epse_g
            fe .+= (transpose(b) * sigma_g) .* det_j
        end
        fint[ids] .+= fe
    end
    return fint, sigma_gp, eps_p_new, alpha_new, eps_e_new
end

function assemble_elastic_stiffness(mesh::MeshData, b_mats::Array{Float64, 3}, det_j::Float64, cfg::Config)
    dmat = elastic_matrix(cfg)
    ke = zeros(Float64, 8, 8)
    for g in 1:4
        b = Matrix(view(b_mats, g, :, :))
        ke .+= transpose(b) * dmat * b .* det_j
    end
    ndof = 2 * size(mesh.nodes, 1)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for elem in eachrow(mesh.elements)
        ids = Int[]
        for nid in elem
            push!(ids, 2 * nid - 1)
            push!(ids, 2 * nid)
        end
        for a in 1:8, b in 1:8
            push!(rows, ids[a])
            push!(cols, ids[b])
            push!(vals, ke[a, b])
        end
    end
    return sparse(rows, cols, vals, ndof, ndof)
end

function average_gauss_to_nodes(mesh::MeshData, sigma_gp::Array{Float64, 3}, alpha_gp::Matrix{Float64}, eps_p_gp::Array{Float64, 3})
    nn = size(mesh.nodes, 1)
    sigma_node = zeros(Float64, nn, 4)
    eps_p_node = zeros(Float64, nn, 4)
    alpha_node = zeros(Float64, nn)
    counts = zeros(Float64, nn)
    for e in 1:size(mesh.elements, 1)
        sig_e = vec(mean(sigma_gp[e, :, :], dims = 1))
        epsp_e = vec(mean(eps_p_gp[e, :, :], dims = 1))
        alpha_e = mean(alpha_gp[e, :])
        for nid in mesh.elements[e, :]
            sigma_node[nid, :] .+= sig_e
            eps_p_node[nid, :] .+= epsp_e
            alpha_node[nid] += alpha_e
            counts[nid] += 1.0
        end
    end
    for nid in 1:nn
        sigma_node[nid, :] ./= counts[nid]
        eps_p_node[nid, :] ./= counts[nid]
        alpha_node[nid] /= counts[nid]
    end
    sigma_vm = [von_mises(view(sigma_node, i, :)) for i in 1:nn]
    return sigma_node, eps_p_node, alpha_node, sigma_vm
end

function solve_fem_reference(cfg::Config, mesh::MeshData)
    @printf("Running cantilever-beam weak-form Julia FEM reference...\n")
    b_mats, det_j = q4_reference_matrices(cfg)
    free = build_free_dofs(mesh)
    k_elastic = assemble_elastic_stiffness(mesh, b_mats, det_j, cfg)
    kff = Matrix(k_elastic[free, free])
    ndof = 2 * size(mesh.nodes, 1)
    ne = size(mesh.elements, 1)
    u_full = zeros(Float64, ndof)
    sigma_gp = zeros(Float64, ne, 4, 4)
    eps_p_gp = zeros(Float64, ne, 4, 4)
    alpha_gp = zeros(Float64, ne, 4)
    eps_e_gp = zeros(Float64, ne, 4, 4)
    history = Matrix{Float64}(undef, 0, 3)
    f_unit = build_top_load_vector(mesh)
    loads = cfg.load_steps == 1 ? [cfg.load_end] : collect(range(cfg.load_start, cfg.load_end, length = cfg.load_steps))
    for (step_id, q) in enumerate(loads)
        f_ext = q .* f_unit
        ref_norm = max(norm(f_ext[free]), 1.0)
        @printf("  FEM load step %02d/%02d: q=%.6f\n", step_id, cfg.load_steps, q)
        for it in 1:cfg.newton_max_iter
            fint, sigma_trial, epsp_trial, alpha_trial, epse_trial = compute_fem_internal_force(mesh, b_mats, det_j, u_full, eps_p_gp, alpha_gp, cfg)
            residual = f_ext - fint
            rel_res = norm(residual[free]) / ref_norm
            history = vcat(history, reshape([step_id, it, rel_res], 1, 3))
            if it == 1 || it == cfg.newton_max_iter || mod(it, cfg.print_every) == 0
                @printf("    Newton iter %02d: rel_res=%.3e\n", it, rel_res)
            end
            if rel_res < cfg.newton_tol
                sigma_gp .= sigma_trial
                eps_p_gp .= epsp_trial
                alpha_gp .= alpha_trial
                eps_e_gp .= epse_trial
                break
            end
            du = kff \ residual[free]
            u_full[free] .+= du
            if it == cfg.newton_max_iter
                sigma_gp .= sigma_trial
                eps_p_gp .= epsp_trial
                alpha_gp .= alpha_trial
                eps_e_gp .= epse_trial
                @printf("    Warning: modified Newton hit the iteration cap.\n")
            end
        end
    end
    sigma_node, eps_p_node, alpha_node, sigma_vm = average_gauss_to_nodes(mesh, sigma_gp, alpha_gp, eps_p_gp)
    return Dict(
        "u" => u_full,
        "history" => history,
        "ux" => u_full[1:2:end],
        "uy" => u_full[2:2:end],
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
        "top_load" => f_unit,
    )
end

function save_fields_csv(path::String, mesh::MeshData, fields::Dict{String, <:Any})
    open(path, "w") do io
        println(io, "node_id,x,y,ux,uy,sigma_xx,sigma_yy,sigma_zz,sigma_xy,sigma_vm,eps_p_xx,eps_p_yy,eps_p_zz,eps_p_xy,eps_p_eq")
        for nid in 1:size(mesh.nodes, 1)
            @printf(io, "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                nid, mesh.nodes[nid, 1], mesh.nodes[nid, 2], fields["ux"][nid], fields["uy"][nid],
                fields["sigma_xx"][nid], fields["sigma_yy"][nid], fields["sigma_zz"][nid], fields["sigma_xy"][nid], fields["sigma_vm"][nid],
                fields["eps_p_xx"][nid], fields["eps_p_yy"][nid], fields["eps_p_zz"][nid], fields["eps_p_xy"][nid], fields["eps_p_eq"][nid]
            )
        end
    end
end

function save_elements_csv(path::String, mesh::MeshData)
    open(path, "w") do io
        println(io, "element_id,n1,n2,n3,n4")
        for eid in 1:size(mesh.elements, 1)
            @printf(io, "%d,%d,%d,%d,%d\n", eid, mesh.elements[eid, 1], mesh.elements[eid, 2], mesh.elements[eid, 3], mesh.elements[eid, 4])
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

function save_history_csv(path::String, history::Matrix{Float64})
    open(path, "w") do io
        println(io, "load_step,newton_iter,rel_residual")
        for i in 1:size(history, 1)
            @printf(io, "%.0f,%.0f,%.12e\n", history[i, 1], history[i, 2], history[i, 3])
        end
    end
end

function save_top_load_csv(path::String, top_load::Vector{Float64})
    open(path, "w") do io
        println(io, "dof_id,load_value")
        for i in 1:length(top_load)
            @printf(io, "%d,%.12e\n", i, top_load[i])
        end
    end
end

function write_run_config(path::String, cfg::Config)
    open(path, "w") do io
        for name in fieldnames(Config)
            println(io, string(name), " = ", getfield(cfg, name))
        end
    end
end

function save_plots_if_needed(cfg::Config, mesh::MeshData, fem::Dict{String, <:Any})
    cfg.save_plots == 0 && return
    Plots.default(dpi = cfg.plot_dpi)
    nx = cfg.fem_nx + 1
    ny = cfg.fem_ny + 1
    xs = reshape(mesh.nodes[:, 1], nx, ny)'
    ys = reshape(mesh.nodes[:, 2], nx, ny)'
    fields = [
        ("ux", "U_x"),
        ("uy", "U_y"),
        ("sigma_vm", "Von Mises"),
        ("eps_p_eq", "Eq. plastic strain"),
    ]
    fig = plot(layout = (2, 2), size = (1200, 820))
    for (idx, (key, title)) in enumerate(fields)
        z = reshape(fem[key], nx, ny)'
        contourf!(fig[idx], xs[1, :], ys[:, 1], z, title = title, xlabel = "X Coordinate", ylabel = "Y Coordinate", colorbar = true, c = :jet, levels = 24)
    end
    savefig(fig, joinpath(cfg.output_dir, "cantilever_beam_force_fem_fields.png"))
    hist = fem["history"]
    hist_fig = plot(hist[:, 2], hist[:, 3], marker = :circle, yscale = :log10, xlabel = "Newton Iteration", ylabel = "Relative Residual", title = "Newton Residual History", legend = false, size = (620, 420))
    savefig(hist_fig, joinpath(cfg.output_dir, "cantilever_beam_force_fem_history.png"))
    loads = cfg.load_steps == 1 ? [cfg.load_end] : collect(range(cfg.load_start, cfg.load_end, length = cfg.load_steps))
    load_fig = plot(loads, zeros(length(loads)), marker = :circle, xlabel = "Load factor", ylabel = "Applied point-load factor", title = "Load schedule", legend = false, size = (620, 420))
    savefig(load_fig, joinpath(cfg.output_dir, "cantilever_beam_force_fem_load.png"))
end

function main()
    cfg = parse_cli!(Config())
    mkpath(cfg.output_dir)
    mesh = build_structured_mesh(cfg)
    fem = solve_fem_reference(cfg, mesh)
    save_fields_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_fields.csv"), mesh, fem)
    save_elements_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_elements.csv"), mesh)
    save_boundary_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_boundary_left.csv"), mesh.left_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_boundary_right.csv"), mesh.right_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_boundary_top.csv"), mesh.top_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_boundary_bottom.csv"), mesh.bottom_nodes)
    save_history_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_history.csv"), fem["history"])
    save_top_load_csv(joinpath(cfg.output_dir, "cantilever_beam_force_fem_top_load.csv"), fem["top_load"])
    write_run_config(joinpath(cfg.output_dir, "cantilever_beam_force_fem_run_config.txt"), cfg)
    save_plots_if_needed(cfg, mesh, fem)
    @printf("Saved outputs to: %s\n", cfg.output_dir)
end

main()
