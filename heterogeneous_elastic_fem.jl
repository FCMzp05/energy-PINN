using LinearAlgebra
using Printf
using SparseArrays
using Statistics

struct Q4Mesh
    nodes::Matrix{Float64}
    quads::Matrix{Int}
end

function plane_strain_matrix(young::Float64, poisson::Float64)
    coef = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return coef .* [
        1.0 - poisson poisson 0.0
        poisson 1.0 - poisson 0.0
        0.0 0.0 (1.0 - 2.0 * poisson) / 2.0
    ]
end

function plane_stress_matrix(young::Float64, poisson::Float64)
    coef = young / (1.0 - poisson^2)
    return coef .* [
        1.0 poisson 0.0
        poisson 1.0 0.0
        0.0 0.0 (1.0 - poisson) / 2.0
    ]
end

function build_q4_mesh(width::Float64, height::Float64, nx::Int, ny::Int)
    xs = collect(range(0.0, width, length = nx + 1))
    ys = collect(range(0.0, height, length = ny + 1))
    nodes = zeros(Float64, (nx + 1) * (ny + 1), 2)
    nid = 1
    for y in ys, x in xs
        nodes[nid, 1] = x
        nodes[nid, 2] = y
        nid += 1
    end

    quads = zeros(Int, nx * ny, 4)
    eid = 1
    for j in 0:(ny - 1), i in 0:(nx - 1)
        n0 = j * (nx + 1) + i + 1
        n1 = n0 + 1
        n3 = n0 + (nx + 1)
        n2 = n3 + 1
        quads[eid, :] .= (n0, n1, n2, n3)
        eid += 1
    end
    return Q4Mesh(nodes, quads)
end

function q4_gauss_rule()
    a = inv(sqrt(3.0))
    pts = [(-a, -a), (a, -a), (a, a), (-a, a)]
    ws = ones(Float64, 4)
    return pts, ws
end

function q4_shape_grad(xi::Float64, eta::Float64)
    return 0.25 .* [
        -(1.0 - eta) -(1.0 - xi)
        1.0 - eta -(1.0 + xi)
        1.0 + eta 1.0 + xi
        -(1.0 + eta) 1.0 - xi
    ]
end

function build_q4_operators(mesh::Q4Mesh)
    ne = size(mesh.quads, 1)
    b_mats = zeros(Float64, ne, 4, 3, 8)
    detjw = zeros(Float64, ne, 4)
    pts, ws = q4_gauss_rule()
    for e in 1:ne
        xy = mesh.nodes[mesh.quads[e, :], :]
        for (g, ((xi, eta), wg)) in enumerate(zip(pts, ws))
            dnds = q4_shape_grad(xi, eta)
            jac = transpose(dnds) * xy
            detj = det(jac)
            grads = dnds * inv(jac)
            b = zeros(Float64, 3, 8)
            for a in 1:4
                b[1, 2 * a - 1] = grads[a, 1]
                b[2, 2 * a] = grads[a, 2]
                b[3, 2 * a - 1] = grads[a, 2]
                b[3, 2 * a] = grads[a, 1]
            end
            b_mats[e, g, :, :] .= b
            detjw[e, g] = detj * wg
        end
    end
    return b_mats, detjw
end

function build_dof_map(quads::Matrix{Int})
    ne = size(quads, 1)
    dof_map = zeros(Int, ne, 8)
    for e in 1:ne
        for (a, nid) in enumerate(quads[e, :])
            dof_map[e, 2 * a - 1] = 2 * nid - 1
            dof_map[e, 2 * a] = 2 * nid
        end
    end
    return dof_map
end

function assemble_stiffness(nnodes::Int, dof_map::Matrix{Int}, b_mats::Array{Float64, 4}, detjw::Matrix{Float64}, dmat_e::Array{Float64, 3}, thickness::Float64)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    ne = size(dof_map, 1)
    for e in 1:ne
        ke = zeros(Float64, 8, 8)
        for g in 1:4
            b = Matrix(view(b_mats, e, g, :, :))
            dmat = Matrix(view(dmat_e, e, :, :))
            ke .+= thickness .* (transpose(b) * dmat * b) .* detjw[e, g]
        end
        ids = dof_map[e, :]
        for i in 1:8, j in 1:8
            push!(rows, ids[i])
            push!(cols, ids[j])
            push!(vals, ke[i, j])
        end
    end
    return sparse(rows, cols, vals, 2 * nnodes, 2 * nnodes)
end

function solve_linear_system(stiff::SparseMatrixCSC{Float64, Int}, fext::Vector{Float64}, bc::Dict{Int, Float64})
    ndof = size(stiff, 1)
    fixed = sort!(collect(keys(bc)))
    free = setdiff(collect(1:ndof), fixed)
    u = zeros(Float64, ndof)
    for (dof, val) in bc
        u[dof] = val
    end
    rhs = fext[free] - Matrix(stiff[free, fixed]) * u[fixed]
    u[free] = Matrix(stiff[free, free]) \ rhs
    return u
end

function nodal_average(nnodes::Int, quads::Matrix{Int}, elem_values_gp::Array{Float64, 3})
    elem_values = dropdims(mean(elem_values_gp; dims = 2), dims = 2)
    nodal = zeros(Float64, nnodes, size(elem_values, 2))
    counts = zeros(Float64, nnodes)
    for e in 1:size(quads, 1)
        for nid in quads[e, :]
            nodal[nid, :] .+= elem_values[e, :]
            counts[nid] += 1.0
        end
    end
    for nid in 1:nnodes
        if counts[nid] > 0.0
            nodal[nid, :] ./= counts[nid]
        end
    end
    return nodal
end

function postprocess_q4(mesh::Q4Mesh, dof_map::Matrix{Int}, b_mats::Array{Float64, 4}, dmat_e::Array{Float64, 3}, u::Vector{Float64})
    ne = size(mesh.quads, 1)
    ue = zeros(Float64, ne, 8)
    for e in 1:ne
        ue[e, :] .= u[dof_map[e, :]]
    end
    strain_gp = zeros(Float64, ne, 4, 3)
    stress_gp = zeros(Float64, ne, 4, 3)
    for e in 1:ne, g in 1:4
        b = Matrix(view(b_mats, e, g, :, :))
        strain_gp[e, g, :] .= b * view(ue, e, :)
        stress_gp[e, g, :] .= Matrix(view(dmat_e, e, :, :)) * view(strain_gp, e, g, :)
    end
    stress_node = nodal_average(size(mesh.nodes, 1), mesh.quads, stress_gp)
    return Dict(
        "ux" => u[1:2:end],
        "uy" => u[2:2:end],
        "sx" => stress_node[:, 1],
        "sy" => stress_node[:, 2],
        "sxy" => stress_node[:, 3],
    )
end

function save_fields_csv(path::String, nodes::Matrix{Float64}, fields::AbstractDict{String, <:Any})
    open(path, "w") do io
        println(io, "node_id,x,y,ux,uy,sx,sy,sxy")
        for nid in 1:size(nodes, 1)
            @printf(
                io,
                "%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                nid,
                nodes[nid, 1],
                nodes[nid, 2],
                fields["ux"][nid],
                fields["uy"][nid],
                fields["sx"][nid],
                fields["sy"][nid],
                fields["sxy"][nid],
            )
        end
    end
end

function save_quads_csv(path::String, quads::Matrix{Int}; material_ids::Union{Nothing, Vector{Int}} = nothing)
    open(path, "w") do io
        if material_ids === nothing
            println(io, "element_id,n1,n2,n3,n4")
            for eid in 1:size(quads, 1)
                q = quads[eid, :]
                @printf(io, "%d,%d,%d,%d,%d\n", eid, q[1], q[2], q[3], q[4])
            end
        else
            println(io, "element_id,n1,n2,n3,n4,material_id")
            for eid in 1:size(quads, 1)
                q = quads[eid, :]
                @printf(io, "%d,%d,%d,%d,%d,%d\n", eid, q[1], q[2], q[3], q[4], material_ids[eid])
            end
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

function save_edges_csv(path::String, edges::Matrix{Int})
    open(path, "w") do io
        println(io, "edge_id,n1,n2")
        for eid in 1:size(edges, 1)
            @printf(io, "%d,%d,%d\n", eid, edges[eid, 1], edges[eid, 2])
        end
    end
end

function write_run_config(path::String, pairs::AbstractVector{<:Pair{String, <:Any}})
    open(path, "w") do io
        for (k, v) in pairs
            println(io, k, "=", v)
        end
    end
end

Base.@kwdef mutable struct Config
    width::Float64 = 5.0
    height::Float64 = 5.0
    thickness::Float64 = 1.0
    inclusion_center_x::Float64 = 2.5
    inclusion_center_y::Float64 = 2.5
    inclusion_radius::Float64 = 1.0
    young_matrix::Float64 = 1.0
    young_inclusion::Float64 = 10.0
    poisson::Float64 = 0.30
    traction_x::Float64 = 1.0
    nx::Int = 22
    ny::Int = 22
    output_dir::String = ""
end

function parse_cli!(cfg::Config)
    i = 1
    while i <= length(ARGS)
        key = ARGS[i]
        i == length(ARGS) && error("Missing value for $key")
        val = ARGS[i + 1]
        name = Symbol(replace(key[3:end], "-" => "_"))
        old = getproperty(cfg, name)
        if old isa Int
            setproperty!(cfg, name, parse(Int, val))
        elseif old isa Float64
            setproperty!(cfg, name, parse(Float64, val))
        else
            setproperty!(cfg, name, val)
        end
        i += 2
    end
    if isempty(cfg.output_dir)
        cfg.output_dir = joinpath(@__DIR__, "outputs", "heterogeneous_elastic_fem")
    end
    return cfg
end

function main()
    cfg = parse_cli!(Config())
    mkpath(cfg.output_dir)
    mesh = build_q4_mesh(cfg.width, cfg.height, cfg.nx, cfg.ny)
    b_mats, detjw = build_q4_operators(mesh)
    dof_map = build_dof_map(mesh.quads)

    material_ids = zeros(Int, size(mesh.quads, 1))
    dmat_e = zeros(Float64, size(mesh.quads, 1), 3, 3)
    dmat_matrix = plane_strain_matrix(cfg.young_matrix, cfg.poisson)
    dmat_inclusion = plane_strain_matrix(cfg.young_inclusion, cfg.poisson)
    right_edges = Int[]
    top_rows = Vector{NTuple{2, Int}}()

    for e in 1:size(mesh.quads, 1)
        q = mesh.quads[e, :]
        cx = mean(mesh.nodes[q, 1])
        cy = mean(mesh.nodes[q, 2])
        r2 = (cx - cfg.inclusion_center_x)^2 + (cy - cfg.inclusion_center_y)^2
        if r2 <= cfg.inclusion_radius^2
            material_ids[e] = 1
            dmat_e[e, :, :] .= dmat_inclusion
        else
            dmat_e[e, :, :] .= dmat_matrix
        end
        if abs(mesh.nodes[q[2], 1] - cfg.width) < 1.0e-10 && abs(mesh.nodes[q[3], 1] - cfg.width) < 1.0e-10
            push!(top_rows, (q[2], q[3]))
        end
    end
    right_edges = reduce(vcat, [collect(t) for t in top_rows])
    right_edges_mat = reshape(right_edges, :, 2)

    stiff = assemble_stiffness(size(mesh.nodes, 1), dof_map, b_mats, detjw, dmat_e, cfg.thickness)
    fext = zeros(Float64, 2 * size(mesh.nodes, 1))
    for edge in eachrow(right_edges_mat)
        n1, n2 = edge
        p1 = view(mesh.nodes, n1, :)
        p2 = view(mesh.nodes, n2, :)
        length_edge = norm(p2 - p1)
        fe = cfg.thickness * length_edge / 2.0 .* [cfg.traction_x, 0.0, cfg.traction_x, 0.0]
        ids = [2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2]
        fext[ids] .+= fe
    end

    left_nodes = findall(i -> abs(mesh.nodes[i, 1]) < 1.0e-10, 1:size(mesh.nodes, 1))
    bc = Dict{Int, Float64}()
    for nid in left_nodes
        bc[2 * nid - 1] = 0.0
        bc[2 * nid] = 0.0
    end

    u = solve_linear_system(stiff, fext, bc)
    fields = postprocess_q4(mesh, dof_map, b_mats, dmat_e, u)
    save_fields_csv(joinpath(cfg.output_dir, "heterogeneous_elastic_fem_fields.csv"), mesh.nodes, fields)
    save_quads_csv(joinpath(cfg.output_dir, "heterogeneous_elastic_fem_mesh.csv"), mesh.quads; material_ids = material_ids)
    save_boundary_csv(joinpath(cfg.output_dir, "heterogeneous_elastic_fem_boundary_left.csv"), left_nodes)
    save_edges_csv(joinpath(cfg.output_dir, "heterogeneous_elastic_fem_right_edges.csv"), right_edges_mat)
    write_run_config(
        joinpath(cfg.output_dir, "heterogeneous_elastic_fem_run_config.txt"),
        [
            "width" => cfg.width,
            "height" => cfg.height,
            "traction_x" => cfg.traction_x,
            "young_matrix" => cfg.young_matrix,
            "young_inclusion" => cfg.young_inclusion,
            "poisson" => cfg.poisson,
            "nx" => cfg.nx,
            "ny" => cfg.ny,
        ],
    )
    @printf("Saved outputs to: %s\n", cfg.output_dir)
end

main()
