using LinearAlgebra
using Printf
using SparseArrays

Base.@kwdef mutable struct Config
    width::Float64 = 0.6
    height::Float64 = 0.6
    radius::Float64 = 0.1
    thickness::Float64 = 1.0
    traction_x::Float64 = 1.0
    young::Float64 = 20.0
    poisson::Float64 = 0.25
    ntheta::Int = 48
    nradial::Int = 18
    radial_bias::Float64 = 1.8
    output_dir::String = ""
end

struct MeshData
    nodes::Matrix{Float64}
    triangles::Matrix{Int}
    left_nodes::Vector{Int}
    bottom_nodes::Vector{Int}
    right_nodes::Vector{Int}
    right_edges::Matrix{Int}
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
        cfg.output_dir = joinpath(@__DIR__, "outputs", "defected_plate_fem")
    end
    return cfg
end

plane_stress_matrix(cfg::Config) = (cfg.young / (1.0 - cfg.poisson^2)) .* [
    1.0 cfg.poisson 0.0
    cfg.poisson 1.0 0.0
    0.0 0.0 (1.0 - cfg.poisson) / 2.0
]

function outer_radius(cfg::Config, theta::Float64)
    cx = cos(theta)
    sy = sin(theta)
    rx = abs(cx) < 1.0e-12 ? Inf : cfg.width / cx
    ry = abs(sy) < 1.0e-12 ? Inf : cfg.height / sy
    return min(rx, ry)
end

node_id(ir::Int, it::Int, nradial::Int) = ir + 1 + it * (nradial + 1)

function build_mesh(cfg::Config)
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
    for jt in 0:(cfg.ntheta - 1), ir in 0:(cfg.nradial - 1)
        n0 = node_id(ir, jt, cfg.nradial)
        n1 = node_id(ir + 1, jt, cfg.nradial)
        n2 = node_id(ir, jt + 1, cfg.nradial)
        n3 = node_id(ir + 1, jt + 1, cfg.nradial)
        triangles[tid, :] .= (n0, n1, n3)
        triangles[tid + 1, :] .= (n0, n3, n2)
        tid += 2
    end

    tol = 1.0e-10
    left_nodes = Int[]
    bottom_nodes = Int[]
    right_nodes = Int[]
    outer_nodes = [node_id(cfg.nradial, jt, cfg.nradial) for jt in 0:cfg.ntheta]
    for nid in 1:size(nodes, 1)
        x = nodes[nid, 1]
        y = nodes[nid, 2]
        abs(x) < tol && push!(left_nodes, nid)
        abs(y) < tol && push!(bottom_nodes, nid)
        abs(x - cfg.width) < tol && push!(right_nodes, nid)
    end
    edge_buf = Vector{NTuple{2, Int}}()
    for k in 1:(length(outer_nodes) - 1)
        n1 = outer_nodes[k]
        n2 = outer_nodes[k + 1]
        if abs(nodes[n1, 1] - cfg.width) < tol && abs(nodes[n2, 1] - cfg.width) < tol
            push!(edge_buf, (n1, n2))
        end
    end
    right_edges = zeros(Int, length(edge_buf), 2)
    for (i, edge) in enumerate(edge_buf)
        right_edges[i, 1] = edge[1]
        right_edges[i, 2] = edge[2]
    end
    return MeshData(nodes, triangles, sort!(unique(left_nodes)), sort!(unique(bottom_nodes)), sort!(unique(right_nodes)), right_edges)
end

function build_tri_operators(mesh::MeshData)
    ne = size(mesh.triangles, 1)
    b_mats = zeros(Float64, ne, 3, 6)
    areas = zeros(Float64, ne)
    for e in 1:ne
        tri = mesh.triangles[e, :]
        xy = mesh.nodes[tri, :]
        x1, y1 = xy[1, 1], xy[1, 2]
        x2, y2 = xy[2, 1], xy[2, 2]
        x3, y3 = xy[3, 1], xy[3, 2]
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        beta = [y2 - y3, y3 - y1, y1 - y2]
        gamma = [x3 - x2, x1 - x3, x2 - x1]
        b = zeros(Float64, 3, 6)
        for a in 1:3
            dndx = beta[a] / area2
            dndy = gamma[a] / area2
            b[1, 2 * a - 1] = dndx
            b[2, 2 * a] = dndy
            b[3, 2 * a - 1] = dndy
            b[3, 2 * a] = dndx
        end
        b_mats[e, :, :] .= b
        areas[e] = 0.5 * abs(area2)
    end
    return b_mats, areas
end

function build_dof_map(triangles::Matrix{Int})
    dof_map = zeros(Int, size(triangles, 1), 6)
    for e in 1:size(triangles, 1), (a, nid) in enumerate(triangles[e, :])
        dof_map[e, 2 * a - 1] = 2 * nid - 1
        dof_map[e, 2 * a] = 2 * nid
    end
    return dof_map
end

function assemble_stiffness(mesh::MeshData, dof_map::Matrix{Int}, b_mats::Array{Float64, 3}, areas::Vector{Float64}, dmat::Matrix{Float64}, thickness::Float64)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for e in 1:size(mesh.triangles, 1)
        b = Matrix(view(b_mats, e, :, :))
        ke = thickness * areas[e] .* (transpose(b) * dmat * b)
        ids = dof_map[e, :]
        for i in 1:6, j in 1:6
            push!(rows, ids[i])
            push!(cols, ids[j])
            push!(vals, ke[i, j])
        end
    end
    return sparse(rows, cols, vals, 2 * size(mesh.nodes, 1), 2 * size(mesh.nodes, 1))
end

function solve_linear(stiff::SparseMatrixCSC{Float64, Int}, fext::Vector{Float64}, bc::Dict{Int, Float64})
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

function nodal_average(nnodes::Int, triangles::Matrix{Int}, elem_values::Matrix{Float64})
    nodal = zeros(Float64, nnodes, size(elem_values, 2))
    counts = zeros(Float64, nnodes)
    for e in 1:size(triangles, 1)
        for nid in triangles[e, :]
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

function postprocess(mesh::MeshData, dof_map::Matrix{Int}, b_mats::Array{Float64, 3}, dmat::Matrix{Float64}, u::Vector{Float64})
    ne = size(mesh.triangles, 1)
    stress_e = zeros(Float64, ne, 3)
    for e in 1:ne
        ue = u[dof_map[e, :]]
        strain = Matrix(view(b_mats, e, :, :)) * ue
        stress_e[e, :] .= dmat * strain
    end
    stress_node = nodal_average(size(mesh.nodes, 1), mesh.triangles, stress_e)
    return Dict(
        "ux" => u[1:2:end],
        "uy" => u[2:2:end],
        "sx" => stress_node[:, 1],
        "sy" => stress_node[:, 2],
        "sxy" => stress_node[:, 3],
    )
end

function save_fields_csv(path::String, nodes::Matrix{Float64}, fields::Dict{String, Vector{Float64}})
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

function save_triangles_csv(path::String, triangles::Matrix{Int})
    open(path, "w") do io
        println(io, "element_id,n1,n2,n3")
        for eid in 1:size(triangles, 1)
            tri = triangles[eid, :]
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

function save_edges_csv(path::String, edges::Matrix{Int})
    open(path, "w") do io
        println(io, "edge_id,n1,n2")
        for eid in 1:size(edges, 1)
            @printf(io, "%d,%d,%d\n", eid, edges[eid, 1], edges[eid, 2])
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

function main()
    cfg = parse_cli!(Config())
    mkpath(cfg.output_dir)
    mesh = build_mesh(cfg)
    b_mats, areas = build_tri_operators(mesh)
    dof_map = build_dof_map(mesh.triangles)
    dmat = plane_stress_matrix(cfg)
    stiff = assemble_stiffness(mesh, dof_map, b_mats, areas, dmat, cfg.thickness)
    fext = zeros(Float64, 2 * size(mesh.nodes, 1))
    for edge in eachrow(mesh.right_edges)
        n1, n2 = edge
        p1 = view(mesh.nodes, n1, :)
        p2 = view(mesh.nodes, n2, :)
        length_edge = norm(p2 - p1)
        fe = cfg.thickness * length_edge / 2.0 .* [cfg.traction_x, 0.0, cfg.traction_x, 0.0]
        ids = [2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2]
        fext[ids] .+= fe
    end

    bc = Dict{Int, Float64}()
    for nid in mesh.left_nodes
        bc[2 * nid - 1] = 0.0
    end
    for nid in mesh.bottom_nodes
        bc[2 * nid] = 0.0
    end

    u = solve_linear(stiff, fext, bc)
    fields = postprocess(mesh, dof_map, b_mats, dmat, u)
    save_fields_csv(joinpath(cfg.output_dir, "defected_plate_fem_fields.csv"), mesh.nodes, fields)
    save_triangles_csv(joinpath(cfg.output_dir, "defected_plate_fem_mesh.csv"), mesh.triangles)
    save_boundary_csv(joinpath(cfg.output_dir, "defected_plate_fem_boundary_left.csv"), mesh.left_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "defected_plate_fem_boundary_bottom.csv"), mesh.bottom_nodes)
    save_boundary_csv(joinpath(cfg.output_dir, "defected_plate_fem_boundary_right.csv"), mesh.right_nodes)
    save_edges_csv(joinpath(cfg.output_dir, "defected_plate_fem_right_edges.csv"), mesh.right_edges)
    write_run_config(joinpath(cfg.output_dir, "defected_plate_fem_run_config.txt"), cfg)
    @printf("Saved outputs to: %s\n", cfg.output_dir)
end

main()
