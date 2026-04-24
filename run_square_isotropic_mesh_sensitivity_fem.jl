using Printf

Base.@kwdef mutable struct Config
    paper_root::String = ""
    output_root::String = ""
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
        setproperty!(cfg, name, val)
        i += 2
    end
    paper_root = dirname(@__DIR__)
    isempty(cfg.paper_root) && (cfg.paper_root = paper_root)
    isempty(cfg.output_root) && (cfg.output_root = joinpath(paper_root, "outputs", "mesh", "square_isotropic_mesh_sensitivity"))
    return cfg
end

function mesh_cases()
    return [
        (label = "mesh_01_12x4", ntheta = 12, nradial = 4),
        (label = "mesh_02_24x8", ntheta = 24, nradial = 8),
        (label = "mesh_03_48x16", ntheta = 48, nradial = 16),
        (label = "mesh_04_96x32", ntheta = 96, nradial = 32),
    ]
end

function read_last_reaction(path::String)
    rows = readlines(path)
    length(rows) >= 2 || return NaN
    cols = split(rows[end], ",")
    return parse(Float64, cols[end])
end

function write_plan_csv(path::String, cases, fem_dirs)
    open(path, "w") do io
        println(io, "label,ntheta,nradial,nnodes,nelements,fem_dir")
        for (case, fem_dir) in zip(cases, fem_dirs)
            nnodes = (case.nradial + 1) * (case.ntheta + 1)
            nelements = 2 * case.nradial * case.ntheta
            @printf(io, "%s,%d,%d,%d,%d,%s\n", case.label, case.ntheta, case.nradial, nnodes, nelements, fem_dir)
        end
    end
end

function write_summary_csv(path::String, cases, fem_dirs, elapsed_list, reaction_list)
    open(path, "w") do io
        println(io, "label,ntheta,nradial,nnodes,nelements,elapsed_sec,final_reaction_y,fem_dir")
        for (case, fem_dir, elapsed_sec, reaction_y) in zip(cases, fem_dirs, elapsed_list, reaction_list)
            nnodes = (case.nradial + 1) * (case.ntheta + 1)
            nelements = 2 * case.nradial * case.ntheta
            @printf(io, "%s,%d,%d,%d,%d,%.6f,%.12e,%s\n", case.label, case.ntheta, case.nradial, nnodes, nelements, elapsed_sec, reaction_y, fem_dir)
        end
    end
end

function main()
    cfg = parse_cli!(Config())
    fem_script = joinpath(cfg.paper_root, "plastic", "perforated_plate", "perforated_plate_square_fem_julia.jl")
    isfile(fem_script) || error("Missing FEM script: $fem_script")
    mkpath(cfg.output_root)
    fem_root = joinpath(cfg.output_root, "fem")
    mkpath(fem_root)

    cases = mesh_cases()
    fem_dirs = String[]
    elapsed_list = Float64[]
    reaction_list = Float64[]

    for case in cases
        out_dir = joinpath(fem_root, case.label)
        mkpath(out_dir)
        push!(fem_dirs, out_dir)
        cmd = `$(Base.julia_cmd()) $fem_script --hardening-mode isotropic --ntheta $(case.ntheta) --nradial $(case.nradial) --output-dir $out_dir`
        @printf("Running FEM mesh case %s | ntheta=%d | nradial=%d\n", case.label, case.ntheta, case.nradial)
        tick = time()
        run(cmd)
        elapsed_sec = time() - tick
        reaction_y = read_last_reaction(joinpath(out_dir, "perforated_plate_square_fem_load_curve.csv"))
        push!(elapsed_list, elapsed_sec)
        push!(reaction_list, reaction_y)
    end

    write_plan_csv(joinpath(cfg.output_root, "mesh_sensitivity_plan.csv"), cases, fem_dirs)
    write_summary_csv(joinpath(cfg.output_root, "mesh_sensitivity_fem_summary.csv"), cases, fem_dirs, elapsed_list, reaction_list)
    @printf("Saved outputs to: %s\n", cfg.output_root)
end

main()
