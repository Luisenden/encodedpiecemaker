using QuantumSavory
using QuantumSavory: Register, X, Z, CNOT
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames
using Statistics
using Plots

const ghzs = [ghz(n) for n in 1:10] # make const in order to not build new every time

# Pre-compute client to generator mapping for O(1) lookup
const CLIENT_TO_GENERATORS = Dict(
    1 => [1, 2],
    2 => [1, 2, 3],
    3 => [1, 3],
    4 => [2, 3],
    5 => [1],
    6 => [2],
    7 => [3]
)

# Encapsulate simulation state
mutable struct SimulationState
    progress::Vector{Vector{Vector{Int}}}
    to_be_measured::Vector{Vector{Int}}
    logs::Vector{Tuple{Float64, Vector{Int}, Float64, Bool}}
end

function noisy_bell_state(target_fidelity::Float64=0.97)
    perfect_pair::StabilizerState = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return target_fidelity*perfect_pair_dm + (1-target_fidelity)*mixed_dm
end

@resumable function projectout(sim, net, slot_idx, gen_set_idx, n_clients_in_set, state::SimulationState)
    push!(state.to_be_measured, popfirst!(state.progress[gen_set_idx]))
    @yield lock(net[1][slot_idx])
    @debug "Projecting out piecemaker qubit at slot $(slot_idx), $(net[1][slot_idx])"
    res = project_traceout!(net[1][slot_idx], σˣ)
    @debug "Tagging client $(slot_idx) with Z correction result $(res) for generator set $(gen_set_idx)"
    tag!(net[1+slot_idx][1], Tag(:updateZ, res, gen_set_idx, n_clients_in_set))
    unlock(net[1][slot_idx])
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, client_slot::RegRef)
    @yield lock(piecemaker_slot) & lock(client_slot)
    apply!((piecemaker_slot, client_slot), CNOT)
    res = project_traceout!(client_slot, Z)
    tag!(net[1 + client_slot.idx][1], Tag(:updateX, res))
    unlock(piecemaker_slot)
    unlock(client_slot)
    @debug "Fused client $(client_slot.idx) with first client $(piecemaker_slot.idx)"
end

function get_oldest_generator_for_candidate(sim, net::RegisterNet, candidate::Int, state::SimulationState, steane_generators)
    # Use pre-computed mapping instead of searching
    potential_gens_indcs = CLIENT_TO_GENERATORS[candidate]
    
    # Pre-allocate and avoid repeated allocations
    n_clients = length(net.registers) - 1
    accesstimes = Vector{Float64}(undef, n_clients)
    for i in 1:n_clients
        accesstimes[i] = net.registers[i+1].accesstimes[1]
    end
    
    # Replace 0.0 with Inf in-place to avoid allocation
    accesstimes_replaced = replace(accesstimes, 0.0 => Inf)
    sorted_indices = sortperm(accesstimes_replaced)
    
    @debug "accesstimes: $(accesstimes), sorted: $(sorted_indices)"

    # Check if all potential generators are empty
    if all(isempty(state.progress[i]) for i in potential_gens_indcs)
        @debug "Progress is empty, cannot find oldest generator for candidate $(candidate), return index $(potential_gens_indcs[1])"
        return now(sim), potential_gens_indcs[1]
    end

    # Optimized search: iterate through sorted clients, check if they're in any valid generator
    for oldest_idx in sorted_indices
        for gen_idx in potential_gens_indcs
            prog = state.progress[gen_idx]
            if !isempty(prog) && oldest_idx ∈ prog[1]
                @debug "Found generator index $(gen_idx) for candidate $(candidate) with oldest client index $(oldest_idx)"
                timestamp = accesstimes[oldest_idx]
                return timestamp, gen_idx
            end
        end
    end
    
    error("No generator found for candidate $(candidate), sorted_indices: $(sorted_indices)")
end

@resumable function GeneratorServiceProt(sim, net, candidate::Int, state::SimulationState, steane_generators, t_GHZ::Float64)
    notadded = true

    (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim, net, candidate, state, steane_generators)
    isempty(state.progress[idx_steane_take]) && push!(state.progress[idx_steane_take], Vector{Int}())

    while now(sim) - timestamp > t_GHZ
        @debug "Generator set $(idx_steane_take) with clients $(state.progress[idx_steane_take][1]) TOO OLD"
        @yield @process projectout(sim, net, state.progress[idx_steane_take][1][1], idx_steane_take, length(state.progress[idx_steane_take][1]), state)
        (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim, net, candidate, state, steane_generators)
    end

    for s in state.progress[idx_steane_take]
        if candidate ∉ s
            push!(s, candidate)
            if length(s) > 1
                @yield @process fusion(sim, net, net[1][s[1]], net[1][candidate])
                @debug "fusing candidate $(candidate) into generator set index $(idx_steane_take) with current starting index $(s[1])"
            end
            notadded = false
            if length(s) == length(steane_generators[idx_steane_take])
                @debug "Generator set $(idx_steane_take) completed with clients $(s), projecting out piecemaker qubit"
                @yield @process projectout(sim, net, s[1], idx_steane_take, length(s), state)
            end
            break
        end
    end

    if notadded
        # Use pre-computed mapping
        potential_gen_idcs = CLIENT_TO_GENERATORS[candidate]
        # Find generator with smallest progress
        min_length = typemax(Int)
        idx = potential_gen_idcs[1]
        for gen_idx in potential_gen_idcs
            len = length(state.progress[gen_idx])
            if len < min_length
                min_length = len
                idx = gen_idx
            end
        end
        push!(state.progress[idx], Vector{Int}([candidate]))
    end
end

function clear_up_slots!(net::RegisterNet, n::Int)
    # Cleanup qubits - single pass
    for q in net[1]
        traceout!(q)
        unlock(q)
    end
    for i in 1:n
        traceout!(net[1 + i][1])
        unlock(net[1 + i][1])
    end
end

@resumable function listen_fuse(sim, net, state::SimulationState, steane_generators, t_GHZ::Float64)
    while true
        @yield onchange_tag(net[1])
        
        while true
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if !isnothing(counterpart)
                slot, _, _ = counterpart
                @yield @process GeneratorServiceProt(sim, net, slot.idx, state, steane_generators, t_GHZ)
                @debug "Sorted client $(slot.idx) into generator sets, current progress: $(state.progress)"
            else
                break
            end
        end
    end
end

@resumable function listen_log(sim, net, state::SimulationState, steane_generators)
    while true
        @yield onchange_tag(net[1])
        isdonemessage = querydelete!(net[1], :Zdone, ❓, ❓)

        if !isnothing(isdonemessage)
            genset = steane_generators[isdonemessage[3][2]]
            n_clients_in_set = isdonemessage[3][3]
            @debug "received Zdone tag: $(isdonemessage)"
            
            discarded = true
            fidelity = 0.0
            
            # Measure fidelity
            if length(genset) != n_clients_in_set
                @debug "MEASURE OUT BEFORE COMPLETION $(state.to_be_measured[1])"
                clients_to_measure = state.to_be_measured[1]
                @yield reduce(&, [lock(net[1+i][1]) for i in clients_to_measure])
                obs_proj = SProjector(StabilizerState(ghzs[length(clients_to_measure)]))
                fidelity = real(observable([net[1+i][1] for i in clients_to_measure], obs_proj))
                # Single cleanup - FIXED: removed double unlock
                for i in clients_to_measure
                    traceout!(net[1 + i][1])
                    unlock(net[1 + i][1])
                end
            else
                @yield reduce(&, [lock(net[1+i][1]) for i in genset])
                obs_proj = SProjector(StabilizerState(ghzs[n_clients_in_set]))
                fidelity = real(observable([net[1+i][1] for i in genset], obs_proj))
                @debug "clients serviced: $(genset) --> fidelity: $(fidelity)"
                # Single cleanup - FIXED: removed double unlock
                for i in genset
                    traceout!(net[1 + i][1])
                    unlock(net[1 + i][1])
                end
                discarded = false
            end
            
            timesteps = now(sim)
            push!(state.logs, (timesteps, popfirst!(state.to_be_measured), fidelity, discarded))
            @debug "Updated progress: $(state.progress)"
        end
    end
end

@resumable function correct_and_inform(sim, net::RegisterNet, client::Int)
    while true
        @yield onchange_tag(net[1+client][1])
        msg1 = querydelete!(net[1+client][1], :updateX, ❓)
        msg2 = querydelete!(net[1+client][1], :updateZ, ❓, ❓, ❓)
        
        if !isnothing(msg1) || !isnothing(msg2)
            if !isnothing(msg1)
                value = msg1[3][2]
                @debug "X received at client $(client), with value $(value)"
                @yield lock(net[1+client][1])
                if value == 2
                    apply!(net[1+client][1], X, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Xdone, client))
            end
            
            if !isnothing(msg2)
                @debug "Z received at client $(client)"
                value = msg2[3][2]
                gen_set_idx = msg2[3][3]
                n_clients_in_set = msg2[3][4]
                @debug "Z received at client $(client), with value $(value), gen_set_idx=$(gen_set_idx), n_clients_in_set=$(n_clients_in_set)"
                @yield lock(net[1+client][1])
                if value == 2
                    noisyZ = NonInstantGate(Z, TXZ)
                    apply!(net[1+client][1], noisyZ, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Zdone, gen_set_idx, n_clients_in_set))
            end
        end
    end
end

function prepare_sim(n, T_link, t_GHZ::Float64, F_link::Float64, steane_generators)
    states_representation = QuantumOpticsRepr()
    noise_model = Depolarization(T_link)

    # Initialize simulation state
    state = SimulationState(
        [Vector{Vector{Int}}() for _ in 1:length(steane_generators)],
        Vector{Vector{Int}}(),
        Vector{Tuple{Float64, Vector{Int}, Float64, Bool}}()
    )

    # Network setup
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n])
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n]

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    sim = get_time_tracker(net)

    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1, 
            pairstate = noisy_bell_state(F_link),
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = 6e-6,
            retry_lock_time = 1e-6, local_busy_time_post = 0.0
        )

        @process entangler()
        @process correct_and_inform(sim, net, i)
    end

    @process listen_fuse(sim, net, state, steane_generators, t_GHZ)
    @process listen_log(sim, net, state, steane_generators)

    return sim, state
end

# Main simulation, measures everything in seconds (s)
steane_generators = [[1,2,3,5], [1,2,4,6], [2,3,4,7]]

n = 7

runtime = 0.1 # 1 second
link_success_prob = 0.1 # 0.0001

TCNOT = 500e-6 # 500 microseconds
TXZ = 1.4e-7 # 0.14 microseconds
T = 1.0 # s

dataframes = DataFrame[]
for F_link in [1.0, 0.97, 0.95]

    for t_GHZ in [10e-6, 100e-6, 1e-3, Inf]
        sim, state = prepare_sim(n, T, t_GHZ, F_link, steane_generators)
        t_wallclock = @elapsed run(sim, runtime)
        
        logs = DataFrame(state.logs, [:timesteps, :clients_serviced, :GHZfidel, :discarded])
        logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)
        logs = transform(logs, :clients_serviced => (x -> [length(c) for c in x]) => :num_clients)
        
        logs[!, "F_link"] .= F_link
        logs[!, "t_GHZ"] .= t_GHZ
        logs[!, "wallclock_time"] .= t_wallclock

        push!(dataframes, logs)
        @info "Completed set $(t_GHZ) in wallclock_time $(t_wallclock) seconds"
    end
end

alllogs = vcat(dataframes...)
@debug "Summary statistics:"
@debug describe(alllogs)
##
using StatsPlots
using LaTeXStrings

filtered_logs = alllogs[(alllogs.discarded .== false), :]

for F_link in unique(filtered_logs.F_link)
    subset = filtered_logs[filtered_logs.F_link .== F_link, :]

    @df subset groupedhist(:GHZfidel,
    group=:t_GHZ,
    bar_position=:stack,
    bins=6,
    title="GHZ Fidelity Distribution",
    xlabel=L"\mathrm{GHZ\ Fidelity}",
    ylabel=L"\mathrm{Count}",
    legend=:topleft,
    labels=reshape([isinf(t) ? "no cutoff" : L"t_{\mathrm{GHZ}} = %$(t*1000) \mathrm{ms}" for t in sort(unique(subset.t_GHZ))], 1, :),
    size=(400, 300),
    )

    savefig("ghz_fidelity_groupedhist_F$(F_link).pdf")



    # Plot the number of non-discarded GHZ states over time for different t_GHZ values
    cumulative_data = combine(groupby(sort(subset, :timesteps), :t_GHZ)) do df
        DataFrame(
            timesteps = df.timesteps,
            cumulative_count = 1:nrow(df) #cumsum(.!df.discarded) #./ nrow(df)
        )
    end

    # Using StatsPlots @df macro with grouping
    @df cumulative_data plot(:timesteps, :cumulative_count,
        group=:t_GHZ,
        yscale=:log10,
        ylims = (1,1000),
        xlabel="Time (s)",
        ylabel="# GHZ states",
        title=L"GHZ States with cut-offs $t_{\mathrm{GHZ}}$",
        legend=:topleft,
        linewidth=2,
        size=(400, 300),
        labels=reshape([isinf(t) ? "no cutoff" : L"t_{\mathrm{GHZ}} = %$(t*1000) \mathrm{ms}" for t in sort(unique(cumulative_data.t_GHZ))], 1, :)
        )


    savefig("ghz_cumulative_over_time+$(F_link).pdf")

end