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

const ghzs = [ghz(n) for n in 1:10] # make const in order to not build new every time

function noisy_bell_state(target_fidelity::Float64=0.97)
    perfect_pair = StabilizerState("XX ZZ")
    perfect_pair_dm = SProjector(perfect_pair)
    mixed_dm = MixedState(SProjector(perfect_pair))
    return target_fidelity*perfect_pair_dm + (1-target_fidelity)*mixed_dm
end


@resumable function projectout(sim, net, slot_idx, gen_set_idx, n_clients_in_set)
    push!(to_be_measured, popfirst!(progress[gen_set_idx]))
    @yield lock(net[1][slot_idx])
    @debug "Projecting out piecemaker qubit at slot $(slot_idx), $(net[1][slot_idx])"
    res = project_traceout!(net[1][slot_idx], σˣ)
    @debug "Tagging client $(slot_idx) with Z correction result $(res) for generator set $(gen_set_idx)"
    tag!(net[1+slot_idx][1], Tag(:updateZ, res, gen_set_idx, n_clients_in_set)) # communicate change to latest node
    unlock(net[1][slot_idx])
end

@resumable function fusion(sim, net, piecemaker_slot::RegRef, client_slot::RegRef)
    @yield lock(piecemaker_slot) & lock(client_slot)
    #noisyCNOT = NonInstantGate(CNOT, TCNOT) # operation time of CNOT gate
    apply!((piecemaker_slot, client_slot), CNOT)

    # TODO: measuring also takes time ... (1 unit in singh et al., 2025)
    res = project_traceout!(client_slot, Z)
    tag!(net[1 + client_slot.idx][1], Tag(:updateX, res)) # communicate change to client node
    unlock(piecemaker_slot); unlock(client_slot)

    @debug "Fused client $(client_slot.idx) with first client $(piecemaker_slot.idx)"
end

function get_oldest_generator_for_candidate(sim, net::RegisterNet, candidate::Int, progress)

    potential_gens_indcs = findall([candidate in gen for gen in steane_generators]) # get all indices where candidate is in the generator
    accesstimes = reduce(vcat, [reg.accesstimes for reg in net.registers[2:end]])
    accesstimes_replaced = replace(accesstimes, 0.0 => Inf)  # Replace 0.0 with Inf to ignore unaccessed clients
    sorted_indices = sortperm(accesstimes_replaced)  # Indices of clients sorted by access time
    @debug "accesstimes: $(accesstimes), sorted: $(sorted_indices)"

    if all(isempty.(progress[potential_gens_indcs]))
        @debug "Progress is empty, cannot find oldest generator for candidate $(candidate), return index $(potential_gens_indcs[1])"
        return now(sim), potential_gens_indcs[1]
    end

    for oldest_idx in sorted_indices
        for (i, prog) in enumerate(progress)
            if !isempty(prog) && (candidate ∈ steane_generators[i])
                if oldest_idx ∈ prog[1]
                    @debug "Found generator index $(i) for candidate $(candidate) with oldest client index $(oldest_idx)"
                    timestamp = accesstimes[oldest_idx]
                    return timestamp, i
                end
            end
        end
    end
    error("No generator found for candidate $(candidate), sorted_indices: $(sorted_indices)")
end

@resumable function GeneratorServiceProt(sim, net, candidate::Int, progress::Vector{Vector{Any}}, steane_generators)

    notadded = true

    (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim ,net, candidate, progress)
    isempty(progress[idx_steane_take]) && push!(progress[idx_steane_take], Vector{Any}()) # initialize if empty

    while now(sim) - timestamp > t_GHZ
        @debug "Generator set $(idx_steane_take) with clients $(progress[idx_steane_take][1]) TOO OLD"
        @yield @process projectout(sim, net, progress[idx_steane_take][1][1], idx_steane_take, length(progress[idx_steane_take][1]))
        (timestamp, idx_steane_take) = get_oldest_generator_for_candidate(sim, net, candidate, progress)
    end

    for s in progress[idx_steane_take]
        if candidate ∉ s # check if candidate is not added yet
            push!(s, candidate)
            if length(s) > 1
                @yield @process fusion(sim, net, net[1][s[1]], net[1][candidate])
                @debug "fusing candidate $(candidate) into generator set index $(idx_steane_take) with current starting index $(s[1])"
            end
            notadded = false # flag the candidate being added
            if length(s) == length(steane_generators[idx_steane_take]) # check if full set is reached, if so measure the piecemaker qubit
                @debug "Generator set $(idx_steane_take) completed with clients $(s), projecting out piecemaker qubit"
                @yield @process projectout(sim, net, s[1], idx_steane_take, length(s))
            end
            break
        end
    end


    if notadded
        idcs = [candidate in gen for gen in steane_generators] # get all indices where candidate is in the generator
        min_idx = argmin(map(x->length(x), progress[idcs])) # find the index with the smallest progress
        idx = findfirst(==(min_idx), cumsum(idcs)) # map back to original index
        push!(progress[idx], Vector{Any}([candidate])) # add new set with candidate to that generator's progress
    end
end



function clear_up_slots!(net::RegisterNet, n::Int)
    # cleanup qubits
    foreach(q -> (traceout!(q); unlock(q)), net[1])
    foreach(q -> (traceout!(q); unlock(q)), [net[1 + i][1] for i in 1:n])
end

@resumable function listen_fuse(sim, net)
    while true
        # Listen for Entanglementcounterpart changed on switch
        @yield onchange_tag(net[1])
        
        while true
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if !isnothing(counterpart)
                
                slot, _, _ = counterpart
                @yield @process GeneratorServiceProt(sim, net, slot.idx, progress, steane_generators)
                @debug "Sorted client $(slot.idx) into generator sets, current progress: $(progress)"
            else
                break
            end
        end
    end
end

@resumable function listen_log(sim, net)
    while true # wait for Zdone tag from any client
        @yield onchange_tag(net[1])
        isdonemessage = querydelete!(net[1], :Zdone, ❓, ❓)

        discarded = true
        if !isnothing(isdonemessage)
            genset = steane_generators[isdonemessage[3][2]]
            n_clients_in_set = isdonemessage[3][3]
            @debug "received Zdone tag: $(isdonemessage)"
            # measure the fidelity to the GHZ state
            if length(genset) != n_clients_in_set
                @info "MEASURE OUT BEFORE COMPLETION $(to_be_measured[1])"
                @yield reduce(&, [lock(net[1+i][1]) for i in to_be_measured[1]])
                obs_proj = SProjector(StabilizerState(ghzs[length(to_be_measured[1])])) # GHZ state projector to measure
                fidelity = real(observable([net[1+i][1] for i in to_be_measured[1]], obs_proj))
                foreach(q -> (traceout!(q); unlock(q)), [net[1 + i][1] for i in to_be_measured[1]]) # free qubits
                unlock.(net[1+i][1] for i in to_be_measured[1])
            else
                @yield reduce(&, [lock(net[1+i][1]) for i in genset])
                obs_proj = SProjector(StabilizerState(ghzs[n_clients_in_set])) # GHZ state projector to measure
                fidelity = real(observable([net[1+i][1] for i in genset], obs_proj))

                @info "clients serviced: $(genset) --> fidelity: $(fidelity)"
                foreach(q -> (traceout!(q); unlock(q)), [net[1 + i][1] for i in genset]) # free qubits
                unlock.(net[1+i][1] for i in genset)
                discarded = false
            end
            timesteps = now(sim)

            push!(logs, (timesteps, popfirst!(to_be_measured), fidelity, discarded))
            @debug "Updated progress: $(progress)"
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
                    #noisyX = NonInstantGate(X, TXZ) # operation time of memory qubits TODO: ask Stefan if this is done in parallel (or timeout for whole system?)
                    apply!(net[1+client][1], X, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Xdone, client)) # notify central node that X correction is done
            end
            if !isnothing(msg2)
                @debug "Z received at client $(client)"
                value = msg2[3][2]
                gen_set_idx = msg2[3][3]
                n_clients_in_set = msg2[3][4]
                @debug "Z received at client $(client), with value $(value), gen_set_idx=$(gen_set_idx), n_clients_in_set=$(n_clients_in_set)"
                @yield lock(net[1+client][1])
                if value == 2
                    noisyZ = NonInstantGate(Z, TXZ) # operation time of memory qubit
                    apply!(net[1+client][1], noisyZ, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Zdone, gen_set_idx, n_clients_in_set)) # notify central node that Z correction is done
            end
        end
    end
end


function prepare_sim(n, T_link)
    states_representation = QuantumOpticsRepr()
    
    noise_model = Depolarization(T_link) # noise model applied to the memory qubits

    # Link success probability
    link_success_prob = 0.0001

    # Network setup
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n]) # storage qubits at the switch
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n] # client qubits

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    # Discrete event simulation
    sim = get_time_tracker(net)

    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1, pairstate=noisy_bell_state(),
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = 0.001, retry_lock_time = 0.0001,
            local_busy_time_post = 0.0
        )

        @process entangler()
        @process correct_and_inform(sim, net, i)
    end

    @process listen_fuse(sim, net)
    @process listen_log(sim, net)

    return sim
end

# main 
# 1 time unit = 10 milliseconds

steane_generators = [[1,2,3,5], [1,2,4,6], [2,3,4,7]]
progress = [[], [], []]

to_be_measured = []


n = 7 # number of clients
runtime = 1000

TCNOT = 0.1 # CNOT gate time
TXZ = 0.1 * 4 # X and Z gate time
t_GHZ = 15.0 # GHZ lifetime
T = 100 # coherence time of qubits (1s)

dataframes = DataFrame[]
logs = Tuple[]
for t_GHZ in [5.0, 15.0, 30.0, 45.0]
    logs = Tuple[]
    sim = prepare_sim(n, T)
    t_wallclock = @elapsed run(sim, runtime)
    logs = DataFrame(logs, [:timesteps, :clients_serviced, :GHZfidel, :discarded])
    logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)
    logs = transform(logs, :clients_serviced => (x -> [length(c) for c in x]) => :num_clients)
    logs[!, "t_GHZ"] .= t_GHZ
    logs[!, "wallclock_time"] .= t_wallclock

    push!(dataframes, logs)
    @debug "Completed set $(t_GHZ) in wallclock_time $(t_wallclock) seconds"
end

alllogs = vcat(dataframes...)
@info alllogs
##
using Plots

@debug (length(alllogs.discarded)-sum(alllogs.discarded))/length(alllogs.discarded) * 100.0 # percentage of discarded

# Filter before creating histogram
filtered_logs = alllogs[alllogs.discarded .== false, :]
histogram(filtered_logs.GHZfidel, 
    title="Distribution of GHZ Fidelities (Non-discarded)",
    xlabel="GHZ Fidelity", 
    ylabel="Count",
    bins=30)