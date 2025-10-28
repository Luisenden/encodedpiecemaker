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

function sort_into_generator!(candidate::Int, progress::Vector{Vector{Any}}, steane_generators::Vector{Set{Int}})

    notadded = true
    idcs_cingen = findall([candidate ∈ gen for gen in steane_generators]) # get all indices where candidate is in the generator
    idx_take = rand(idcs_cingen) # randomly pick one of those indices
    isempty(progress[idx_take]) && push!(progress[idx_take], Set{Int}()) # initialize if empty
    for s in progress[idx_take] 
        if candidate ∉ s # check if candidate is not added yet
            push!(s, candidate)
            notadded = false # flag the candidate being added
            break
        end
    end

    if notadded
        idcs = [candidate in gen for gen in steane_generators] # get all indices where candidate is in the generator
        min_idx = argmin(map(x->length(x), progress[idcs])) # find the index with the smallest progress
        idx = findfirst(==(min_idx), cumsum(idcs)) # map back to original index
        push!(progress[idx], Set([candidate])) # add new set with candidate to that generator's progress
    end

end


function fusion(piecemaker_slot::RegRef, client_slot::RegRef)
    noisyCNOT = NonInstantGate(CNOT, TCNOT) # operation time of CNOT gate
    apply!((piecemaker_slot, client_slot), noisyCNOT)

    # TODO: measuring also takes time ... (1 unit in singh et al., 2025)
    res = project_traceout!(client_slot, Z)
    return res
end

function clear_up_qubits!(net::RegisterNet, n::Int)
    # cleanup qubits
    foreach(q -> (traceout!(q); unlock(q)), net[1])
    foreach(q -> (traceout!(q); unlock(q)), [net[1 + i][1] for i in 1:n])
end

@resumable function listen_fuse_log(sim, net)

    current_clients = Int[] # initial current_clients empty
    while true
        # Listen for Entanglementcounterpart changed on switch
        @yield onchange_tag(net[1])
        
        while true
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if !isnothing(counterpart)
                
                slot, _, _ = counterpart
                push!(current_clients, slot.idx)

                if length(current_clients) > 1 # after first bell pair has arrived
                    # fuse subsequent Bellpair with the first client
                    first_client_idx = current_clients[1]
                    @yield lock(net[1][first_client_idx]) & lock(net[1][slot.idx])
                    res = fusion(net[1][first_client_idx], net[1][slot.idx])
                    tag!(net[1 + slot.idx][1], Tag(:updateX, res)) # communicate change to client node
                    unlock(net[1][first_client_idx]); unlock(net[1][slot.idx])
                    @info "Fused client $(slot.idx) with first client $(first_client_idx)"
                end
            else
                break
            end
        end

        if length(current_clients) > 2
            @yield lock(net[1][current_clients[1]])
            res = project_traceout!(net[1][current_clients[1]], σˣ)
            unlock(net[1][current_clients[1]])

            tag!(net[1+current_clients[1]][1], Tag(:updateZ, res)) # communicate change to latest node

            Xcount = 0
            while true # wait for Xdone tags from all current clients
                @yield onchange_tag(net[1])
                msg = querydelete!(net[1], :Xdone, ❓)
                @info "Received Xdone tag: $(msg), current_clients length: $(length(current_clients))"
                if !isnothing(msg) Xcount += 1 end
                if Xcount == length(current_clients)-1 break end
            end
            
            while true # wait for Zdone tag from any client
                @yield onchange_tag(net[1])
                isdonemessage = querydelete!(net[1], :Zdone)
                if !isnothing(isdonemessage)
                    @info "received Zdone tag: $(isdonemessage)"
                    # measure the fidelity to the GHZ state
                    @yield reduce(&, [lock(net[1+i][1]) for i in current_clients])
                    obs_proj = SProjector(StabilizerState(ghzs[length(current_clients)])) # GHZ state projector to measure
                    fidelity = real(observable([net[1+i][1] for i in current_clients], obs_proj))

                    @info "clients serviced: $(current_clients) --> fidelity: $(fidelity)"
                    foreach(q -> (traceout!(q); unlock(q)), [net[1 + i][1] for i in current_clients]) # free qubits
                    unlock.(net[1+i][1] for i in current_clients)

                    timesteps = now(sim)

                    # log results
                    push!(logs, (timesteps, current_clients, fidelity))
                    break
                end
            end

            current_clients = Int[] # reset current_clients only when round completed
        end
    end
end

@resumable function correct_and_inform(sim, net::RegisterNet, client::Int)
    while true
        @yield onchange_tag(net[1+client][1])
        msg1 = querydelete!(net[1+client][1], :updateX, ❓)
        msg2 = querydelete!(net[1+client][1], :updateZ, ❓)
        if !isnothing(msg1) || !isnothing(msg2)
            if !isnothing(msg1)
                value = msg1[3][2]
                @info "X received at client $(client), with value $(value)"
                @yield lock(net[1+client][1])
                if value == 2
                    noisyX = NonInstantGate(X, TXZ) # operation time of memory qubits TODO: ask Stefan if this is done in parallel (or timeout for whole system?)
                    apply!(net[1+client][1], noisyX, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Xdone, client)) # notify central node that X correction is done
            end
            if !isnothing(msg2)
                @info "Z received at client $(client)"
                value = msg2[3][2]
                @yield lock(net[1+client][1])
                if value == 2
                    noisyZ = NonInstantGate(Z, TXZ) # operation time of memory qubit
                    apply!(net[1+client][1], noisyZ, time = now(sim))
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Zdone)) # notify central node that Z correction is done
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
    switch = Register([Qubit() for _ in 1:n], [states_representation for _ in 1:n], [noise_model for _ in 1:n]) # storage qubits at the switch, first qubit is the "piecemaker" qubit
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n] # client qubits

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    # Discrete event simulation
    sim = get_time_tracker(net)

    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1,
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = 0.001, retry_lock_time = 0.0001
        )

        @process entangler()
        @process correct_and_inform(sim, net, i)
    end

    @process listen_fuse_log(sim, net)

    return sim
end

# main 
# time is measured in 10 milliseconds

steane_generators = [Set([1,2,3,5]), Set([1,2,4,6]), Set([2,3,4,7])]
progress = Dict(1 => [], 2 => [], 3 => [])


n = 10 # number of clients
runtime = 100

TCNOT = 0.1 # CNOT gate time
TXZ = 0.1 # X and Z gate time

dataframes = DataFrame[]
logs = Tuple[]
for (i, T_link) in enumerate([10])#, 100, 1000])
    logs = Tuple[]
    sim = prepare_sim(n, T_link)
    t_wallclock = @elapsed run(sim, runtime)
    logs = DataFrame(logs, [:timesteps, :clients_serviced, :GHZfidel])
    logs = transform(logs, :timesteps => (x -> [0.0; diff(x)]) => :time_diff)
    logs = transform(logs, :clients_serviced => (x -> [length(c) for c in x]) => :num_clients)
    logs[!, "Set"] .= i
    logs[!, "wallclock_time"] .= t_wallclock

    push!(dataframes, logs)
    @info "Completed set $(i) with T_link=$(T_link), wallclock time=$(t_wallclock) seconds"
end
alllogs = vcat(dataframes...)
# grouped_stats = combine(groupby(alllogs, [:Set, :wallclock_time])) do df
#     describe(df[:, [:GHZfidel, :time_diff, :num_clients]], :mean, :std, :min, :max, :median)
# end

# @info grouped_stats
@info alllogs