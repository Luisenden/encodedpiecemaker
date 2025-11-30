using QuantumSavory
using QuantumSavory: Register, X, Z, CNOT
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames

const ghzs = [ghz(n) for n in 1:10] # make const in order to not build new every time

function bicycle_generators(p::Integer; SA=[0], SB=[0,2], one_based::Bool=false)
    p ≥ 1 || throw(ArgumentError("p must be ≥ 1"))
    # Normalize masks to 0..p-1 and deduplicate
    SA = unique(mod.(SA, p))
    SB = unique(mod.(SB, p))

    Xgen = Vector{Vector{Int}}(undef, p)
    Zgen = Vector{Vector{Int}}(undef, p)

    for i in 0:p-1
        # X_i
        L_A = [mod(i + s, p) for s in SA]
        R_B = [p + mod(i + t, p) for t in SB]
        Xgen[i+1] = vcat(L_A, R_B)

        # Z_i
        L_B = [mod(i + t, p) for t in SB]
        R_A = [p + mod(i + s, p) for s in SA]
        Zgen[i+1] = vcat(L_B, R_A)
    end

    if one_based
        Xgen = [x .+ 1 for x in Xgen]
        Zgen = [z .+ 1 for z in Zgen]
    end

    return Xgen, Zgen
end

function module_counts(gens::Vector{<:AbstractVector{<:Integer}};
                       n::Union{Nothing,Int}=nothing,
                       one_based::Bool=false,
                       unique_within::Bool=true)
    isempty(gens) && throw(ArgumentError("gens must be non-empty"))

    # Infer n if needed
    if n === nothing
        maxidx = maximum(maximum(g) for g in gens if !isempty(g))
        n = one_based ? maxidx : maxidx + 1
    end

    counts = zeros(Int, n)
    for g in gens
        items = unique_within ? unique(g) : g
        for idx in items
            if one_based
                1 ≤ idx ≤ n || throw(ArgumentError("index $idx out of 1..$n"))
                counts[idx] += 1
            else
                0 ≤ idx ≤ n-1 || throw(ArgumentError("index $idx out of 0..$(n-1)"))
                counts[idx + 1] += 1
            end
        end
    end
    return counts
end

function fusion(piecemaker_slot::RegRef, client_slot::RegRef)
    noisyCNOT = NonInstantGate(CNOT, 0.1) # operation time of CNOT gate
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
    
    initialize!(net[1][n+1], X1, time = now(sim)) # TODO: this should only be done when first Bellpair arrives

    current_clients = Int[] # initial empty current_clients
    while true
        # Listen for Entanglementcounterpart changed on switch
        @yield onchange_tag(net[1])
        
        while true
            current_clientspart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if !isnothing(current_clientspart)
                slot, _, _ = current_clientspart

                # fuse the Bellpair with the piecemaker qubit
                @yield lock(net[1][n+1]) & lock(net[1][slot.idx])
                res = fusion(net[1][n+1], net[1][slot.idx])
                tag!(net[1 + slot.idx][1], Tag(:updateX, res)) # communicate change to client node
                unlock(net[1][n+1]); unlock(net[1][slot.idx])
                push!(current_clients, slot.idx)
                @info "Fused client $(slot.idx) with piecemaker qubit"
            else
                break
            end
        end

        if length(current_clients) > 2
            @yield lock(net[1][n+1])
            res = project_traceout!(net[1][n+1], σˣ)
            unlock(net[1][n+1])

            tag!(net[1+current_clients[1]][1], Tag(:updateZ, res)) # communicate change to latest node

            Xcount = 0
            while true # wait for Xdone tags from all current clients
                @yield onchange_tag(net[1])
                msg = querydelete!(net[1], :Xdone, ❓)
                @info "Received Xdone tag: $(msg), current_clients length: $(length(current_clients))"
                if !isnothing(msg) Xcount += 1 end
                if Xcount == length(current_clients) break end
            end
            
            while true # wait for Zdone tag from any client
                @yield onchange_tag(net[1])
                isdonemessage = querydelete!(net[1], :Zdone)
                if !isnothing(isdonemessage)
                    @info "received Zdone tag: $(isdonemessage)"
                    # measure the fidelity to the GHZ state
                    @yield reduce(&, [lock(net[1+i][1]) for i in current_clients])
                    obs_proj = SProjector(StabilizerState(ghzs[length(current_clients)])) # GHZ state projector to measure
                    fidelity = real(observable([net[1+i][1] for i in current_clients], obs_proj; time = now(sim)))

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
            #@yield timeout(sim, 0.1) # operation time of memory qubit
            initialize!(net[1][n+1], X1, time = now(sim)) # re-initialize piecemaker qubit
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
                @debug "X received at client $(client), with value $(value)"
                @yield lock(net[1+client][1])
                if value == 2
                    noisyX = NonInstantGate(X, 0.1) # operation time of memory qubits TODO: ask Stefan if this is done in parallel (or timeout for whole system?)
                    apply!(net[1+client][1], noisyX)
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Xdone, client)) # notify central node that X correction is done
            end
            if !isnothing(msg2)
                @debug "Z received at client $(client)"
                value = msg2[3][2]
                @yield lock(net[1+client][1])
                if value == 2
                    noisyZ = NonInstantGate(Z, 0.1) # operation time of memory qubit
                    apply!(net[1+client][1], noisyZ)
                end
                unlock(net[1+client][1])
                tag!(net[1][1], Tag(:Zdone)) # notify central node that Z correction is done
            end
        end
    end
end


function prepare_sim(n)
    states_representation = QuantumOpticsRepr()
    
    T_link = 10
    noise_model = Depolarization(T_link) # noise model applied to the memory qubits

    # Link success probability
    link_success_prob = 1.0 #0.0001

    # Network setup
    switch = Register([Qubit() for _ in 1:(n+1)], [states_representation for _ in 1:(n+1)], [noise_model for _ in 1:(n+1)]) # storage qubits at the switch, first qubit is the "piecemaker" qubit
    clients = [Register([Qubit()], [states_representation], [noise_model]) for _ in 1:n] # client qubits

    graph = star_graph(n+1)
    net = RegisterNet(graph, [switch, clients...])

    # Discrete event simulation
    sim = get_time_tracker(net)

    for i in 1:n
        entangler = EntanglerProt(
            sim = sim, net = net, nodeA = 1, chooseA = i, nodeB = 1 + i, chooseB = 1,
            success_prob = link_success_prob, rounds = -1, attempts = -1, attempt_time = 0.001,
        )

        @process entangler()
        @process correct_and_inform(sim, net, i)
    end

    @process listen_fuse_log(sim, net)

    return sim
end

# main 

logs = Tuple[]
n = 20 # number of clients
runtime = 5

Xgen, Zgen = bicycle_generators(10; SA=[0], SB=[0,2], one_based=true)
modcounts_X = module_counts(Xgen; one_based=true)
modcounts_Z = module_counts(Zgen; one_based=true)

@info "Bicycle code generators for n=$(n):"
@info "X generators: $(Xgen)"
@info "Z generators: $(Zgen)"
@info "Module counts for X generators: $(modcounts_X)"
@info "Module counts for Z generators: $(modcounts_Z)"

sim = prepare_sim(n)
run(sim, runtime) # run for 100000 time units


logs = DataFrame(logs, [:timesteps, :clients_serviced, :GHZfidel])
@info "Final logs: $(logs)"