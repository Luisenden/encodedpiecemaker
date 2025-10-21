using QuantumSavory
using QuantumSavory: Register, X, Z, CNOT, @debugstantGate
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames

const ghzs = [ghz(n) for n in 1:7] # make const in order to not build new every time

function fusion(piecemaker_slot::RegRef, client_slot::RegRef)
    noisyCNOT = NonInstantGate(CNOT, 0.2) # operation time of CNOT gate
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

    counter = Int[] # initial empty counter
    while true
        # Listen for EntanglementCounterpart changed on switch
        @yield onchange_tag(net[1])
        
        while true
            counterpart = querydelete!(net[1], EntanglementCounterpart, ❓, ❓)
            if !isnothing(counterpart)
                slot, _, _ = counterpart

                # fuse the Bellpair with the piecemaker qubit
                @yield lock(net[1][n+1]) & lock(net[1][slot.idx])
                res = fusion(net[1][n+1], net[1][slot.idx])
                tag!(net[1 + slot.idx][1], Tag(:updateX, res)) # communicate change to client node
                unlock(net[1][n+1]); unlock(net[1][slot.idx])
                push!(counter, slot.idx)
                @debug "Fused client $(slot.idx) with piecemaker qubit"
            else
                break
            end
        end

        if length(counter) > 2
            @yield lock(net[1][n+1])
            res = project_traceout!(net[1][n+1], σˣ)
            unlock(net[1][n+1])

            tag!(net[1+counter[1]][1], Tag(:updateZ, res)) # communicate change to latest node
            
            while true # wait for Zdone tag from any client
                @yield onchange_tag(net[1])
                isdonemessage = querydelete!(net[1], :Zdone)
                if !isnothing(isdonemessage)

                    # measure the fidelity to the GHZ state
                    @yield reduce(&, [lock(net[1+i][1]) for i in counter])
                    obs_proj = SProjector(StabilizerState(ghzs[length(counter)])) # GHZ state projector to measure
                    fidelity = real(observable([net[1+i][1] for i in counter], obs_proj; time = now(sim)))

                    @debug "clients serviced: $(counter) --> fidelity: $(fidelity)"
                    foreach(q -> (traceout!(q); unlock(q)), [net[1 + i][1] for i in counter]) # free qubits
                    unlock.(net[1+i][1] for i in counter)

                    timesteps = now(sim)

                    # log results
                    push!(logs, (timesteps, counter, fidelity))
                    break
                end
            end

            counter = Int[] # reset counter only when round completed
            #@yield timeout(sim, 10.0) # operation time of memory qubit
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
                @yield lock(net[1+client][1])
                @debug "X received at client $(client), with value $(value)"
                if value == 2
                    #noisyX = NonInstantGate(X, 0.1) # operation time of memory qubit
                    apply!(net[1+client][1], X)
                end
                unlock(net[1+client][1])
            end
            if !isnothing(msg2)
                @debug "Z received at client $(client)"
                value = msg2[3][2]
                @yield lock(net[1+client][1])
                if value == 2
                    #noisyZ = NonInstantGate(Z, 0.1) # operation time of memory qubit
                    apply!(net[1+client][1], Z)
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
    link_success_prob = 0.0001

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
runtime = 100

sim = prepare_sim(n)
run(sim, runtime) # run for 100000 time units


logs = DataFrame(logs, [:timesteps, :clients_serviced, :GHZfidel])
@info "Final logs: $(logs)"