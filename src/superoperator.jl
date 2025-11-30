using QuantumSavory
using QuantumSavory: Register, X, Z, Y, CNOT, I, ZCZ
using QuantumSavory.ProtocolZoo
using QuantumClifford
using ConcurrentSim
using ResumableFunctions
using Graphs
using NetworkLayout
using DataFrames
using Statistics

function noisy_ghz(target_fidelity::Float64=1.0, n::Int=4)
    perfect_state::StabilizerState = StabilizerState(ghz(n))
    perfect_dm = SProjector(perfect_state)
    mixed_dm = MixedState(SProjector(perfect_state))
    return target_fidelity*perfect_dm + (1-target_fidelity)*mixed_dm
end

n = 4
const perfect_ghz = noisy_ghz(1.0, 4)


# 1. Prepare Choi state (this is what comes out of the system after applying the protocol)
ancilla_register = Register(n, QuantumOpticsRepr())
initialize!([ancilla_register[i] for i in 1:n], noisy_ghz(1.0, 4)) # ancilla hold the GHZ (perfect in this dummy case)

data_register = Register(n, QuantumOpticsRepr())
ref_register = Register(n, QuantumOpticsRepr())
for i in 1:n
    initialize!([data_register[i], ref_register[i]], StabilizerState(ghz(2))) # the data + reference hold 4 Bell pairs
end

for i in 1:n
    apply!([ancilla_register[i], data_register[i]], ZCZ) # CNOT from ancilla to data
end
# apply!(data_register[1], Y)

s_i_outcomes = [project_traceout!(ancilla_register[i], X) - 1 for i in 1:n] # Measure ancilla in X basis (outcomes 1 and 2 in Julia so -1 to get 0 and 1)
append!(vec_s_i_outcomes, [s_i_outcomes])

s = (-1)^ sum(s_i_outcomes)
@info s
registers = RegRef[]
for i in 1:n
    append!(registers, [ref_register[i], data_register[i]])
end

# 2. Prepare general theoretical state in different error patterns
vec_error_patterns = []
vec_s_plus = []
if s == -1
    for prd in Iterators.product([[I,Z,X,Y] for i in 1:n]...)

        Ψ = reduce(⊗, [StabilizerState(ghz(2)) for i in 1:n])

        P =  reduce(⊗, [(I ⊗ Z) for i in 1:n]) # Stabilizer (either XXXX or ZZZZ)
        I_full = reduce(⊗, [I for i in 1:2n])  # Identity on 2n qubits

        # Projector for data qubits onto eigenspaces
        P⁺ = (I_full + P) / sqrt(2)

        P_m = reduce(⊗, [(I ⊗ prd[i]) for i in 1:n]) # Error pattern on data qubits

        Ψ⁺ = P_m * P⁺ * Ψ 
        P_Ψ⁺ = projector(Ψ⁺)

        p_plus = real(observable(registers, P_Ψ⁺))
        append!(vec_s_plus, p_plus)
        append!(vec_error_patterns, [prd])
    end
end
# df = DataFrame(s_i_outcome = vec_s_i_outcomes, p_plus = vec_p_plus, p_minus = vec_p_minus)
# ##

# grouped_stats = combine(groupby(df, :s_i_outcome), 
#     :p_plus => mean => :p_plus,
#     :p_minus => mean => :p_minus,
#     nrow => :count)

# @info "Grouped statistics:"
# @info grouped_stat
