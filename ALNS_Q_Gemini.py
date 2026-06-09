# -*- coding: utf-8 -*-
import os
import random
import time
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# =====================================================================
# UTILITY
# =====================================================================
def compute_arpd(values, best_known):
    """
    Average Relative Percentage Deviation.
    ARPD = mean( (value - best_known) / best_known * 100 )
    Returns 0 when best_known <= 0 to avoid division by zero.
    """
    if best_known <= 0:
        return 0.0
    return float(np.mean([(v - best_known) / best_known * 100.0 for v in values]))

# Global cache for tool sizes to optimize search speed in the knapsack solver
GLOBAL_TOOL_SIZES = {}

# =====================================================================
# Paper-exact parameter settings from Dang et al. (2021)
# =====================================================================
PAPER_B                    = 1
PAPER_POP_SIZE             = 100
PAPER_ELITISM_RATE         = 0.10
PAPER_UNIFORM_MUTATION_PROB = 0.01
PAPER_SWAP_MUTATION_PROB   = 0.01
PAPER_TOURNAMENT_RATE      = 0.20
PAPER_NO_IMPROVEMENT_LIMIT = 20
PAPER_MAX_TIME_SECONDS     = 3600.0
PAPER_SETUP_TIME           = 1.0
PAPER_THETA_M              = 72.0


# =====================================================================
# 1. TOOL REPLACEMENT METHOD  (TRM) — exact enumeration ILP
# =====================================================================
def solve_trm_ilp_exact(tools_in_magazine, tool_sizes, scores, needed_capacity):
    """
    Exact solver for the ILP in Dang et al. (2021) Section 5.6, eqs (23)-(25).

        min   sum_{t in TM_m} sc_t * lambda_t
        s.t.  sum_{t in TM_m} phi_t * lambda_t >= phi_m^S
              lambda_t in {0,1}

    The magazine holds at most ~10 tool sets in practice, so complete
    enumeration over 2^|TM_m| subsets is exact and fast.
    Deterministic tie-breaks ensure reproducibility for a fixed seed.
    """
    if needed_capacity <= 0:
        return []

    tools = sorted(list(tools_in_magazine))
    n = len(tools)
    best_key = None
    best_subset = []

    for mask in range(1, 1 << n):
        subset = [tools[i] for i in range(n) if mask & (1 << i)]
        freed   = sum(tool_sizes[t] for t in subset)
        if freed < needed_capacity:
            continue
        obj = sum(scores.get(t, 0) for t in subset)
        key = (obj, freed, len(subset), tuple(subset))
        if best_key is None or key < best_key:
            best_key   = key
            best_subset = subset

    return best_subset


def solve_trm_knapsack(tools_in_magazine, tool_sizes, scores, needed_capacity):
    return solve_trm_ilp_exact(tools_in_magazine, tool_sizes, scores, needed_capacity)


# =====================================================================
# 2. SOLUTION REPRESENTATION
# =====================================================================
class Individual:
    def __init__(self, job_vector, machine_vector):
        self.job_vector     = list(job_vector)
        self.machine_vector = list(machine_vector)
        self.fitness    = float('inf')
        self.tardiness  = 0.0
        self.setups     = 0

    def get_signature(self):
        return (tuple(self.job_vector), tuple(self.machine_vector))


# =====================================================================
# 3. DECODER  (schedule evaluator + TRM)
# =====================================================================
class Decoder:
    def __init__(self, ops_by_job, num_machines, magazine_capacity, setup_time=1.0):
        self.ops_by_job   = ops_by_job
        self.num_machines = num_machines
        self.C            = magazine_capacity
        self.tau          = setup_time

    def evaluate(self, individual):
        job_vec  = individual.job_vector
        mach_vec = individual.machine_vector
        n        = len(job_vec)

        # --- Pre-populate magazines with the first unique tool on each machine ---
        T_m               = {m: set() for m in range(1, self.num_machines + 1)}
        mach_tool_sequence = {m: [] for m in range(1, self.num_machines + 1)}
        temp_occ = {}
        for g in range(n):
            j_id  = job_vec[g]
            m_id  = mach_vec[g]
            occ   = temp_occ.get(j_id, 0)
            temp_occ[j_id] = occ + 1
            t_ij  = self.ops_by_job[j_id][occ]['tool_set']
            if t_ij not in mach_tool_sequence[m_id]:
                mach_tool_sequence[m_id].append(t_ij)

        for m_id in range(1, self.num_machines + 1):
            cur = 0
            for t_ij in mach_tool_sequence[m_id]:
                phi = GLOBAL_TOOL_SIZES[t_ij]
                if cur + phi <= self.C:
                    T_m[m_id].add(t_ij)
                    cur += phi
                else:
                    break

        # --- Build per-machine succeeding-ops list (for TRM scoring) ---
        succeeding = {m: [] for m in range(1, self.num_machines + 1)}
        temp_occ_2 = {}
        for g in range(n):
            j_id = job_vec[g]
            m_id = mach_vec[g]
            occ  = temp_occ_2.get(j_id, 0)
            temp_occ_2[j_id] = occ + 1
            op   = self.ops_by_job[j_id][occ]
            succeeding[m_id].append((op['tool_set'], op['size']))

        # --- Main scheduling simulation ---
        a_m             = {m: 0.0 for m in range(1, self.num_machines + 1)}
        job_finish_times = {}
        total_tardiness  = 0.0
        total_setups     = 0
        occ_counts       = {}

        for g in range(n):
            j_id = job_vec[g]
            m_id = mach_vec[g]
            occ  = occ_counts.get(j_id, 0)
            occ_counts[j_id] = occ + 1

            op   = self.ops_by_job[j_id][occ]
            t_ij = op['tool_set']
            phi_t = op['size']
            r_ij, p_ij, d_ij = op['r'], op['p'], op['d']

            succeeding[m_id].pop(0)

            z_ijt = 0
            if t_ij not in T_m[m_id]:
                z_ijt = 1
                cur_size  = sum(GLOBAL_TOOL_SIZES[t] for t in T_m[m_id])
                free_space = self.C - cur_size

                if free_space < phi_t:
                    needed_space = phi_t - free_space
                    future_tools = [item[0] for item in succeeding[m_id]]
                    future_unique = []
                    for ft in future_tools:
                        if ft in T_m[m_id] and ft not in future_unique:
                            future_unique.append(ft)

                    scores_map = {}
                    for ft in T_m[m_id]:
                        if ft in future_unique:
                            u = future_unique.index(ft) + 1
                            scores_map[ft] = len(future_unique) - (u - 1)
                        else:
                            scores_map[ft] = 0

                    zero_tools  = [t for t in T_m[m_id] if scores_map[t] == 0]
                    zero_weight = sum(GLOBAL_TOOL_SIZES[t] for t in zero_tools)

                    for t in zero_tools:
                        T_m[m_id].remove(t)

                    if zero_weight < needed_space:
                        remaining_need = needed_space - zero_weight
                        evict = solve_trm_knapsack(
                            list(T_m[m_id]), GLOBAL_TOOL_SIZES, scores_map, remaining_need
                        )
                        for t in evict:
                            T_m[m_id].remove(t)

                T_m[m_id].add(t_ij)
                total_setups += 1

            prev_finish = job_finish_times.get((j_id, occ - 1), 0.0) if occ > 0 else 0.0
            start_time  = max(r_ij, a_m[m_id], prev_finish)
            end_time    = start_time + p_ij + self.tau * z_ijt

            a_m[m_id]                  = end_time
            job_finish_times[(j_id, occ)] = end_time
            total_tardiness += max(0.0, end_time - d_ij)

        individual.tardiness = total_tardiness
        individual.setups    = total_setups
        individual.fitness   = total_tardiness + self.tau * total_setups


# =====================================================================
# 4. PRACTITIONER HEURISTIC  (paper Section 6)
# =====================================================================
class PractitionerHeuristic:
    def __init__(self, jobs_data, num_machines, magazine_capacity,
                 tool_setup_time=PAPER_SETUP_TIME, theta_m=PAPER_THETA_M):
        self.O        = jobs_data
        self.M        = list(range(1, num_machines + 1))
        self.C        = magazine_capacity
        self.tau      = tool_setup_time
        self.theta_m  = theta_m
        self.T_m      = {m: set() for m in self.M}
        self.a_m      = {m: 0.0   for m in self.M}
        self.tool_sizes = {op['tool_set']: op['size'] for op in self.O if 'tool_set' in op}

    def get_magazine_size(self, machine):
        return sum(self.tool_sizes[t] for t in self.T_m[machine])

    def run(self):
        O_hat = sorted(self.O, key=lambda x: x['d'])

        # Phase 1: initial tool allocation
        for op in O_hat:
            t_ij  = op['tool_set']
            phi_t = op['size']
            m_T   = [m for m in self.M if t_ij in self.T_m[m]]
            if not m_T:
                M_C = [m for m in self.M if (self.C - self.get_magazine_size(m)) >= phi_t]
                if M_C:
                    m_star = min(M_C, key=lambda m: (len(self.T_m[m]), m))
                    self.T_m[m_star].add(t_ij)

        # Phase 2: assignment and sequencing
        total_tardiness  = 0
        total_setups     = 0
        job_finish_times = {}
        occ_counts       = {}
        job_vector, machine_vector = [], []

        for op in O_hat:
            job_id = op['job_id']
            t_ij   = op['tool_set']
            phi_t  = op['size']
            r_ij, p_ij, d_ij = op['r'], op['p'], op['d']

            occ = occ_counts.get(job_id, 0)
            occ_counts[job_id] = occ + 1

            m_P = min(self.M, key=lambda m: self.a_m[m])
            m_T_list = [m for m in self.M if t_ij in self.T_m[m]]
            m_T = m_T_list[0] if m_T_list else None

            def calc_xi(machine):
                prev = job_finish_times.get((job_id, occ - 1), 0.0) if occ > 0 else 0.0
                return max(r_ij, self.a_m[machine], prev)

            if m_T is not None:
                if m_T != m_P and (calc_xi(m_T) - calc_xi(m_P)) >= self.theta_m:
                    m_star = m_P
                    z_ijt  = 1
                else:
                    m_star = m_T
                    z_ijt  = 0
            else:
                m_star = m_P
                z_ijt  = 1

            if z_ijt == 1:
                phi_s = phi_t - (self.C - self.get_magazine_size(m_star))
                while phi_s > 0 and self.T_m[m_star]:
                    removed = random.choice(sorted(self.T_m[m_star]))
                    self.T_m[m_star].remove(removed)
                    phi_s = phi_t - (self.C - self.get_magazine_size(m_star))
                self.T_m[m_star].add(t_ij)
                total_setups += 1

            start_time = calc_xi(m_star)
            end_time   = start_time + p_ij + self.tau * z_ijt
            self.a_m[m_star] = end_time
            job_finish_times[(job_id, occ)] = end_time
            total_tardiness += max(0.0, end_time - d_ij)

            job_vector.append(job_id)
            machine_vector.append(m_star)

        ind         = Individual(job_vector, machine_vector)
        ind.fitness = total_tardiness + self.tau * total_setups
        return ind


# =====================================================================
# 5. Q-LEARNING OPERATOR SELECTOR
# =====================================================================
class QLearningSelector:
    """
    Tabular Q-learning for adaptive operator selection in ALNS.

    State  : discretised solution quality bucket (0 = best, N_STATES-1 = worst)
             computed as floor(normalised_gap * N_STATES).
             The normalised gap is (current - best) / max(1, best).

    Action : (destroy_index, repair_index) pair — one Q-table entry per
             (state, destroy, repair) triple.

    Reward : improvement in objective value achieved by the operator pair.
             r = current_fitness - candidate_fitness   (positive = improvement)
             Clipped to [-max_reward, max_reward] to keep Q-values bounded.

    Update : Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

    Exploration : epsilon-greedy with linear decay from epsilon_start to
                  epsilon_end over the first half of the run budget.
    """

    N_STATES    = 5      # number of solution-quality buckets
    MAX_REWARD  = 50.0   # clip reward to [-MAX_REWARD, +MAX_REWARD]

    def __init__(
        self,
        n_destroy,
        n_repair,
        alpha       = 0.20,   # learning rate
        gamma       = 0.90,   # discount factor
        epsilon_start = 0.40, # initial exploration rate
        epsilon_end   = 0.05, # final exploration rate
        epsilon_decay_steps = None,   # set to max_iterations / 2 at run time
    ):
        self.n_destroy = n_destroy
        self.n_repair  = n_repair
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Q-table shape: (N_STATES, n_destroy, n_repair)
        self.Q = np.zeros((self.N_STATES, n_destroy, n_repair), dtype=float)

        # Tracking
        self.visit_counts = np.zeros_like(self.Q, dtype=int)
        self._step = 0

    # ------------------------------------------------------------------
    def _state(self, current_fitness, best_fitness):
        """Map solution quality to a discrete state index."""
        gap = (current_fitness - best_fitness) / max(1.0, abs(best_fitness))
        gap = max(0.0, gap)              # negative gap = new best (state 0)
        s   = int(gap * self.N_STATES)
        return min(s, self.N_STATES - 1)

    def select(self, current_fitness, best_fitness):
        """Epsilon-greedy action selection. Returns (destroy_idx, repair_idx)."""
        s = self._state(current_fitness, best_fitness)
        if random.random() < self.epsilon:
            # Explore
            d = random.randrange(self.n_destroy)
            r = random.randrange(self.n_repair)
        else:
            # Exploit: pick argmax over (destroy, repair) pairs
            flat_idx = int(np.argmax(self.Q[s]))
            d, r     = divmod(flat_idx, self.n_repair)
        return d, r, s

    def update(self, s, d, r_idx, reward, next_fitness, best_fitness):
        """Standard Q-learning update."""
        reward   = float(np.clip(reward, -self.MAX_REWARD, self.MAX_REWARD))
        s_next   = self._state(next_fitness, best_fitness)
        q_old    = self.Q[s, d, r_idx]
        q_next   = float(np.max(self.Q[s_next]))
        td_error = reward + self.gamma * q_next - q_old
        self.Q[s, d, r_idx] = q_old + self.alpha * td_error
        self.visit_counts[s, d, r_idx] += 1
        self._step += 1

    def decay_epsilon(self, total_steps):
        """Linear decay; called once per iteration."""
        if self.epsilon_decay_steps is None:
            return
        progress = min(1.0, self._step / max(1, self.epsilon_decay_steps))
        self.epsilon = (
            self.epsilon_start
            + progress * (self.epsilon_end - self.epsilon_start)
        )

    def q_table_summary(self):
        """Return a DataFrame summarising the learned Q-values per (destroy, repair)."""
        rows = []
        for d in range(self.n_destroy):
            for r in range(self.n_repair):
                rows.append({
                    "destroy_idx":  d,
                    "repair_idx":   r,
                    "Q_mean_states": round(float(np.mean(self.Q[:, d, r])), 4),
                    "Q_max_states":  round(float(np.max(self.Q[:, d, r])),  4),
                    "visits":        int(self.visit_counts[:, d, r].sum()),
                })
        return pd.DataFrame(rows)


# =====================================================================
# 6. ALNS WITH Q-LEARNING OPERATOR SELECTION  (ALNS-QL)
# =====================================================================
class ALNS_QL:
    """
    Adaptive Large Neighborhood Search with Q-Learning operator selection.

    Differences from the original ALNS-AOS version:
    -  Operator selection uses a Q-learning agent (QLearningSelector) instead
       of the roulette-wheel AOS weight update.
    -  The QL agent observes the solution quality state, picks a
       (destroy, repair) pair, executes it, receives a reward proportional
       to the fitness improvement, and updates its Q-table.
    -  Simulated-annealing acceptance criterion is unchanged.
    -  All destroy/repair operators are identical to the AOS version so that
       any performance difference is attributable solely to the selection
       mechanism.

    Q-learning hyper-parameters:
        alpha  = 0.20   learning rate
        gamma  = 0.90   discount factor
        eps    = 0.40 → 0.05   epsilon-greedy exploration, linear decay
    """

    def __init__(
        self,
        jobs_data,
        num_machines,
        magazine_capacity,
        setup_time      = PAPER_SETUP_TIME,
        destroy_fraction = (0.03, 0.08),  # OPTIMIZED: Much faster local search neighborhood
        start_temperature = None,
        cooling_rate    = 0.995,
        min_temperature = 1e-6,
        max_insert_positions = 12,
        max_machine_candidates = 3,
        cache_evaluations = True,
        # Q-learning hyper-parameters
        ql_alpha          = 0.20,
        ql_gamma          = 0.90,
        ql_epsilon_start  = 0.40,
        ql_epsilon_end    = 0.05,
    ):
        self.jobs_data    = jobs_data
        self.num_machines = num_machines
        self.C            = magazine_capacity
        self.tau          = setup_time
        self.destroy_fraction        = destroy_fraction
        self.temperature             = start_temperature
        self.cooling_rate            = cooling_rate
        self.min_temperature         = min_temperature
        self.max_insert_positions    = max_insert_positions
        self.max_machine_candidates  = max_machine_candidates
        self.cache_evaluations       = cache_evaluations
        self.eval_cache              = {}
        self.eval_cache_hits         = 0
        self.eval_cache_misses       = 0

        # Build ops_by_job lookup
        self.ops_by_job = {}
        self.flat_ops   = []
        for op in jobs_data:
            job_id = int(op['job_id'])
            self.ops_by_job.setdefault(job_id, []).append(op)
            self.flat_ops.append(job_id)
        for job_id in self.ops_by_job:
            self.ops_by_job[job_id].sort(key=lambda x: x['op_id'])

        self.decoder = Decoder(self.ops_by_job, self.num_machines, self.C, self.tau)

        # Register operators (order must remain stable — indices are Q-table keys)
        self.destroy_names = [
            "random_removal",
            "worst_due_date_removal",
            "machine_overload_removal",
            "setup_related_removal",
        ]
        self.repair_names = [
            "greedy_best_insert",
            "regret2_insert",
            "edd_insert",
            "least_loaded_insert",
        ]
        self.destroy_fns = [
            self.destroy_random_removal,
            self.destroy_worst_due_date_removal,
            self.destroy_machine_overload_removal,
            self.destroy_setup_related_removal,
        ]
        self.repair_fns = [
            self.repair_greedy_best_insert,
            self.repair_regret2_insert,
            self.repair_edd_insert,
            self.repair_least_loaded_insert,
        ]

        # Q-learning agent (decay steps set to max_iterations/2 at run time)
        self.ql = QLearningSelector(
            n_destroy       = len(self.destroy_names),
            n_repair        = len(self.repair_names),
            alpha           = ql_alpha,
            gamma           = ql_gamma,
            epsilon_start   = ql_epsilon_start,
            epsilon_end     = ql_epsilon_end,
        )

        # Store hyper-parameters for reporting
        self.ql_alpha         = ql_alpha
        self.ql_gamma         = ql_gamma
        self.ql_epsilon_start = ql_epsilon_start
        self.ql_epsilon_end   = ql_epsilon_end

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def clone(self, ind):
        new = Individual(ind.job_vector, ind.machine_vector)
        new.fitness   = ind.fitness
        new.tardiness = ind.tardiness
        new.setups    = ind.setups
        for attr in ["runtime", "generations", "stop_reason", "history"]:
            if hasattr(ind, attr):
                setattr(new, attr, getattr(ind, attr))
        return new

    def _evaluate(self, job_vec, mach_vec):
        key = (tuple(job_vec), tuple(mach_vec))
        if self.cache_evaluations and key in self.eval_cache:
            self.eval_cache_hits += 1
            return self.clone(self.eval_cache[key])
        ind = Individual(job_vec, mach_vec)
        self.decoder.evaluate(ind)
        if self.cache_evaluations:
            self.eval_cache_misses += 1
            self.eval_cache[key] = self.clone(ind)
        return ind

    def _accept(self, candidate, current):
        if candidate.fitness <= current.fitness:
            return True
        temp = max(self.min_temperature, self.temperature)
        return random.random() < np.exp(-(candidate.fitness - current.fitness) / temp)

    def _num_to_remove(self, n):
        lo, hi = self.destroy_fraction
        frac   = random.uniform(lo, hi)
        # OPTIMIZED: Cap removed jobs to 8 max to speed up repair phases
        return max(1, min(n - 1, 8, int(round(frac * n))))

    def _remove_positions(self, individual, positions):
        positions    = sorted(set(positions))
        removed_jobs = [individual.job_vector[i] for i in positions]
        partial_jobs = [v for i, v in enumerate(individual.job_vector)  if i not in positions]
        partial_machs= [v for i, v in enumerate(individual.machine_vector) if i not in positions]
        return partial_jobs, partial_machs, removed_jobs

    def _operation_data_for_insertion(self, partial_jobs, job_id):
        occ = partial_jobs.count(job_id)
        occ = min(occ, len(self.ops_by_job[job_id]) - 1)
        return self.ops_by_job[job_id][occ]

    def _position_candidates(self, partial_jobs, job_id):
        n = len(partial_jobs)
        if n + 1 <= self.max_insert_positions:
            return list(range(n + 1))
        op  = self._operation_data_for_insertion(partial_jobs, job_id)
        due = float(op['d'])
        occ = {}
        edd_pos = n
        for idx, existing_job in enumerate(partial_jobs):
            k = occ.get(existing_job, 0)
            occ[existing_job] = k + 1
            if float(self.ops_by_job[existing_job][k]['d']) > due:
                edd_pos = idx
                break
        positions = {0, n, edd_pos}
        for delta in [-3, -2, -1, 1, 2, 3]:
            pos = edd_pos + delta
            if 0 <= pos <= n:
                positions.add(pos)
        remaining = [i for i in range(n + 1) if i not in positions]
        budget    = max(0, self.max_insert_positions - len(positions))
        if remaining and budget > 0:
            positions.update(random.sample(remaining, min(budget, len(remaining))))
        return sorted(positions)

    def _machine_candidates(self, partial_jobs, partial_machs, job_id):
        op   = self._operation_data_for_insertion(partial_jobs, job_id)
        tool = op['tool_set']
        loads         = {m: 0.0 for m in range(1, self.num_machines + 1)}
        tool_presence = {m: 0   for m in range(1, self.num_machines + 1)}
        occ = {}
        for job, mach in zip(partial_jobs, partial_machs):
            k = occ.get(job, 0)
            occ[job] = k + 1
            eop = self.ops_by_job[job][k]
            loads[mach] += float(eop['p'])
            if eop['tool_set'] == tool:
                tool_presence[mach] += 1
        candidates = []
        same_tool = [m for m, cnt in tool_presence.items() if cnt > 0]
        if same_tool:
            candidates.append(max(same_tool, key=lambda m: (tool_presence[m], -loads[m])))
        for m, _ in sorted(loads.items(), key=lambda kv: kv[1]):
            if m not in candidates:
                candidates.append(m)
            if len(candidates) >= self.max_machine_candidates:
                break
        return candidates

    # ------------------------------------------------------------------
    # Destroy operators
    # ------------------------------------------------------------------
    def destroy_random_removal(self, individual, q):
        positions = random.sample(range(len(individual.job_vector)), q)
        return self._remove_positions(individual, positions)

    def destroy_worst_due_date_removal(self, individual, q):
        occ    = {}
        scored = []
        for idx, job in enumerate(individual.job_vector):
            k = occ.get(job, 0)
            occ[job] = k + 1
            op    = self.ops_by_job[job][k]
            score = (-float(op['d']), float(op['p']), random.random())
            scored.append((score, idx))
        positions = [idx for _, idx in sorted(scored, reverse=True)[:q]]
        return self._remove_positions(individual, positions)

    def destroy_machine_overload_removal(self, individual, q):
        loads = {m: 0.0 for m in range(1, self.num_machines + 1)}
        occ   = {}
        for job, mach in zip(individual.job_vector, individual.machine_vector):
            k = occ.get(job, 0)
            occ[job] = k + 1
            loads[mach] += float(self.ops_by_job[job][k]['p'])
        overloaded = max(loads, key=loads.get)
        cand_pos   = [i for i, m in enumerate(individual.machine_vector) if m == overloaded]
        if len(cand_pos) < q:
            extra    = [i for i in range(len(individual.job_vector)) if i not in cand_pos]
            cand_pos += random.sample(extra, min(len(extra), q - len(cand_pos)))
        positions = random.sample(cand_pos, q)
        return self._remove_positions(individual, positions)

    def destroy_setup_related_removal(self, individual, q):
        occ   = {}
        tools = []
        for job in individual.job_vector:
            k = occ.get(job, 0)
            occ[job] = k + 1
            tools.append(self.ops_by_job[job][k]['tool_set'])
        setup_pos = []
        last_tool = {}
        for idx, (mach, tool) in enumerate(zip(individual.machine_vector, tools)):
            if mach in last_tool and last_tool[mach] != tool:
                setup_pos.append(idx)
            last_tool[mach] = tool
        if len(setup_pos) < q:
            rest      = [i for i in range(len(individual.job_vector)) if i not in setup_pos]
            setup_pos += random.sample(rest, min(len(rest), q - len(setup_pos)))
        positions = random.sample(setup_pos, q)
        return self._remove_positions(individual, positions)

    # ------------------------------------------------------------------
    # Repair operators
    # ------------------------------------------------------------------
    def _best_single_insertion(self, partial_jobs, partial_machs, job_id,
                                machine_candidates=None):
        if machine_candidates is None:
            machine_candidates = self._machine_candidates(partial_jobs, partial_machs, job_id)
        best = None
        for pos in self._position_candidates(partial_jobs, job_id):
            for mach in machine_candidates:
                tj = partial_jobs[:pos]  + [job_id] + partial_jobs[pos:]
                tm = partial_machs[:pos] + [mach]   + partial_machs[pos:]
                cand = self._evaluate(tj, tm)
                key  = (cand.fitness, cand.tardiness, cand.setups, pos, mach)
                if best is None or key < best[0]:
                    best = (key, cand)
        return best[1]

    def repair_greedy_best_insert(self, partial_jobs, partial_machs, removed_jobs):
        jobs = list(removed_jobs)
        random.shuffle(jobs)
        cj, cm = list(partial_jobs), list(partial_machs)
        for job_id in jobs:
            best = self._best_single_insertion(cj, cm, job_id)
            cj, cm = best.job_vector, best.machine_vector
        return self._evaluate(cj, cm)

    def repair_regret2_insert(self, partial_jobs, partial_machs, removed_jobs):
        remaining = list(removed_jobs)
        cj, cm    = list(partial_jobs), list(partial_machs)
        while remaining:
            best_choice = None
            for job_id in remaining:
                candidates = []
                for pos in self._position_candidates(cj, job_id):
                    for mach in self._machine_candidates(cj, cm, job_id):
                        tj   = cj[:pos]  + [job_id] + cj[pos:]
                        tm   = cm[:pos]  + [mach]   + cm[pos:]
                        cand = self._evaluate(tj, tm)
                        candidates.append((cand.fitness, cand))
                candidates.sort(key=lambda x: x[0])
                best_fit   = candidates[0][0]
                second_fit = candidates[1][0] if len(candidates) > 1 else best_fit
                regret     = second_fit - best_fit
                # Tiebreak: prefer job whose best insertion is cheapest
                choice_key = (regret, best_fit, random.random())
                if best_choice is None or choice_key > best_choice[0]:
                    best_choice = (choice_key, job_id, candidates[0][1])
            _, chosen_job, chosen_ind = best_choice
            cj, cm = chosen_ind.job_vector, chosen_ind.machine_vector
            remaining.remove(chosen_job)
        return self._evaluate(cj, cm)

    def repair_edd_insert(self, partial_jobs, partial_machs, removed_jobs):
        jobs = sorted(removed_jobs,
                      key=lambda j: self._operation_data_for_insertion(partial_jobs, j)['d'])
        cj, cm = list(partial_jobs), list(partial_machs)
        for job_id in jobs:
            op  = self._operation_data_for_insertion(cj, job_id)
            occ = {}
            pos = len(cj)
            for idx, existing_job in enumerate(cj):
                k = occ.get(existing_job, 0)
                occ[existing_job] = k + 1
                if self.ops_by_job[existing_job][k]['d'] > op['d']:
                    pos = idx
                    break
            best = None
            for mach in range(1, self.num_machines + 1):
                tj   = cj[:pos]  + [job_id] + cj[pos:]
                tm   = cm[:pos]  + [mach]   + cm[pos:]
                cand = self._evaluate(tj, tm)
                key  = (cand.fitness, cand.tardiness, cand.setups, mach)
                if best is None or key < best[0]:
                    best = (key, cand)
            cj, cm = best[1].job_vector, best[1].machine_vector
        return self._evaluate(cj, cm)

    def repair_least_loaded_insert(self, partial_jobs, partial_machs, removed_jobs):
        cj, cm = list(partial_jobs), list(partial_machs)
        for job_id in removed_jobs:
            loads = {m: 0.0 for m in range(1, self.num_machines + 1)}
            occ   = {}
            for job, mach in zip(cj, cm):
                k = occ.get(job, 0)
                occ[job] = k + 1
                loads[mach] += float(self.ops_by_job[job][k]['p'])
            ll   = min(loads, key=loads.get)
            best = self._best_single_insertion(cj, cm, job_id, [ll])
            cj, cm = best.job_vector, best.machine_vector
        return self._evaluate(cj, cm)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        initial_solution,
        max_iterations        = 2000,
        max_time_seconds      = 60.0,
        no_improvement_limit  = 300,
        record_history        = True,
        verbose               = False,
    ):
        start_clock = time.time()
        current = self.clone(initial_solution)
        self.decoder.evaluate(current)
        best = self.clone(current)

        # SA temperature initialisation
        if self.temperature is None:
            self.temperature = max(1.0, 0.50 * abs(current.fitness))

        # OPTIMIZED: Cap epsilon decay steps to 150 to ensure exploration/exploitation transition occurs
        self.ql.epsilon_decay_steps = min(max_iterations // 2, 150)

        history     = []
        no_improve  = 0
        stop_reason = "iteration_limit"

        for it in range(1, max_iterations + 1):
            elapsed = time.time() - start_clock
            if elapsed >= max_time_seconds:
                stop_reason = "time_limit"
                break
            if no_improve >= no_improvement_limit:
                stop_reason = "no_improvement_limit"
                break

            # --- Q-learning: select operators ---
            d_idx, r_idx, state = self.ql.select(current.fitness, best.fitness)
            destroy_name = self.destroy_names[d_idx]
            repair_name  = self.repair_names[r_idx]

            q = self._num_to_remove(len(current.job_vector))
            partial_jobs, partial_machs, removed_jobs = \
                self.destroy_fns[d_idx](current, q)
            candidate = self.repair_fns[r_idx](partial_jobs, partial_machs, removed_jobs)

            # Sla de fitness op van de huidige oplossing vóór de eventuele acceptatie
            previous_fitness = current.fitness

            # --- SA acceptance ---
            accepted     = self._accept(candidate, current)
            improved_best = candidate.fitness < best.fitness

            if accepted:
                current = candidate

            if improved_best:
                best       = self.clone(candidate)
                no_improve = 0
            else:
                no_improve += 1

            # --- Q-learning: compute reward and update ---
            # OPTIMIZED: Relative rewards based on percentage improvement
            reward = ((previous_fitness - candidate.fitness) / max(1.0, previous_fitness)) * 100.0
            
            self.ql.update(state, d_idx, r_idx, reward, current.fitness, best.fitness)
            self.ql.decay_epsilon(max_iterations)

            # SA cooling
            self.temperature = max(self.min_temperature,
                                   self.temperature * self.cooling_rate)

            if record_history:
                history.append({
                    "iteration":       it,
                    "runtime":         float(time.time() - start_clock),
                    "best_fitness":    float(best.fitness),
                    "current_fitness": float(current.fitness),
                    "candidate_fitness": float(candidate.fitness),
                    "accepted":        bool(accepted),
                    "improved_best":   bool(improved_best),
                    "destroy":         destroy_name,
                    "repair":          repair_name,
                    "q_removed":       int(q),
                    "temperature":     float(self.temperature),
                    "no_improve":      int(no_improve),
                    "ql_epsilon":      float(self.ql.epsilon),
                    "ql_state":        int(state),
                    "ql_reward":       float(reward),
                    "eval_cache_hits":   int(self.eval_cache_hits),
                    "eval_cache_misses": int(self.eval_cache_misses),
                })

            if verbose and (it == 1 or it % 50 == 0 or improved_best):
                print(
                    f"it={it:5d} best={best.fitness:.4f} cur={current.fitness:.4f} "
                    f"cand={candidate.fitness:.4f} acc={accepted} "
                    f"d={destroy_name[:6]} r={repair_name[:6]} "
                    f"eps={self.ql.epsilon:.3f} T={self.temperature:.3f}"
                )

        best.alns_iterations   = len(history) if record_history else it
        best.alns_runtime      = time.time() - start_clock
        best.alns_stop_reason  = stop_reason
        best.alns_history      = history
        best.ql_table_summary  = self.ql.q_table_summary()
        best.ql_visit_counts   = self.ql.visit_counts.copy()
        best.alns_eval_cache_hits   = self.eval_cache_hits
        best.alns_eval_cache_misses = self.eval_cache_misses
        return best


# =====================================================================
# 7. DATA LOADER
# =====================================================================
def load_actual_kmwe_instance(filepath):
    """
    Parse a real KMWE CSV file.
    Header format (first 5 lines): key,value rows including M and C.
    Operation data starts at row 6 with columns:
        job_id, op_id, r, p, d, tool_set, size
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"KMWE CSV not found: {filepath}")

    num_machines      = 2
    magazine_capacity = 80

    with open(filepath, 'r') as f:
        for _ in range(5):
            line  = f.readline().strip()
            parts = line.split(',')
            if len(parts) >= 2:
                key, val = parts[0].strip(), parts[1].strip()
                if key == 'M':
                    num_machines = int(val)
                elif key == 'C':
                    magazine_capacity = int(val)

    df = pd.read_csv(filepath, skiprows=5)
    expected = ['job_id', 'op_id', 'r', 'p', 'd', 'tool_set', 'size']
    if len(df.columns) != len(expected):
        raise ValueError(f"Unexpected column count in {filepath}")
    df.columns = expected
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='raise')

    jobs_data = df.to_dict(orient='records')

    GLOBAL_TOOL_SIZES.clear()
    for op in jobs_data:
        GLOBAL_TOOL_SIZES[op['tool_set']] = op['size']

    return jobs_data, num_machines, magazine_capacity


def resolve_kmwe_case_file(case_name):
    for path in [
        os.path.join(case_name, f"{case_name}.csv"),
        os.path.join(case_name, f"Base {case_name}.csv"),
        f"{case_name}.csv",
    ]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"KMWE CSV for {case_name!r} not found.")


# =====================================================================
# 8. SINGLE-RUN HELPER
# =====================================================================
def run_alns_ql_on_file(
    case_file,
    seed                  = 0,
    alns_time_seconds     = 60.0,
    alns_iterations       = 2000,
    alns_no_improvement_limit = 300,
    verbose               = False,
):
    """
    Run PH → ALNS-QL on one KMWE file. Returns (alns_solution, result_dict).
    """
    random.seed(seed)
    np.random.seed(seed)

    jobs_data, m_case, c_case = load_actual_kmwe_instance(case_file)

    t0 = time.time()
    ph_engine   = PractitionerHeuristic(jobs_data, m_case, c_case)
    ph_solution = ph_engine.run()
    ph_runtime  = time.time() - t0

    # Re-evaluate PH solution through Decoder for TRP-consistent fitness
    ops_by_job = {}
    for op in jobs_data:
        ops_by_job.setdefault(int(op['job_id']), []).append(op)
    for jid in ops_by_job:
        ops_by_job[jid].sort(key=lambda x: x['op_id'])
    Decoder(ops_by_job, m_case, c_case, PAPER_SETUP_TIME).evaluate(ph_solution)

    alns_engine   = ALNS_QL(jobs_data, m_case, c_case, PAPER_SETUP_TIME)
    alns_solution = alns_engine.run(
        ph_solution,
        max_time_seconds     = alns_time_seconds,
        max_iterations       = alns_iterations,
        no_improvement_limit = alns_no_improvement_limit,
        record_history       = True,
        verbose              = verbose,
    )
    # Final re-evaluation to ensure fitness is consistent
    Decoder(ops_by_job, m_case, c_case, PAPER_SETUP_TIME).evaluate(alns_solution)

    result = {
        "case_file":             case_file,
        "seed":                  seed,
        "PH_fitness":            ph_solution.fitness,
        "PH_tardiness":          ph_solution.tardiness,
        "PH_setups":             ph_solution.setups,
        "PH_runtime":            ph_runtime,
        "ALNS_fitness":          alns_solution.fitness,
        "ALNS_tardiness":        alns_solution.tardiness,
        "ALNS_setups":           alns_solution.setups,
        "ALNS_runtime":          alns_solution.alns_runtime,
        "ALNS_iterations":       alns_solution.alns_iterations,
        "ALNS_stop":             alns_solution.alns_stop_reason,
        "ALNS_cache_hits":       alns_solution.alns_eval_cache_hits,
        "ALNS_cache_misses":     alns_solution.alns_eval_cache_misses,
        "Improvement_vs_PH_%":  ((alns_solution.fitness - ph_solution.fitness)
                                  / max(1.0, ph_solution.fitness)) * 100.0,
    }
    return alns_solution, result


# =====================================================================
# 9. TABLE 8 EXPERIMENT  (6M140 slices)
# =====================================================================
def run_alns_ql_table8_replications(
    num_runs              = 10,
    alns_time_seconds     = 300.0,            # OPTIMIZED: Extended time limit for complete exploration
    alns_iterations       = 1000,            # OPTIMIZED: Balanced iteration budget
    alns_no_improvement_limit = 200,          # OPTIMIZED: Balanced stagnation check
):
    """
    Table 8-style experiment: 6M140 sliced at n = 15,25,30,60,90,120,140.
    Reports PH and ALNS-QL means, std, ARPD, compute time, and Wilcoxon test.
    """
    print("\n" + "=" * 130)
    print(f" TABLE 8  —  ALNS-QL vs PH  on 6M140 slices  ({num_runs} seeds) ".center(130, "#"))
    print("=" * 130)

    case_file = resolve_kmwe_case_file("6M140")
    full_data, m_val, c_val = load_actual_kmwe_instance(case_file)
    df_sorted = pd.DataFrame(full_data).sort_values(by="r")

    rows       = []
    all_records = []

    for n_slice in [15, 25, 30, 60, 90, 120, 140]:
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        records    = []

        for seed in tqdm(range(num_runs), desc=f"n={n_slice:3d}", unit="run"):
            random.seed(seed)
            np.random.seed(seed)

            GLOBAL_TOOL_SIZES.clear()
            for op in sliced_ops:
                GLOBAL_TOOL_SIZES[op['tool_set']] = op['size']

            t0 = time.time()
            ph_engine   = PractitionerHeuristic(sliced_ops, m_val, c_val)
            ph_solution = ph_engine.run()
            ph_runtime  = time.time() - t0

            ops_by_job = {}
            for op in sliced_ops:
                ops_by_job.setdefault(int(op['job_id']), []).append(op)
            for jid in ops_by_job:
                ops_by_job[jid].sort(key=lambda x: x['op_id'])
            Decoder(ops_by_job, m_val, c_val, PAPER_SETUP_TIME).evaluate(ph_solution)

            alns_engine   = ALNS_QL(sliced_ops, m_val, c_val, PAPER_SETUP_TIME)
            alns_solution = alns_engine.run(
                ph_solution,
                max_time_seconds     = alns_time_seconds,
                max_iterations       = alns_iterations,
                no_improvement_limit = alns_no_improvement_limit,
                record_history       = True,
                verbose              = False,
            )
            Decoder(ops_by_job, m_val, c_val, PAPER_SETUP_TIME).evaluate(alns_solution)

            records.append({
                "PH_fitness":    ph_solution.fitness,
                "PH_runtime":    ph_runtime,
                "ALNS_fitness":  alns_solution.fitness,
                "ALNS_runtime":  alns_solution.alns_runtime,
                "ALNS_iterations": alns_solution.alns_iterations,
                "ALNS_stop":     alns_solution.alns_stop_reason,
            })

        all_records.append(records)
        ph   = np.array([r["PH_fitness"]   for r in records], dtype=float)
        alns = np.array([r["ALNS_fitness"]  for r in records], dtype=float)
        best_known = float(min(np.min(ph), np.min(alns)))

        rows.append({
            "n":              n_slice,
            "PH_μ":           round(float(np.mean(ph)),   2),
            "PH_σ":           round(float(np.std(ph)),    2),
            "PH_ARPD":        round(compute_arpd(ph, best_known), 2),
            "PH_C.T.(s)":     round(float(np.mean([r["PH_runtime"]   for r in records])), 3),
            "ALNS_μ":         round(float(np.mean(alns)), 2),
            "ALNS_σ":         round(float(np.std(alns)),  2),
            "ALNS_ARPD":      round(compute_arpd(alns, best_known), 2),
            "ALNS_C.T.(s)":   round(float(np.mean([r["ALNS_runtime"] for r in records])), 3),
            "ALNS_it_μ":      round(float(np.mean([r["ALNS_iterations"] for r in records])), 1),
            "Gap_vs_PH (%)":  f"{((np.mean(alns)-np.mean(ph))/max(1.0,np.mean(ph)))*100:.2f}%",
            "StopReasons":    ",".join(sorted(set(r["ALNS_stop"] for r in records))),
        })

    summary = pd.DataFrame(rows)
    print("\n[TABLE 8 SUMMARY]\n")
    print(summary.to_string(index=False))

    # --- Wilcoxon signed-rank tests ---
    print("\n[STATISTICAL TESTS — Wilcoxon signed-rank, ALNS-QL vs PH]\n")
    for row_data, records in zip(rows, all_records):
        ph_vals   = np.array([r["PH_fitness"]   for r in records])
        alns_vals = np.array([r["ALNS_fitness"]  for r in records])
        if np.all(ph_vals == alns_vals):
            print(f"  n={row_data['n']:3d}: all values identical, test skipped.")
            continue
        stat, p = stats.wilcoxon(ph_vals, alns_vals, alternative='greater')
        sig = "YES ***" if p < 0.05 else ("marginal" if p < 0.10 else "NO")
        print(f"  n={row_data['n']:3d}: W={stat:.1f}  p={p:.4f}  significant={sig}")

    return summary


# =====================================================================
# 10. TABLE 14 EXPERIMENT  (base cases)
# =====================================================================
def run_alns_ql_base_case_replications(
    num_runs              = 10,
    alns_time_seconds     = 300.0,            # OPTIMIZED: Extended time limit for complete exploration
    alns_iterations       = 1000,            # OPTIMIZED: Balanced iteration budget
    alns_no_improvement_limit = 200,          # OPTIMIZED: Balanced stagnation check
):
    """
    Table 14-style experiment: all four KMWE base cases.
    Reports PH and ALNS-QL means, std, ARPD, compute time, and Wilcoxon test.
    Also prints the converged Q-table (average over runs) for each case.
    """
    print("\n" + "=" * 130)
    print(f" TABLE 14  —  ALNS-QL vs PH  on base cases  ({num_runs} seeds) ".center(130, "#"))
    print("=" * 130)

    case_names  = ["2M38", "2M46", "6M140", "6M163"]
    rows        = []
    all_records = []
    # Store last Q-table per case for reporting
    ql_summaries = {}

    for case_name in case_names:
        case_file = resolve_kmwe_case_file(case_name)
        records   = []
        last_ql_summary = None

        for seed in tqdm(range(num_runs), desc=f"Running {case_name}", leave=False):
            alns_sol, result = run_alns_ql_on_file(
                case_file,
                seed                  = seed,
                alns_time_seconds     = alns_time_seconds,
                alns_iterations       = alns_iterations,
                alns_no_improvement_limit = alns_no_improvement_limit,
                verbose               = False,
            )
            records.append(result)
            last_ql_summary = alns_sol.ql_table_summary

        all_records.append(records)
        ql_summaries[case_name] = last_ql_summary

        ph   = np.array([r["PH_fitness"]  for r in records], dtype=float)
        alns = np.array([r["ALNS_fitness"] for r in records], dtype=float)
        best_known = float(min(np.min(ph), np.min(alns)))

        rows.append({
            "BaseCase":       case_name,
            "PH_μ":           round(float(np.mean(ph)),   2),
            "PH_σ":           round(float(np.std(ph)),    2),
            "PH_ARPD":        round(compute_arpd(ph, best_known), 2),
            "PH_C.T.(s)":     round(float(np.mean([r["PH_runtime"]   for r in records])), 3),
            "ALNS_μ":         round(float(np.mean(alns)), 2),
            "ALNS_σ":         round(float(np.std(alns)),  2),
            "ALNS_ARPD":      round(compute_arpd(alns, best_known), 2),
            "ALNS_C.T.(s)":   round(float(np.mean([r["ALNS_runtime"] for r in records])), 3),
            "ALNS_it_μ":      round(float(np.mean([r["ALNS_iterations"] for r in records])), 1),
            "Gap_vs_PH (%)":  f"{((np.mean(alns)-np.mean(ph))/max(1.0,np.mean(ph)))*100:.2f}%",
            "StopReasons":    ",".join(sorted(set(r["ALNS_stop"] for r in records))),
        })

    summary = pd.DataFrame(rows)
    print("\n[TABLE 14 SUMMARY]\n")
    print(summary.to_string(index=False))

    # --- Wilcoxon tests ---
    print("\n[STATISTICAL TESTS — Wilcoxon signed-rank, ALNS-QL vs PH]\n")
    for row_data, records in zip(rows, all_records):
        ph_vals   = np.array([r["PH_fitness"]   for r in records])
        alns_vals = np.array([r["ALNS_fitness"]  for r in records])
        if np.all(ph_vals == alns_vals):
            print(f"  {row_data['BaseCase']}: all values identical, test skipped.")
            continue
        stat, p = stats.wilcoxon(ph_vals, alns_vals, alternative='greater')
        sig = "YES ***" if p < 0.05 else ("marginal" if p < 0.10 else "NO")
        print(f"  {row_data['BaseCase']}: W={stat:.1f}  p={p:.4f}  significant={sig}")

    # --- Q-table summaries (last seed per case) ---
    print("\n[Q-TABLE SUMMARIES  (operator index → name mapping below)]\n")
    destroy_labels = [
        "0:random_removal",
        "1:worst_due_date",
        "2:machine_overload",
        "3:setup_related",
    ]
    repair_labels = [
        "0:greedy_best_insert",
        "1:regret2_insert",
        "2:edd_insert",
        "3:least_loaded_insert",
    ]
    print("Destroy operators:", destroy_labels)
    print("Repair  operators:", repair_labels)
    for case_name, ql_df in ql_summaries.items():
        print(f"\n  {case_name}:")
        print(ql_df.to_string(index=False))

    return summary


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    run_alns_ql_table8_replications(num_runs=10)
    run_alns_ql_base_case_replications(num_runs=10)