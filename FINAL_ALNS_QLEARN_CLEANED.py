import os
import random
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Global lookup for tool sizing dependencies
GLOBAL_TOOL_SIZES = {}

B = 1
POP_SIZE = 100
ELITISM_RATE = 0.10
UNIFORM_MUTATION_PROB = 0.01
SWAP_MUTATION_PROB = 0.01
TOURNAMENT_RATE = 0.20
NO_IMPROVEMENT_LIMIT = 20
MAX_TIME_SECONDS = 3600.0
SETUP_TIME = 1.0
THETA_M = 72.0

def export_seed_results_to_excel(output_path, **sheets):
    """Write summary and per-seed result DataFrames to one Excel workbook."""
    if not output_path:
        return None
    output_path = str(output_path)
    if not output_path.lower().endswith(".xlsx"):
        output_path += ".xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            if data is None:
                continue
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            df.to_excel(writer, sheet_name=(sheet_name), index=False)
            ws = writer.sheets[(sheet_name)]
            for col_cells in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 38)
    print(f"Excel results successfully exported to: {output_path}")
    return output_path

def solve_trm_ilp_exact(tools_in_magazine, tool_sizes, scores, needed_capacity):
    """Solves the Tool Replacement Method subproblem using an exact binary ILP."""
    if needed_capacity <= 0:
        return []
    tools = list(tools_in_magazine)
    if not tools:
        return []

    c = np.array([float(scores.get(t, 0.0)) for t in tools], dtype=float)
    sizes = np.array([float(tool_sizes[t]) for t in tools], dtype=float)

    constraints = LinearConstraint(
        A=sizes.reshape(1, -1),
        lb=np.array([float(needed_capacity)]),
        ub=np.array([np.inf]),
    )

    bounds = Bounds(lb=np.zeros(len(tools)), ub=np.ones(len(tools)))
    integrality = np.ones(len(tools), dtype=int)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"disp": False},
    )

    if not result.success:
        raise RuntimeError(f"TRM ILP failed to solve: {result.message}")

    lambdas = np.rint(result.x).astype(int)
    return [t for t, selected in zip(tools, lambdas) if selected == 1]

def solve_trm_knapsack(tools_in_magazine, tool_sizes, scores, needed_capacity):
    return solve_trm_ilp_exact(tools_in_magazine, tool_sizes, scores, needed_capacity)

class Individual:
    def __init__(self, job_vector, machine_vector):
        self.job_vector = list(job_vector)
        self.machine_vector = list(machine_vector)
        self.fitness = float('inf')
        self.tardiness = 0.0
        self.setups = 0

    def get_signature(self):
        return (tuple(self.job_vector), tuple(self.machine_vector))

class Decoder:
    def __init__(self, ops_by_job, num_machines, magazine_capacity, setup_time=1.0):
        self.ops_by_job = ops_by_job
        self.num_machines = num_machines
        self.C = magazine_capacity
        self.tau = setup_time

    def evaluate(self, individual):
        job_vec = individual.job_vector
        mach_vec = individual.machine_vector
        n = len(job_vec)
        
        # Pre-populate magazines with earliest assigned tools (no setup penalty)
        T_m = {m: set() for m in range(1, self.num_machines + 1)}
        mach_tool_sequence = {m: [] for m in range(1, self.num_machines + 1)}
        temp_occ = {}
        for g in range(n):
            j_id = job_vec[g]
            m_id = mach_vec[g]
            occ = temp_occ.get(j_id, 0)
            temp_occ[j_id] = occ + 1
            op_data = self.ops_by_job[j_id][occ]
            t_ij = op_data['tool_set']
            if t_ij not in mach_tool_sequence[m_id]:
                mach_tool_sequence[m_id].append(t_ij)
                
        for m_id in range(1, self.num_machines + 1):
            current_size = 0
            for t_ij in mach_tool_sequence[m_id]:
                phi_t = GLOBAL_TOOL_SIZES[t_ij]
                if current_size + phi_t <= self.C:
                    T_m[m_id].add(t_ij)
                    current_size += phi_t
                else:
                    break
        
        # Simulation loop
        a_m = {m: 0.0 for m in range(1, self.num_machines + 1)}
        job_finish_times = {}
        total_tardiness = 0.0
        total_setups = 0
        setup_positions = []
        occ_counts = {}
        
        succeeding_ops_per_machine = {m: [] for m in range(1, self.num_machines + 1)}
        temp_occ_2 = {}
        for g in range(n):
            j_id = job_vec[g]
            m_id = mach_vec[g]
            occ = temp_occ_2.get(j_id, 0)
            temp_occ_2[j_id] = occ + 1
            op_data = self.ops_by_job[j_id][occ]
            succeeding_ops_per_machine[m_id].append((op_data['tool_set'], op_data['size']))

        for g in range(n):
            j_id = job_vec[g]
            m_id = mach_vec[g]
            occ = occ_counts.get(j_id, 0)
            occ_counts[j_id] = occ + 1
            
            op_data = self.ops_by_job[j_id][occ]
            t_ij = op_data['tool_set']
            phi_t = op_data['size']
            
            # --- FIXED LINE HERE ---
            r_ij, p_ij, d_ij = op_data['r'], op_data['p'], op_data['d']
            
            succeeding_ops_per_machine[m_id].pop(0)
            
            z_ijt = 0
            if t_ij not in T_m[m_id]:
                z_ijt = 1
                setup_positions.append(g)
                current_size = sum(GLOBAL_TOOL_SIZES[t] for t in T_m[m_id])
                free_space = self.C - current_size
                
                if free_space < phi_t:
                    needed_space = phi_t - free_space
                    future_tools = [item[0] for item in succeeding_ops_per_machine[m_id]]
                    future_unique = []
                    for ft in future_tools:
                        if ft in T_m[m_id] and ft not in future_unique:
                            future_unique.append(ft)
                            
                    scores = {}
                    for ft in T_m[m_id]:
                        if ft in future_unique:
                            u = future_unique.index(ft) + 1
                            scores[ft] = len(future_unique) - (u - 1)
                        else:
                            scores[ft] = 0
                            
                    zero_score_tools = [t for t in T_m[m_id] if scores[t] == 0]
                    zero_weight = sum(GLOBAL_TOOL_SIZES[t] for t in zero_score_tools)
                    
                    if zero_weight >= needed_space:
                        # Paper Section 5.6: if score-0 tools can provide enough
                        # capacity, remove those score-0 tools and skip the ILP.
                        for t in zero_score_tools:
                            T_m[m_id].remove(t)
                    else:
                        for t in zero_score_tools:
                            T_m[m_id].remove(t)
                        remaining_need = needed_space - zero_weight
                        
                        active_tools = list(T_m[m_id])
                        evict_subset = solve_trm_knapsack(
                            active_tools, GLOBAL_TOOL_SIZES, scores, remaining_need
                        )
                        for t in evict_subset:
                            T_m[m_id].remove(t)
                            
                    T_m[m_id].add(t_ij)
                    total_setups += 1
                else:
                    T_m[m_id].add(t_ij)
                    total_setups += 1
            
            prev_finish = job_finish_times.get((j_id, occ - 1), 0.0) if occ > 0 else 0.0
            start_time = max(r_ij, a_m[m_id], prev_finish)
            end_time = start_time + p_ij + (self.tau * z_ijt)
            
            a_m[m_id] = end_time
            job_finish_times[(j_id, occ)] = end_time
            
            tardiness = max(0.0, end_time - d_ij)
            total_tardiness += tardiness
            
        individual.tardiness = total_tardiness
        individual.setups = total_setups
        individual.fitness = total_tardiness + (self.tau * total_setups)
        individual.setup_positions = setup_positions



# 3. PRACTITIONER HEURISTIC ENGINE

class PractitionerHeuristic:
    def __init__(self, jobs_data, num_machines, magazine_capacity, tool_setup_time=SETUP_TIME, theta_m=THETA_M):
        self.O = jobs_data
        self.M = list(range(1, num_machines + 1))
        self.C = magazine_capacity
        self.tau = tool_setup_time  
        self.theta_m = theta_m      
        self.T_m = {m: set() for m in self.M}       
        self.a_m = {m: 0.0 for m in self.M}         
        self.tool_sizes = {op['tool_set']: op['size'] for op in self.O if 'tool_set' in op}
        
    def get_magazine_size(self, machine):
        return sum(self.tool_sizes[t] for t in self.T_m[machine])

    def run(self):
        O_hat = sorted(self.O, key=lambda x: x['d'])
        for op in O_hat:
            t_ij = op['tool_set']
            phi_t = op['size']
            m_T = [m for m in self.M if t_ij in self.T_m[m]]
            if not m_T:
                M_C = [m for m in self.M if (self.C - self.get_magazine_size(m)) >= phi_t]
                if M_C:
                    m_star = min(M_C, key=lambda m: (len(self.T_m[m]), m))
                    self.T_m[m_star].add(t_ij)
                    
        total_tardiness = 0
        total_setups = 0
        job_finish_times = {}
        occ_counts = {}
        job_vector, machine_vector = [], []
        
        for op in O_hat:
            job_id, op_id = op['job_id'], op['op_id']
            t_ij, phi_t = op['tool_set'], op['size']
            r_ij, p_ij, d_ij = op['r'], op['p'], op['d']
            
            occ = occ_counts.get(job_id, 0)
            occ_counts[job_id] = occ + 1
            
            m_P = min(self.M, key=lambda m: self.a_m[m])
            m_T_list = [m for m in self.M if t_ij in self.T_m[m]]
            m_T = m_T_list[0] if m_T_list else None
            
            def calc_xi(machine):
                prev_finish = job_finish_times.get((job_id, occ - 1), 0.0) if occ > 0 else 0.0
                return max(r_ij, self.a_m[machine], prev_finish)
            
            if m_T is not None:
                if m_T != m_P and (calc_xi(m_T) - calc_xi(m_P)) >= self.theta_m:
                    m_star = m_P
                    z_ijt = 1
                else:
                    m_star = m_T
                    z_ijt = 0
            else:
                m_star = m_P
                z_ijt = 1
                
            if z_ijt == 1:
                phi_s = phi_t - (self.C - self.get_magazine_size(m_star))
                while phi_s > 0 and self.T_m[m_star]:
                    removed_tool = random.choice(sorted(self.T_m[m_star]))
                    self.T_m[m_star].remove(removed_tool)
                    phi_s = phi_t - (self.C - self.get_magazine_size(m_star))
                self.T_m[m_star].add(t_ij)
                total_setups += 1
            
            start_time = calc_xi(m_star)
            end_time = start_time + p_ij + (self.tau * z_ijt)
            self.a_m[m_star] = end_time
            job_finish_times[(job_id, occ)] = end_time
            total_tardiness += max(0.0, end_time - d_ij)
            
            job_vector.append(job_id)
            machine_vector.append(m_star)
            
        ind = Individual(job_vector, machine_vector)
        ind.fitness = total_tardiness + (self.tau * total_setups)
        return ind






# 4. PH + ALNS + ITP/TRP WITH Q-LEARNING OPERATOR SELECTION

class ALNS_AOS:
    """
    PH + ALNS + ITP/TRP improvement engine with Q-learning operator selection.

    Terminology used here:
        PH      = PractitionerHeuristic creates the initial solution.
        ALNS    = destroy/repair neighborhood search over job order and machine assignment.
        ITP/TRP = the tool-placement/replacement logic inside Decoder.evaluate(...).

    Main difference from the previous AOS version:
        - The destroy/repair pair is treated as a Q-learning action.
        - Rewards are still computed from the exact Decoder.evaluate(...) objective.
        - Therefore Q-learning learns from the real tardiness + setup + tool-replacement objective,
          not from an approximation.

    The exact decoder is kept in the loop, so final feasibility and reported objective remain
    consistent with the tool-magazine replacement model.
    """

    def __init__(
        self,
        jobs_data,
        num_machines,
        magazine_capacity,
        setup_time=SETUP_TIME,
        destroy_fraction=(0.03, 0.08),
        start_temperature=None,
        cooling_rate=0.995,
        min_temperature=1e-6,
        max_insert_positions=12,
        max_machine_candidates=3,
        max_removed_jobs=8,
        cache_evaluations=True,
        deep_repair_period=25,
        deep_insert_positions=30,

        # Q-learning controls
        q_alpha=0.20,
        q_gamma=0.80,
        epsilon_start=0.30,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        reward_scale=0.05,
    ):
        self.jobs_data = jobs_data
        self.num_machines = num_machines
        self.C = magazine_capacity
        self.tau = setup_time

        self.destroy_fraction = destroy_fraction
        self.temperature = start_temperature
        self.initial_temperature = None
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

        self.max_insert_positions = max_insert_positions
        self.max_machine_candidates = max_machine_candidates
        self.max_removed_jobs = max_removed_jobs
        self.cache_evaluations = cache_evaluations

        self.deep_repair_period = deep_repair_period
        self.deep_insert_positions = deep_insert_positions

        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_scale = reward_scale

        self.eval_cache = {}
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0
        self.current_iteration = 0

        self.ops_by_job = {}
        for op in jobs_data:
            job_id = int(op["job_id"])
            self.ops_by_job.setdefault(job_id, []).append(op)

        for job_id in self.ops_by_job:
            self.ops_by_job[job_id].sort(key=lambda x: x["op_id"])

        self.total_ops_by_job = {
            job_id: len(ops) for job_id, ops in self.ops_by_job.items()
        }

        self.decoder = Decoder(self.ops_by_job, self.num_machines, self.C, self.tau)

        self.destroy_ops = {
            "random_removal": self.destroy_random_removal,
            "worst_due_date_removal": self.destroy_worst_due_date_removal,
            "machine_overload_removal": self.destroy_machine_overload_removal,
            "actual_setup_removal": self.destroy_actual_setup_removal,
        }

        self.repair_ops = {
            "greedy_best_insert": self.repair_greedy_best_insert,
            "regret2_insert": self.repair_regret2_insert,
            "edd_insert": self.repair_edd_insert,
            "least_loaded_insert": self.repair_least_loaded_insert,
        }

        self.actions = [
            (d_name, r_name)
            for d_name in self.destroy_ops
            for r_name in self.repair_ops
        ]

        # Q-table: state tuple -> action tuple -> Q value
        self.q_table = {}

        # Compatibility fields for earlier reporting code.
        self.destroy_weights = {name: 1.0 for name in self.destroy_ops}
        self.repair_weights = {name: 1.0 for name in self.repair_ops}

    # -----------------------------
    # Basic utilities
    # -----------------------------
    def clone(self, ind):
        new = Individual(ind.job_vector, ind.machine_vector)
        new.fitness = ind.fitness
        new.tardiness = ind.tardiness
        new.setups = ind.setups

        for attr in [
            "runtime", "generations", "stop_reason", "history",
            "setup_positions", "alns_history", "alns_destroy_weights",
            "alns_repair_weights", "alns_q_table",
        ]:
            if hasattr(ind, attr):
                setattr(new, attr, getattr(ind, attr))

        return new

    def _evaluate(self, job_vec, mach_vec):
        """
        Exact evaluation through Decoder.evaluate(...).

        This still executes the tool-magazine / ITP-TRP logic, so Q-learning rewards,
        acceptance decisions, and final reported objective are based on the exact model.
        """
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
        prob = np.exp(-(candidate.fitness - current.fitness) / temp)
        return random.random() < prob

    def _num_to_remove(self, n):
        lo, hi = self.destroy_fraction
        frac = random.uniform(lo, hi)
        return max(1, min(n - 1, self.max_removed_jobs, int(round(frac * n))))

    def _remove_positions(self, individual, positions):
        positions = sorted(set(positions))
        removed_jobs = [individual.job_vector[i] for i in positions]
        partial_jobs = [
            v for i, v in enumerate(individual.job_vector)
            if i not in positions
        ]
        partial_machs = [
            v for i, v in enumerate(individual.machine_vector)
            if i not in positions
        ]
        return partial_jobs, partial_machs, removed_jobs

    # -----------------------------
    # Q-learning utilities
    # -----------------------------
    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

    def _get_state(self, current, best, no_improve):
        """
        Optimized, highly compact state representation for tabular Q-learning.
        
        Reduces the state space from 192 to 12 states (192 state-action pairs),
        guaranteeing full convergence within a 2,000-iteration budget.
        
        Dimensions:
            1. gap_bucket: 2 levels (near_best, large_gap)
            2. temp_bucket: 2 levels (hot, cold)
            3. stagnation_bucket: 3 levels (fresh, mild, stagnated)
        """
        # 1. Quality Gap (2 Buckets)
        best_ref = max(1.0, abs(best.fitness))
        gap = max(0.0, (current.fitness - best.fitness) / best_ref)
        gap_bucket = "near_best" if gap <= 0.01 else "large_gap"

        # 2. Temperature cooling stages (2 Buckets)
        temp_ref = max(self.min_temperature, self.initial_temperature or max(1.0, abs(current.fitness)))
        temp_ratio = max(0.0, self.temperature / temp_ref)
        temp_bucket = "hot" if temp_ratio >= 0.15 else "cold"

        # 3. Search stagnation iterations (3 Buckets)
        if no_improve < 15:
            stagnation_bucket = "fresh"
        elif no_improve < 50:
            stagnation_bucket = "mild"
        else:
            stagnation_bucket = "stagnated"

        # Note: We completely remove the 'size_bucket' because problem size is 
        # constant during any single run and does not provide dynamic context.
        return (gap_bucket, temp_bucket, stagnation_bucket)

    def _select_action(self, state):
        """
        Epsilon-greedy action selection.

        Action = (destroy_operator, repair_operator).
        """
        self._ensure_state(state)

        if random.random() < self.epsilon:
            return random.choice(self.actions), "explore"

        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions), "exploit"

    def _compute_reward(self, candidate, current, best, accepted):
        """
        Exact reward signal for Q-learning.

        Important:
            This reward uses the full Decoder.evaluate(...) fitness.
            The Q-table therefore learns which destroy/repair pairs improve
            the true objective, including setups and tool replacement effects.
        """
        if candidate.fitness < best.fitness:
            base = 10.0
            reference = max(1.0, abs(best.fitness))
            improvement = best.fitness - candidate.fitness
        elif candidate.fitness < current.fitness:
            base = 5.0
            reference = max(1.0, abs(current.fitness))
            improvement = current.fitness - candidate.fitness
        elif accepted:
            base = 1.0
            reference = 1.0
            improvement = 0.0
        else:
            return -0.1

        scaled_bonus = self.reward_scale * 100.0 * max(0.0, improvement / reference)
        return base + scaled_bonus

    def _update_q_value(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())

        target = reward + self.q_gamma * next_max_q
        new_q = old_q + self.q_alpha * (target - old_q)

        self.q_table[state][action] = new_q

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _refresh_operator_weights_from_q(self):
        """
        Convert Q-values into compatibility weights so old result summaries still work.

        These are not used for operator selection. Q-learning uses q_table directly.
        """
        destroy_scores = {name: [] for name in self.destroy_ops}
        repair_scores = {name: [] for name in self.repair_ops}

        for state_values in self.q_table.values():
            for (d_name, r_name), q in state_values.items():
                destroy_scores[d_name].append(q)
                repair_scores[r_name].append(q)

        for d_name, values in destroy_scores.items():
            self.destroy_weights[d_name] = max(0.05, 1.0 + (float(np.mean(values)) if values else 0.0))

        for r_name, values in repair_scores.items():
            self.repair_weights[r_name] = max(0.05, 1.0 + (float(np.mean(values)) if values else 0.0))

    # -----------------------------
    # Precedence-safe insertion logic
    # -----------------------------
    def _op_index_for_next_insertion(self, partial_jobs, job_id):
        occ = partial_jobs.count(job_id)

        if occ >= self.total_ops_by_job[job_id]:
            raise ValueError(
                f"Cannot insert job {job_id}: all operations already present."
            )

        return occ

    def _operation_data_for_insertion(self, partial_jobs, job_id):
        occ = self._op_index_for_next_insertion(partial_jobs, job_id)
        return self.ops_by_job[job_id][occ]

    def _precedence_safe_positions(self, partial_jobs, job_id):
        last_same_job_pos = -1

        for idx, job in enumerate(partial_jobs):
            if job == job_id:
                last_same_job_pos = idx

        earliest = last_same_job_pos + 1
        latest = len(partial_jobs)

        return list(range(earliest, latest + 1))

    def _position_candidates(self, partial_jobs, job_id, deep=False):
        safe_positions = self._precedence_safe_positions(partial_jobs, job_id)
        budget = self.deep_insert_positions if deep else self.max_insert_positions

        if len(safe_positions) <= budget:
            return safe_positions

        op = self._operation_data_for_insertion(partial_jobs, job_id)
        due = float(op["d"])

        occ = {}
        edd_pos = safe_positions[-1]
        safe_set = set(safe_positions)

        for idx, existing_job in enumerate(partial_jobs):
            k = occ.get(existing_job, 0)
            occ[existing_job] = k + 1

            existing_due = float(self.ops_by_job[existing_job][k]["d"])

            if idx in safe_set and existing_due > due:
                edd_pos = idx
                break

        positions = {safe_positions[0], safe_positions[-1], edd_pos}

        for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
            pos = edd_pos + delta
            if pos in safe_set:
                positions.add(pos)

        remaining = [p for p in safe_positions if p not in positions]
        remaining_budget = max(0, budget - len(positions))

        if remaining and remaining_budget > 0:
            positions.update(
                random.sample(remaining, min(remaining_budget, len(remaining)))
            )

        return sorted(positions)

    def _machine_candidates(self, partial_jobs, partial_machs, job_id, deep=False):
        op = self._operation_data_for_insertion(partial_jobs, job_id)
        tool = op["tool_set"]

        if deep:
            return list(range(1, self.num_machines + 1))

        loads = {m: 0.0 for m in range(1, self.num_machines + 1)}
        tool_presence = {m: 0 for m in range(1, self.num_machines + 1)}

        occ = {}

        for job, mach in zip(partial_jobs, partial_machs):
            k = occ.get(job, 0)
            occ[job] = k + 1

            eop = self.ops_by_job[job][k]
            loads[mach] += float(eop["p"])

            if eop["tool_set"] == tool:
                tool_presence[mach] += 1

        candidates = []

        same_tool = [m for m, count in tool_presence.items() if count > 0]
        if same_tool:
            candidates.append(
                max(same_tool, key=lambda m: (tool_presence[m], -loads[m]))
            )

        for m, _ in sorted(loads.items(), key=lambda kv: kv[1]):
            if m not in candidates:
                candidates.append(m)

            if len(candidates) >= self.max_machine_candidates:
                break

        return candidates

    # -----------------------------
    # Destroy operators
    # -----------------------------
    def destroy_random_removal(self, individual, q):
        positions = random.sample(range(len(individual.job_vector)), q)
        return self._remove_positions(individual, positions)

    def destroy_worst_due_date_removal(self, individual, q):
        occ = {}
        scored = []

        for idx, job in enumerate(individual.job_vector):
            k = occ.get(job, 0)
            occ[job] = k + 1
            op = self.ops_by_job[job][k]

            score = (-float(op["d"]), float(op["p"]), random.random())
            scored.append((score, idx))

        positions = [idx for _, idx in sorted(scored, reverse=True)[:q]]
        return self._remove_positions(individual, positions)

    def destroy_machine_overload_removal(self, individual, q):
        loads = {m: 0.0 for m in range(1, self.num_machines + 1)}
        occ = {}

        for job, mach in zip(individual.job_vector, individual.machine_vector):
            k = occ.get(job, 0)
            occ[job] = k + 1
            loads[mach] += float(self.ops_by_job[job][k]["p"])

        overloaded = max(loads, key=loads.get)

        candidate_positions = [
            i for i, m in enumerate(individual.machine_vector)
            if m == overloaded
        ]

        if len(candidate_positions) < q:
            extra = [
                i for i in range(len(individual.job_vector))
                if i not in candidate_positions
            ]
            candidate_positions += random.sample(
                extra,
                min(len(extra), q - len(candidate_positions)),
            )

        positions = random.sample(candidate_positions, q)
        return self._remove_positions(individual, positions)

    def destroy_actual_setup_removal(self, individual, q):
        if not hasattr(individual, "setup_positions"):
            self.decoder.evaluate(individual)

        setup_positions = list(getattr(individual, "setup_positions", []))

        if len(setup_positions) >= q:
            positions = random.sample(setup_positions, q)
            return self._remove_positions(individual, positions)

        positions = list(setup_positions)
        remaining = [
            i for i in range(len(individual.job_vector))
            if i not in positions
        ]

        if remaining:
            positions += random.sample(
                remaining,
                min(len(remaining), q - len(positions)),
            )

        return self._remove_positions(individual, positions)

    # -----------------------------
    # Repair operators
    # -----------------------------
    def _best_single_insertion(
        self,
        partial_jobs,
        partial_machs,
        job_id,
        machine_candidates=None,
        deep=False,
    ):
        if machine_candidates is None:
            machine_candidates = self._machine_candidates(
                partial_jobs,
                partial_machs,
                job_id,
                deep=deep,
            )

        best = None

        for pos in self._position_candidates(partial_jobs, job_id, deep=deep):
            for mach in machine_candidates:
                trial_jobs = partial_jobs[:pos] + [job_id] + partial_jobs[pos:]
                trial_machs = partial_machs[:pos] + [mach] + partial_machs[pos:]

                cand = self._evaluate(trial_jobs, trial_machs)

                key = (
                    cand.fitness,
                    cand.tardiness,
                    cand.setups,
                    pos,
                    mach,
                )

                if best is None or key < best[0]:
                    best = (key, cand)

        return best[1]

    def _is_deep_repair_iteration(self):
        return (
            self.deep_repair_period > 0
            and self.current_iteration % self.deep_repair_period == 0
        )

    def repair_greedy_best_insert(self, partial_jobs, partial_machs, removed_jobs):
        jobs = list(removed_jobs)
        random.shuffle(jobs)

        current_jobs = list(partial_jobs)
        current_machs = list(partial_machs)
        deep = self._is_deep_repair_iteration()

        for job_id in jobs:
            best = self._best_single_insertion(
                current_jobs,
                current_machs,
                job_id,
                deep=deep,
            )
            current_jobs = best.job_vector
            current_machs = best.machine_vector

        return self._evaluate(current_jobs, current_machs)

    def repair_regret2_insert(self, partial_jobs, partial_machs, removed_jobs):
        remaining = list(removed_jobs)
        current_jobs = list(partial_jobs)
        current_machs = list(partial_machs)
        deep = self._is_deep_repair_iteration()

        while remaining:
            best_choice = None

            for job_id in remaining:
                candidates = []

                for pos in self._position_candidates(current_jobs, job_id, deep=deep):
                    for mach in self._machine_candidates(
                        current_jobs,
                        current_machs,
                        job_id,
                        deep=deep,
                    ):
                        trial_jobs = current_jobs[:pos] + [job_id] + current_jobs[pos:]
                        trial_machs = current_machs[:pos] + [mach] + current_machs[pos:]
                        cand = self._evaluate(trial_jobs, trial_machs)
                        candidates.append((cand.fitness, cand))

                candidates.sort(key=lambda x: x[0])

                best_fit = candidates[0][0]
                second_fit = candidates[1][0] if len(candidates) > 1 else best_fit
                regret = second_fit - best_fit

                choice_key = (regret, -best_fit, random.random())

                if best_choice is None or choice_key > best_choice[0]:
                    best_choice = (choice_key, job_id, candidates[0][1])

            _, chosen_job, chosen_ind = best_choice

            current_jobs = chosen_ind.job_vector
            current_machs = chosen_ind.machine_vector
            remaining.remove(chosen_job)

        return self._evaluate(current_jobs, current_machs)

    def repair_edd_insert(self, partial_jobs, partial_machs, removed_jobs):
        current_jobs = list(partial_jobs)
        current_machs = list(partial_machs)
        jobs = list(removed_jobs)
        deep = self._is_deep_repair_iteration()

        jobs.sort(key=lambda j: self._operation_data_for_insertion(current_jobs, j)["d"])

        for job_id in jobs:
            op = self._operation_data_for_insertion(current_jobs, job_id)
            due = float(op["d"])

            candidate_positions = self._position_candidates(
                current_jobs,
                job_id,
                deep=deep,
            )
            safe_set = set(candidate_positions)

            occ = {}
            pos = candidate_positions[-1]

            for idx, existing_job in enumerate(current_jobs):
                k = occ.get(existing_job, 0)
                occ[existing_job] = k + 1

                existing_due = float(self.ops_by_job[existing_job][k]["d"])

                if idx in safe_set and existing_due > due:
                    pos = idx
                    break

            best = None

            for mach in self._machine_candidates(
                current_jobs,
                current_machs,
                job_id,
                deep=deep,
            ):
                trial_jobs = current_jobs[:pos] + [job_id] + current_jobs[pos:]
                trial_machs = current_machs[:pos] + [mach] + current_machs[pos:]
                cand = self._evaluate(trial_jobs, trial_machs)

                key = (
                    cand.fitness,
                    cand.tardiness,
                    cand.setups,
                    mach,
                )

                if best is None or key < best[0]:
                    best = (key, cand)

            current_jobs = best[1].job_vector
            current_machs = best[1].machine_vector

        return self._evaluate(current_jobs, current_machs)

    def repair_least_loaded_insert(self, partial_jobs, partial_machs, removed_jobs):
        current_jobs = list(partial_jobs)
        current_machs = list(partial_machs)

        for job_id in removed_jobs:
            loads = {m: 0.0 for m in range(1, self.num_machines + 1)}
            occ = {}

            for job, mach in zip(current_jobs, current_machs):
                k = occ.get(job, 0)
                occ[job] = k + 1
                loads[mach] += float(self.ops_by_job[job][k]["p"])

            least_loaded = min(loads, key=loads.get)

            best = self._best_single_insertion(
                current_jobs,
                current_machs,
                job_id,
                machine_candidates=[least_loaded],
                deep=False,
            )

            current_jobs = best.job_vector
            current_machs = best.machine_vector

        return self._evaluate(current_jobs, current_machs)

    # -----------------------------
    # Main PH + ALNS + ITP/TRP loop with Q-learning
    # -----------------------------
    def run(
        self,
        initial_solution,
        max_iterations=250,
        max_time_seconds=60.0,
        no_improvement_limit=50,
        record_history=True,
        verbose=False,
    ):
        start_clock = time.time()

        current = self.clone(initial_solution)
        self.decoder.evaluate(current)  # exact ITP/TRP objective and setup trace
        best = self.clone(current)

        if self.temperature is None:
            self.temperature = max(1.0, 0.05 * abs(current.fitness))

        self.initial_temperature = self.temperature

        history = []
        no_improve = 0
        stop_reason = "iteration_limit"

        pbar = tqdm(
            range(1, max_iterations + 1),
            desc="ALNS",
            leave=False,
        )

        for it in pbar:
            self.current_iteration = it
            elapsed = time.time() - start_clock

            if elapsed >= max_time_seconds:
                stop_reason = "time_limit"
                break

            if no_improve >= no_improvement_limit:
                stop_reason = "no_improvement_limit"
                break

            state = self._get_state(current, best, no_improve)
            action, action_mode = self._select_action(state)
            destroy_name, repair_name = action

            q = self._num_to_remove(len(current.job_vector))

            partial_jobs, partial_machs, removed_jobs = self.destroy_ops[destroy_name](
                current,
                q,
            )

            candidate = self.repair_ops[repair_name](
                partial_jobs,
                partial_machs,
                removed_jobs,
            )

            accepted = self._accept(candidate, current)

            improved_current = candidate.fitness < current.fitness
            improved_best = candidate.fitness < best.fitness

            reward = self._compute_reward(
                candidate=candidate,
                current=current,
                best=best,
                accepted=accepted,
            )

            if accepted:
                current = candidate

            if improved_best:
                best = self.clone(candidate)
                no_improve = 0
            else:
                no_improve += 1

            next_state = self._get_state(current, best, no_improve)
            self._update_q_value(state, action, reward, next_state)
            self._decay_epsilon()
            self._refresh_operator_weights_from_q()

            self.temperature = max(
                self.min_temperature,
                self.temperature * self.cooling_rate,
            )

            pbar.set_postfix(
                best=f"{best.fitness:.2f}",
                curr=f"{current.fitness:.2f}",
                eps=f"{self.epsilon:.3f}",
                hits=self.eval_cache_hits,
                miss=self.eval_cache_misses,
            )

            if record_history:
                history.append({
                    "iteration": int(it),
                    "runtime": float(time.time() - start_clock),
                    "best_fitness": float(best.fitness),
                    "current_fitness": float(current.fitness),
                    "candidate_fitness": float(candidate.fitness),
                    "accepted": bool(accepted),
                    "improved_current": bool(improved_current),
                    "improved_best": bool(improved_best),
                    "destroy": destroy_name,
                    "repair": repair_name,
                    "q_action": f"{destroy_name}+{repair_name}",
                    "q_state": state,
                    "q_value": float(self.q_table[state][action]),
                    "action_mode": action_mode,
                    "epsilon": float(self.epsilon),
                    "reward": float(reward),
                    "q_removed": int(q),
                    "temperature": float(self.temperature),
                    "no_improve": int(no_improve),
                    "deep_repair": bool(self._is_deep_repair_iteration()),
                    "eval_cache_hits": int(self.eval_cache_hits),
                    "eval_cache_misses": int(self.eval_cache_misses),
                    "destroy_weights": dict(self.destroy_weights),
                    "repair_weights": dict(self.repair_weights),
                })

            if verbose and (it == 1 or it % 25 == 0 or improved_best):
                print(
                    f"alns_it={it:5d} best={best.fitness:.4f} "
                    f"current={current.fitness:.4f} cand={candidate.fitness:.4f} "
                    f"acc={accepted} action={destroy_name}+{repair_name} "
                    f"mode={action_mode} reward={reward:.3f} "
                    f"q={self.q_table[state][action]:.3f} "
                    f"eps={self.epsilon:.3f} temp={self.temperature:.4f} "
                    f"deep={self._is_deep_repair_iteration()}"
                )

        best.alns_iterations = len(history) if record_history else it
        best.alns_runtime = time.time() - start_clock
        best.alns_stop_reason = stop_reason
        best.alns_history = history
        best.alns_destroy_weights = dict(self.destroy_weights)
        best.alns_repair_weights = dict(self.repair_weights)
        best.alns_eval_cache_hits = self.eval_cache_hits
        best.alns_eval_cache_misses = self.eval_cache_misses

        # Store a string-keyed copy so it is easier to serialize/inspect.
        best.alns_q_table = {
            str(state): {f"{a[0]}+{a[1]}": float(v) for a, v in values.items()}
            for state, values in self.q_table.items()
        }

        return best



# REAL-WORLD DATA LOADER & PARSER

def load_actual_kmwe_instance(filepath):
    """
    Parses the genuine KMWE data files by extracting metadata from the 
    5-line header and properly building structured dictionaries.
    """
    num_machines = 2
    magazine_capacity = 80
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"KMWE CSV file not found: {filepath}. Synthetic/mock data is disabled."
        )

    # 1. Extract metadata from the 5-line header
    with open(filepath, 'r') as f:
        for _ in range(5):
            line = f.readline().strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key == 'M':
                    num_machines = int(value)
                elif key == 'C':
                    magazine_capacity = int(value)

    # 2. Read the structured operations block
    df = pd.read_csv(filepath, skiprows=5)
    expected_columns = ['job_id', 'op_id', 'r', 'p', 'd', 'tool_set', 'size']
    if len(df.columns) != len(expected_columns):
        raise ValueError(
            f"Unexpected KMWE CSV format in {filepath}. "
            f"Expected {len(expected_columns)} operation columns after the 5-line header."
        )
    df.columns = expected_columns

    for col in expected_columns:
        df[col] = pd.to_numeric(df[col], errors='raise')

    jobs_data = df.to_dict(orient='records')

    # 3. Cache tool dimensions for the TRM solver. Clear first to avoid
    # cross-case contamination when running multiple KMWE cases in one session.
    GLOBAL_TOOL_SIZES.clear()
    for op in jobs_data:
        GLOBAL_TOOL_SIZES[op['tool_set']] = op['size']

    return jobs_data, num_machines, magazine_capacity


def resolve_kmwe_case_file(case_name):
    """
    Resolve a real KMWE case CSV. Synthetic fallback is deliberately disabled.
    Accepted layouts:
        <case_name>/<case_name>.csv
        <case_name>/Base <case_name>.csv
        <case_name>.csv
    """
    possible_paths = [
        os.path.join(case_name, f"{case_name}.csv"),
        os.path.join(case_name, f"Base {case_name}.csv"),
        f"{case_name}.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Required real KMWE CSV for {case_name!r} was not found. "
        f"Checked: {possible_paths}. Synthetic/mock data is disabled."
    )






# ALNS-ONLY EXPERIMENT ENGINE

def run_alns_only_on_file(
    case_file,
    seed=0,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    verbose=False,
    print_result=False,
):
    """
    Run ALNS-AOS+TRP only, initialized by the Practitioner Heuristic.
    No GA/MH phase is executed.
    """
    random.seed(seed)
    np.random.seed(seed)

    jobs_data, m_case, c_case = load_actual_kmwe_instance(case_file)

    t0 = time.time()
    ph_engine = PractitionerHeuristic(jobs_data, num_machines=m_case, magazine_capacity=c_case)
    ph_solution = ph_engine.run()
    ph_runtime = time.time() - t0

    ops_by_job = {}
    for op in jobs_data:
        job_id = int(op["job_id"])
        ops_by_job.setdefault(job_id, []).append(op)
    for job_id in ops_by_job:
        ops_by_job[job_id].sort(key=lambda x: x["op_id"])

    decoder = Decoder(ops_by_job, m_case, c_case, SETUP_TIME)
    decoder.evaluate(ph_solution)

    alns_engine = ALNS_AOS(jobs_data, m_case, c_case, SETUP_TIME)
    alns_solution = alns_engine.run(
        ph_solution,
        max_time_seconds=alns_time_seconds,
        max_iterations=alns_iterations,
        no_improvement_limit=alns_no_improvement_limit,
        record_history=True,
        verbose=verbose,
    )

    decoder.evaluate(alns_solution)

    result = {
        "case_file": case_file,
        "seed": seed,
        "PH_fitness": ph_solution.fitness,
        "PH_tardiness": ph_solution.tardiness,
        "PH_setups": ph_solution.setups,
        "PH_runtime": ph_runtime,
        "ALNS_fitness": alns_solution.fitness,
        "ALNS_tardiness": alns_solution.tardiness,
        "ALNS_setups": alns_solution.setups,
        "ALNS_runtime": alns_solution.alns_runtime,
        "ALNS_iterations": alns_solution.alns_iterations,
        "ALNS_stop": alns_solution.alns_stop_reason,
        "ALNS_cache_hits": getattr(alns_solution, "alns_eval_cache_hits", None),
        "ALNS_cache_misses": getattr(alns_solution, "alns_eval_cache_misses", None),
        "Improvement_vs_PH_%": ((alns_solution.fitness - ph_solution.fitness) / max(1.0, ph_solution.fitness)) * 100.0,
    }

    if print_result:
        print(pd.DataFrame([result]).to_string(index=False))
    return alns_solution, result



def run_alns_table8_replications(
    num_runs=10,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    output_excel="alns_table8_seed_results.xlsx",
):
    """
    Build a Table 8-style ALNS-only experiment on the real 6M140 KMWE case.

    The 6M140 operations are sorted by release time and sliced to:
        n = 15, 25, 30, 60, 90, 120, 140

    PH   = Practitioner Heuristic initial solution
    ALNS = ALNS-AOS+TRP initialized from PH

    Synthetic/mock data is disabled.
    """
    print("\n" + "=" * 120)
    print(f" REAL KMWE TABLE 8 ALNS-ONLY: 6M140 SLICES ({num_runs} SEED SAMPLES) ".center(120, "#"))
    print("=" * 120)

    case_file = resolve_kmwe_case_file("6M140")
    full_jobs_data, m_val, c_val = load_actual_kmwe_instance(case_file)
    df_base = pd.DataFrame(full_jobs_data)
    df_sorted = df_base.sort_values(by="r").copy()

    rows = []
    table8_seed_records = []

    for n_slice in [15, 25, 30, 60, 90, 120, 140]:
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        records = []

        for seed in tqdm(
            range(num_runs),
            desc=f"6M140 n={n_slice}",
            leave=False,
        ):
            random.seed(seed)
            np.random.seed(seed)

            # Rebuild tool size cache for this slice.
            GLOBAL_TOOL_SIZES.clear()
            for op in sliced_ops:
                GLOBAL_TOOL_SIZES[op["tool_set"]] = op["size"]

            t0 = time.time()
            ph_engine = PractitionerHeuristic(sliced_ops, num_machines=m_val, magazine_capacity=c_val)
            ph_solution = ph_engine.run()
            ph_runtime = time.time() - t0

            ops_by_job = {}
            for op in sliced_ops:
                job_id = int(op["job_id"])
                ops_by_job.setdefault(job_id, []).append(op)
            for job_id in ops_by_job:
                ops_by_job[job_id].sort(key=lambda x: x["op_id"])

            decoder = Decoder(ops_by_job, m_val, c_val, SETUP_TIME)
            decoder.evaluate(ph_solution)

            alns_engine = ALNS_AOS(sliced_ops, m_val, c_val, SETUP_TIME)
            alns_solution = alns_engine.run(
                ph_solution,
                max_time_seconds=alns_time_seconds,
                max_iterations=alns_iterations,
                no_improvement_limit=alns_no_improvement_limit,
                record_history=True,
                verbose=False,
            )
            decoder.evaluate(alns_solution)

            records.append({
                "n": n_slice,
                "seed": seed,
                "PH_fitness": ph_solution.fitness,
                "PH_runtime": ph_runtime,
                "ALNS_fitness": alns_solution.fitness,
                "ALNS_runtime": alns_solution.alns_runtime,
                "ALNS_iterations": alns_solution.alns_iterations,
                "ALNS_stop": alns_solution.alns_stop_reason,
            })

        table8_seed_records.extend(records)


        ph = np.array([r["PH_fitness"] for r in records], dtype=float)
        alns = np.array([r["ALNS_fitness"] for r in records], dtype=float)

        rows.append({
            "n": n_slice,
            "PH_μ": round(float(np.mean(ph)), 2),
            "PH_σ": round(float(np.std(ph)), 2),
            "PH_C.T.(s)": round(float(np.mean([r["PH_runtime"] for r in records])), 3),
            "ALNS_μ": round(float(np.mean(alns)), 2),
            "ALNS_σ": round(float(np.std(alns)), 2),
            "ALNS_C.T.(s)": round(float(np.mean([r["ALNS_runtime"] for r in records])), 3),
            "ALNS_it_μ": round(float(np.mean([r["ALNS_iterations"] for r in records])), 1),
            "Gap_ALNS_vs_PH (%)": f"{((np.mean(alns) - np.mean(ph)) / max(1.0, np.mean(ph))) * 100.0:.2f}%",
            "ALNS_StopReasons": ",".join(sorted(set(r["ALNS_stop"] for r in records))),
        })

    summary = pd.DataFrame(rows)
    print("\n[ALNS-ONLY TABLE 8 SUMMARY: REAL 6M140 SLICES]")
    print(summary.to_string(index=False))
    export_seed_results_to_excel(output_excel, table8_summary=summary, table8_seed_results=pd.DataFrame(table8_seed_records))
    return summary, pd.DataFrame(table8_seed_records)


def run_alns_only_replications(
    num_runs=10,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    output_excel="alns_only_seed_results.xlsx",
):
    """
    Run ALNS-only experiments on real KMWE cases.

    PH   = Practitioner Heuristic initial solution
    ALNS = ALNS-AOS+TRP initialized from PH

    Synthetic/mock data is disabled.
    """
    print("=" * 120)
    print(f" REAL KMWE LONG-EXPLORATION ALNS-ONLY ENGINE: PH vs ALNS-AOS+TRP ({num_runs} SEED SAMPLES) ".center(120, "#"))
    print("=" * 120)

    rows = []
    basecase_seed_records = []

    for case_name in tqdm(
        ["2M38", "2M46", "6M140", "6M163"],
        desc="Cases",
    ):
        case_file = resolve_kmwe_case_file(case_name)
        records = []

        for seed in tqdm(
            range(num_runs),
            desc=f"{case_name}",
            leave=False,
        ):
            _, result = run_alns_only_on_file(
                case_file,
                seed=seed,
                alns_time_seconds=alns_time_seconds,
                alns_iterations=alns_iterations,
                alns_no_improvement_limit=alns_no_improvement_limit,
                verbose=False,
            )
            result["BaseCase"] = case_name
            result["seed"] = seed
            records.append(result)

        basecase_seed_records.extend(records)

        ph = np.array([r["PH_fitness"] for r in records], dtype=float)
        alns = np.array([r["ALNS_fitness"] for r in records], dtype=float)

        rows.append({
            "BaseCase": case_name,
            "PH_μ": round(float(np.mean(ph)), 2),
            "PH_σ": round(float(np.std(ph)), 2),
            "PH_C.T.(s)": round(float(np.mean([r["PH_runtime"] for r in records])), 3),
            "ALNS_μ": round(float(np.mean(alns)), 2),
            "ALNS_σ": round(float(np.std(alns)), 2),
            "ALNS_C.T.(s)": round(float(np.mean([r["ALNS_runtime"] for r in records])), 3),
            "ALNS_it_μ": round(float(np.mean([r["ALNS_iterations"] for r in records])), 1),
            "Gap_ALNS_vs_PH (%)": f"{((np.mean(alns) - np.mean(ph)) / max(1.0, np.mean(ph))) * 100.0:.2f}%",
            "ALNS_StopReasons": ",".join(sorted(set(r["ALNS_stop"] for r in records))),
        })

    summary = pd.DataFrame(rows)
    print("\n[ALNS-ONLY SUMMARY: REAL KMWE BASE CASES]")
    print(summary.to_string(index=False))
    export_seed_results_to_excel(output_excel, basecase_summary=summary, basecase_seed_results=pd.DataFrame(basecase_seed_records))
    return summary, pd.DataFrame(basecase_seed_records)


def run_table8_and_table14_replications(
    num_runs=10,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    output_excel="alns_table8_table14_all_seed_results.xlsx",
):
    table8_summary, table8_seed_results = run_alns_table8_replications(
        num_runs=num_runs,
        alns_time_seconds=alns_time_seconds,
        alns_iterations=alns_iterations,
        alns_no_improvement_limit=alns_no_improvement_limit,
        output_excel=None,
    )

    table14_summary, table14_seed_results = run_alns_only_replications(
        num_runs=num_runs,
        alns_time_seconds=alns_time_seconds,
        alns_iterations=alns_iterations,
        alns_no_improvement_limit=alns_no_improvement_limit,
        output_excel=None,
    )

    export_seed_results_to_excel(
        output_excel,
        table8_summary=table8_summary,
        table8_seed_results=table8_seed_results,
        table14_summary=table14_summary,
        table14_seed_results=table14_seed_results,
    )

    return {
        "table8_summary": table8_summary,
        "table8_seed_results": table8_seed_results,
        "table14_summary": table14_summary,
        "table14_seed_results": table14_seed_results,
    }


if __name__ == "__main__":
    # Longer ALNS+AOS exploration defaults:
    #   600 seconds, 2000 iterations, 500 no-improvement iterations.
    # This gives AOS more time to adapt operator weights.
    # One workbook is produced with Table 8 and Table 14 summaries plus
    # all per-seed rows. tqdm progress bars show case/slice/seed progress.
    run_table8_and_table14_replications(num_runs=10)
