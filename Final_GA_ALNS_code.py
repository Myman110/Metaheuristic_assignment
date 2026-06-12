import os
import random
import time
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # Fallback so the script still runs if tqdm is not installed.
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class _DummyTqdm:
                def update(self, *a, **k): pass
                def set_postfix(self, *a, **k): pass
                def close(self): pass
            return _DummyTqdm()
        return iterable

# Global cache for tool sizes to optimize search speed in the knapsack solver
GLOBAL_TOOL_SIZES = {}


# =====================================================================
# Seed-level Excel export helpers
# =====================================================================
def _safe_sheet_name(name):
    return str(name)[:31].replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_").replace("[", "(").replace("]", ")").replace(":", "-")

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
            df.to_excel(writer, sheet_name=_safe_sheet_name(sheet_name), index=False)
            ws = writer.sheets[_safe_sheet_name(sheet_name)]
            for col_cells in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 38)
    print(f"\n[EXCEL EXPORT] Seed-level results written to: {output_path}")
    return output_path

# =====================================================================
# Paper-exact parameter settings from Dang et al. (2021)
# =====================================================================
PAPER_B = 1
PAPER_POP_SIZE = 100
PAPER_ELITISM_RATE = 0.10
PAPER_UNIFORM_MUTATION_PROB = 0.01
PAPER_SWAP_MUTATION_PROB = 0.01
PAPER_TOURNAMENT_RATE = 0.20
PAPER_NO_IMPROVEMENT_LIMIT = 20
PAPER_MAX_TIME_SECONDS = 3600.0
PAPER_SETUP_TIME = 1.0
PAPER_THETA_M = 72.0


# =====================================================================
# 1. THE EXACT TOOL REPLACEMENT METHOD (TRM) SOLVER
# =====================================================================
def solve_trm_ilp_exact(tools_in_magazine, tool_sizes, scores, needed_capacity):
    """
    Exact ILP from Dang et al. (2021), Section 5.6, equations (23)-(25).

    Variables:
        lambda_t = 1 if tool set t is removed, 0 otherwise.

    Model:
        min  sum_{t in TM_m} sc_t * lambda_t
        s.t. sum_{t in TM_m} phi_t * lambda_t >= phi_m^S
             lambda_t in {0,1}

    The magazine only contains a small number of tool sets in practice,
    so complete enumeration is a transparent exact solver for this tiny ILP.
    We include deterministic tie-breaks so repeated runs are reproducible
    for the same random seed.
    """
    if needed_capacity <= 0:
        return []

    tools = sorted(list(tools_in_magazine))
    n = len(tools)
    best_key = None
    best_subset = []

    # Enumerate all non-empty removal subsets. This directly mirrors the
    # paper's discussion that the number of combinations is 2^|TM_m| - 1.
    for mask in range(1, 1 << n):
        subset = [tools[i] for i in range(n) if mask & (1 << i)]
        freed_capacity = sum(tool_sizes[t] for t in subset)
        if freed_capacity < needed_capacity:
            continue

        objective = sum(scores.get(t, 0) for t in subset)
        # Primary objective: min score. Deterministic tie-breaks: min extra
        # freed capacity, then fewer removals, then lexicographic subset.
        key = (objective, freed_capacity, len(subset), tuple(subset))
        if best_key is None or key < best_key:
            best_key = key
            best_subset = subset

    return best_subset


# Backward-compatible name used by the decoder.
def solve_trm_knapsack(tools_in_magazine, tool_sizes, scores, needed_capacity):
    return solve_trm_ilp_exact(tools_in_magazine, tool_sizes, scores, needed_capacity)


# =====================================================================
# 2. CHROMOSOME INDIVIDUAL AND DECODER
# =====================================================================
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


# =====================================================================
# 3. PRACTITIONER HEURISTIC ENGINE
# =====================================================================
class PractitionerHeuristic:
    def __init__(self, jobs_data, num_machines, magazine_capacity, tool_setup_time=PAPER_SETUP_TIME, theta_m=PAPER_THETA_M):
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




# =====================================================================
# 4. ALNS WITH ADAPTIVE OPERATOR SELECTION AND TRP
# =====================================================================
class ALNS_AOS:
    """
    Improved ALNS-AOS version.

    Improvements over the original:
        1. Precedence-safe candidate insertion.
        2. Setup-aware destroy using actual decoder setup positions.
        3. Reward scaling by improvement magnitude.
        4. Candidate-list search retained to avoid runtime explosion.
        5. Optional deep repair every N iterations.

    The Decoder still performs the full schedule simulation and TRP/TRM logic.
    ALNS only searches operation order and machine assignment neighborhoods.
    """

    def __init__(
        self,
        jobs_data,
        num_machines,
        magazine_capacity,
        setup_time=PAPER_SETUP_TIME,
        reaction_factor=0.20,
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
        reward_scale=0.05,
    ):
        self.jobs_data = jobs_data
        self.num_machines = num_machines
        self.C = magazine_capacity
        self.tau = setup_time
        self.reaction_factor = reaction_factor
        self.destroy_fraction = destroy_fraction
        self.temperature = start_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_insert_positions = max_insert_positions
        self.max_machine_candidates = max_machine_candidates
        self.max_removed_jobs = max_removed_jobs
        self.cache_evaluations = cache_evaluations
        self.deep_repair_period = deep_repair_period
        self.deep_insert_positions = deep_insert_positions
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

        self.total_ops_by_job = {job_id: len(ops) for job_id, ops in self.ops_by_job.items()}
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
            "alns_repair_weights",
        ]:
            if hasattr(ind, attr):
                setattr(new, attr, getattr(ind, attr))

        return new

    def _evaluate(self, job_vec, mach_vec):
        # Full evaluation is expensive because Decoder.evaluate also solves TRP.
        # Cache exact schedule evaluations so repeated insertions do not redo TRP.
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

    def _sample_operator(self, weights):
        names = list(weights.keys())
        vals = np.array([max(1e-12, weights[n]) for n in names], dtype=float)
        probs = vals / vals.sum()
        return str(np.random.choice(names, p=probs))

    def _update_weight(self, weights, name, reward):
        rho = self.reaction_factor
        weights[name] = (1.0 - rho) * weights[name] + rho * reward
        weights[name] = max(0.05, weights[name])

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
        partial_jobs = [v for i, v in enumerate(individual.job_vector) if i not in positions]
        partial_machs = [v for i, v in enumerate(individual.machine_vector) if i not in positions]
        return partial_jobs, partial_machs, removed_jobs

    # -----------------------------
    # Precedence-safe insertion logic
    # -----------------------------
    def _op_index_for_next_insertion(self, partial_jobs, job_id):
        """
        The next inserted occurrence of job_id must be the first missing operation
        of that job. This replaces the unsafe fallback that clipped the occurrence
        index to the last operation.
        """
        occ = partial_jobs.count(job_id)

        if occ >= self.total_ops_by_job[job_id]:
            raise ValueError(f"Cannot insert job {job_id}: all operations already present.")

        return occ

    def _operation_data_for_insertion(self, partial_jobs, job_id):
        occ = self._op_index_for_next_insertion(partial_jobs, job_id)
        return self.ops_by_job[job_id][occ]

    def _precedence_safe_positions(self, partial_jobs, job_id):
        """
        Since the chromosome stores repeated job IDs, operation precedence is
        represented by occurrence order. The next missing operation of a job must
        be placed after all existing occurrences of that job.
        """
        last_same_job_pos = -1
        for idx, job in enumerate(partial_jobs):
            if job == job_id:
                last_same_job_pos = idx

        earliest = last_same_job_pos + 1
        latest = len(partial_jobs)
        return list(range(earliest, latest + 1))

    def _position_candidates(self, partial_jobs, job_id, deep=False):
        # Candidate-list ALNS: retain a capped candidate set, but only from
        # precedence-safe positions.
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
            positions.update(random.sample(remaining, min(remaining_budget, len(remaining))))

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

        # Prefer a machine already using the same tool, if any.
        same_tool = [m for m, count in tool_presence.items() if count > 0]
        if same_tool:
            candidates.append(max(same_tool, key=lambda m: (tool_presence[m], -loads[m])))

        # Then prefer least-loaded machines.
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
        candidate_positions = [i for i, m in enumerate(individual.machine_vector) if m == overloaded]

        if len(candidate_positions) < q:
            extra = [i for i in range(len(individual.job_vector)) if i not in candidate_positions]
            candidate_positions += random.sample(extra, min(len(extra), q - len(candidate_positions)))

        positions = random.sample(candidate_positions, q)
        return self._remove_positions(individual, positions)

    def destroy_actual_setup_removal(self, individual, q):
        """
        Uses actual setup positions recorded by Decoder.evaluate instead of the
        older proxy based on consecutive tool changes.
        """
        if not hasattr(individual, "setup_positions"):
            self.decoder.evaluate(individual)

        setup_positions = list(getattr(individual, "setup_positions", []))

        if len(setup_positions) >= q:
            positions = random.sample(setup_positions, q)
            return self._remove_positions(individual, positions)

        positions = list(setup_positions)
        remaining = [i for i in range(len(individual.job_vector)) if i not in positions]
        if remaining:
            positions += random.sample(remaining, min(len(remaining), q - len(positions)))

        return self._remove_positions(individual, positions)

    # -----------------------------
    # Repair operators
    # -----------------------------
    def _best_single_insertion(self, partial_jobs, partial_machs, job_id, machine_candidates=None, deep=False):
        if machine_candidates is None:
            machine_candidates = self._machine_candidates(partial_jobs, partial_machs, job_id, deep=deep)

        best = None
        for pos in self._position_candidates(partial_jobs, job_id, deep=deep):
            for mach in machine_candidates:
                trial_jobs = partial_jobs[:pos] + [job_id] + partial_jobs[pos:]
                trial_machs = partial_machs[:pos] + [mach] + partial_machs[pos:]
                cand = self._evaluate(trial_jobs, trial_machs)
                key = (cand.fitness, cand.tardiness, cand.setups, pos, mach)
                if best is None or key < best[0]:
                    best = (key, cand)

        return best[1]

    def _is_deep_repair_iteration(self):
        return self.deep_repair_period > 0 and self.current_iteration % self.deep_repair_period == 0

    def repair_greedy_best_insert(self, partial_jobs, partial_machs, removed_jobs):
        jobs = list(removed_jobs)
        random.shuffle(jobs)
        current_jobs, current_machs = list(partial_jobs), list(partial_machs)
        deep = self._is_deep_repair_iteration()

        for job_id in jobs:
            best = self._best_single_insertion(current_jobs, current_machs, job_id, deep=deep)
            current_jobs, current_machs = best.job_vector, best.machine_vector

        return self._evaluate(current_jobs, current_machs)

    def repair_regret2_insert(self, partial_jobs, partial_machs, removed_jobs):
        remaining = list(removed_jobs)
        current_jobs, current_machs = list(partial_jobs), list(partial_machs)
        deep = self._is_deep_repair_iteration()

        while remaining:
            best_choice = None

            for job_id in remaining:
                candidates = []
                for pos in self._position_candidates(current_jobs, job_id, deep=deep):
                    for mach in self._machine_candidates(current_jobs, current_machs, job_id, deep=deep):
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
            current_jobs, current_machs = chosen_ind.job_vector, chosen_ind.machine_vector
            remaining.remove(chosen_job)

        return self._evaluate(current_jobs, current_machs)

    def repair_edd_insert(self, partial_jobs, partial_machs, removed_jobs):
        current_jobs, current_machs = list(partial_jobs), list(partial_machs)
        jobs = list(removed_jobs)
        deep = self._is_deep_repair_iteration()

        jobs.sort(key=lambda j: self._operation_data_for_insertion(current_jobs, j)["d"])

        for job_id in jobs:
            op = self._operation_data_for_insertion(current_jobs, job_id)
            due = float(op["d"])
            candidate_positions = self._position_candidates(current_jobs, job_id, deep=deep)
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
            for mach in self._machine_candidates(current_jobs, current_machs, job_id, deep=deep):
                trial_jobs = current_jobs[:pos] + [job_id] + current_jobs[pos:]
                trial_machs = current_machs[:pos] + [mach] + current_machs[pos:]
                cand = self._evaluate(trial_jobs, trial_machs)
                key = (cand.fitness, cand.tardiness, cand.setups, mach)
                if best is None or key < best[0]:
                    best = (key, cand)

            current_jobs, current_machs = best[1].job_vector, best[1].machine_vector

        return self._evaluate(current_jobs, current_machs)

    def repair_least_loaded_insert(self, partial_jobs, partial_machs, removed_jobs):
        current_jobs, current_machs = list(partial_jobs), list(partial_machs)

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
            current_jobs, current_machs = best.job_vector, best.machine_vector

        return self._evaluate(current_jobs, current_machs)

    # -----------------------------
    # Reward logic
    # -----------------------------
    def _compute_reward(self, candidate, current, best, accepted):
        """
        Scaled reward: keeps the original categories but adds a small bonus
        proportional to relative improvement magnitude.
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
            improvement = 0.0
            reference = 1.0
        else:
            return 0.1

        scaled_bonus = self.reward_scale * 100.0 * max(0.0, improvement / reference)
        return base + scaled_bonus

    # -----------------------------
    # Main ALNS-AOS loop
    # -----------------------------
    def run(
        self,
        initial_solution,
        max_iterations=250,
        max_time_seconds=60.0,
        no_improvement_limit=50,
        record_history=True,
        verbose=False,
        show_progress=False,
        progress_desc="ALNS-AOS",
    ):
        start_clock = time.time()
        current = self.clone(initial_solution)
        self.decoder.evaluate(current)  # Ensure initial solution has TRP-consistent objective and setup trace.
        best = self.clone(current)

        if self.temperature is None:
            self.temperature = max(1.0, 0.05 * abs(current.fitness))

        history = []
        no_improve = 0
        stop_reason = "iteration_limit"

        iterator = tqdm(
            range(1, max_iterations + 1),
            desc=progress_desc,
            leave=False,
            disable=not show_progress,
        )

        for it in iterator:
            self.current_iteration = it
            elapsed = time.time() - start_clock

            if elapsed >= max_time_seconds:
                stop_reason = "time_limit"
                break
            if no_improve >= no_improvement_limit:
                stop_reason = "no_improvement_limit"
                break

            destroy_name = self._sample_operator(self.destroy_weights)
            repair_name = self._sample_operator(self.repair_weights)
            q = self._num_to_remove(len(current.job_vector))

            partial_jobs, partial_machs, removed_jobs = self.destroy_ops[destroy_name](current, q)
            candidate = self.repair_ops[repair_name](partial_jobs, partial_machs, removed_jobs)

            accepted = self._accept(candidate, current)
            improved_current = candidate.fitness < current.fitness
            improved_best = candidate.fitness < best.fitness

            reward = self._compute_reward(candidate, current, best, accepted)

            if accepted:
                current = candidate

            if improved_best:
                best = self.clone(candidate)
                no_improve = 0
            else:
                no_improve += 1

            self._update_weight(self.destroy_weights, destroy_name, reward)
            self._update_weight(self.repair_weights, repair_name, reward)
            self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)

            if show_progress and (it == 1 or it % 10 == 0 or improved_best):
                iterator.set_postfix(
                    best=round(float(best.fitness), 2),
                    curr=round(float(current.fitness), 2),
                    noimp=int(no_improve),
                    temp=round(float(self.temperature), 3),
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
                    "q_removed": int(q),
                    "reward": float(reward),
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
                    f"acc={accepted} d={destroy_name} r={repair_name} "
                    f"reward={reward:.3f} temp={self.temperature:.4f} "
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
        return best




# =====================================================================
# REAL-WORLD DATA LOADER & PARSER
# =====================================================================
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





# =====================================================================
# ALNS-ONLY EXPERIMENT ENGINE
# =====================================================================
def run_alns_only_on_file(
    case_file,
    seed=0,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    verbose=False,
    show_progress=False,
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

    decoder = Decoder(ops_by_job, m_case, c_case, PAPER_SETUP_TIME)
    decoder.evaluate(ph_solution)

    alns_engine = ALNS_AOS(jobs_data, m_case, c_case, PAPER_SETUP_TIME)
    alns_solution = alns_engine.run(
        ph_solution,
        max_time_seconds=alns_time_seconds,
        max_iterations=alns_iterations,
        no_improvement_limit=alns_no_improvement_limit,
        record_history=True,
        verbose=verbose,
        show_progress=show_progress,
        progress_desc=f"ALNS-only seed={seed}",
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

    print(pd.DataFrame([result]).to_string(index=False))
    return alns_solution, result



def run_alns_table8_replications(
    num_runs=10,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
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

    for n_slice in [15, 25, 30, 60, 90, 120, 140]:
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        records = []

        for seed in range(num_runs):
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

            decoder = Decoder(ops_by_job, m_val, c_val, PAPER_SETUP_TIME)
            decoder.evaluate(ph_solution)

            alns_engine = ALNS_AOS(sliced_ops, m_val, c_val, PAPER_SETUP_TIME)
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
                "PH_fitness": ph_solution.fitness,
                "PH_runtime": ph_runtime,
                "ALNS_fitness": alns_solution.fitness,
                "ALNS_runtime": alns_solution.alns_runtime,
                "ALNS_iterations": alns_solution.alns_iterations,
                "ALNS_stop": alns_solution.alns_stop_reason,
            })

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
    return summary


def run_alns_only_replications(
    num_runs=10,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
):
    """
    Run ALNS-only experiments on real KMWE cases.

    PH   = Practitioner Heuristic initial solution
    ALNS = ALNS-AOS+TRP initialized from PH

    Synthetic/mock data is disabled.
    """
    print("=" * 120)
    print(f" DEPRECATED ALNS-ONLY ENGINE: use run_ph_ga_mh_alns_ilp_on_file instead ({num_runs} SEED SAMPLES) ".center(120, "#"))
    print("=" * 120)

    rows = []

    for case_name in ["2M38", "2M46", "6M140", "6M163"]:
        case_file = resolve_kmwe_case_file(case_name)
        records = []

        for seed in range(num_runs):
            _, result = run_alns_only_on_file(
                case_file,
                seed=seed,
                alns_time_seconds=alns_time_seconds,
                alns_iterations=alns_iterations,
                alns_no_improvement_limit=alns_no_improvement_limit,
                verbose=False,
            )
            records.append(result)

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
    return summary


# =====================================================================
# 4. CROSSOVER & MUTATION HELPER FUNCTIONS
# =====================================================================
def transform_apmx(p1, p2):
    p1_occ = {}
    mapping = {}
    for i, job in enumerate(p1):
        occ = p1_occ.get(job, 0) + 1
        p1_occ[job] = occ
        mapping[(job, occ)] = i + 1
        
    p2_occ = {}
    tp2 = []
    for job in p2:
        occ = p2_occ.get(job, 0) + 1
        p2_occ[job] = occ
        tp2.append(mapping[(job, occ)])
    return list(range(1, len(p1) + 1)), tp2

def pmx_crossover(p1, p2):
    n = len(p1)
    cx1 = random.randint(0, n - 2)
    cx2 = random.randint(cx1 + 1, n - 1)
    child1, child2 = [None]*n, [None]*n
    child1[cx1:cx2+1] = p2[cx1:cx2+1]
    child2[cx1:cx2+1] = p1[cx1:cx2+1]
    
    map1 = {p2[i]: p1[i] for i in range(cx1, cx2 + 1)}
    map2 = {p1[i]: p2[i] for i in range(cx1, cx2 + 1)}
    
    for i in range(n):
        if i < cx1 or i > cx2:
            val1 = p1[i]
            while val1 in map1: val1 = map1[val1]
            child1[i] = val1
            val2 = p2[i]
            while val2 in map2: val2 = map2[val2]
            child2[i] = val2
    return child1, child2, cx1, cx2

def apply_pox_edd(job_vector, cx1, cx2, ops_by_job):
    n = len(job_vector)
    occ_counts = {}
    outside_elements = []
    for idx, job in enumerate(job_vector):
        occ = occ_counts.get(job, 0)
        occ_counts[job] = occ + 1
        if idx < cx1 or idx > cx2:
            outside_elements.append((job, ops_by_job[job][occ], idx))
            
    outside_elements.sort(key=lambda x: x[1]['d'])
    new_job_vector = list(job_vector)
    outside_indices = [i for i in range(n) if i < cx1 or i > cx2]
    for idx, (job, _, _) in zip(outside_indices, outside_elements):
        new_job_vector[idx] = job
    return new_job_vector

def build_pox_machine_vector(job_vector, ops_by_job, num_machines, magazine_capacity):
    T_m = {m: set() for m in range(1, num_machines + 1)}
    p_m = {m: 0.0 for m in range(1, num_machines + 1)}
    mach_vec = []
    occ_counts = {}
    
    for job in job_vector:
        occ = occ_counts.get(job, 0)
        occ_counts[job] = occ + 1
        op_data = ops_by_job[job][occ]
        t_ij, phi_t, p_ij = op_data['tool_set'], op_data['size'], op_data['p']
        
        m_T_list = [m for m in range(1, num_machines + 1) if t_ij in T_m[m]]
        if m_T_list:
            m_star = m_T_list[0]
        else:
            M_C = []
            for m in range(1, num_machines + 1):
                current_size = sum(GLOBAL_TOOL_SIZES[t] for t in T_m[m])
                if magazine_capacity - current_size >= phi_t:
                    M_C.append(m)
            m_star = min(M_C, key=lambda m: p_m[m]) if M_C else min(range(1, num_machines + 1), key=lambda m: p_m[m])
                
        p_m[m_star] += p_ij
        mach_vec.append(m_star)
    return mach_vec


# =====================================================================
# 5. MATHEURISTIC RESOLUTION MODEL (MH)
# =====================================================================
class Matheuristic:
    def __init__(self, jobs_data, num_machines, magazine_capacity, setup_time=PAPER_SETUP_TIME):
        self.num_machines = num_machines
        self.C = magazine_capacity
        self.tau = setup_time
        self.jobs_data = jobs_data
        
        self.ops_by_job = {}
        self.flat_ops = []
        for op in jobs_data:
            job_id = int(op['job_id'])
            if job_id not in self.ops_by_job:
                self.ops_by_job[job_id] = []
            self.ops_by_job[job_id].append(op)
            self.flat_ops.append(job_id)
            
        for job_id in self.ops_by_job:
            self.ops_by_job[job_id].sort(key=lambda x: x['op_id'])
            
        self.decoder = Decoder(self.ops_by_job, self.num_machines, self.C, self.tau)

    def run(
        self,
        max_time_seconds=PAPER_MAX_TIME_SECONDS,
        no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
        pop_size=PAPER_POP_SIZE,
        tournament_rate=PAPER_TOURNAMENT_RATE,
        elitism_rate=PAPER_ELITISM_RATE,
        swap_mutation_prob=PAPER_SWAP_MUTATION_PROB,
        uniform_mutation_prob=PAPER_UNIFORM_MUTATION_PROB,
        pox_iterations=PAPER_B,
        max_generations=None,
        record_history=True,
        verbose=False,
    ):
        """
        Paper-aligned matheuristic loop.

        Dang et al. stop after either maxTime=3600 seconds or Gc=20
        consecutive generations without improvement. `max_generations` is
        optional and is only for debugging; leave it as None for the paper
        stopping rule.
        """
        population = []
        ph_engine = PractitionerHeuristic(self.jobs_data, self.num_machines, self.C, self.tau)
        ph_baseline = ph_engine.run()
        self.decoder.evaluate(ph_baseline)
        population.append(ph_baseline)

        n_ops = len(self.flat_ops)
        for _ in range(pop_size - 1):
            rand_job_vec = list(self.flat_ops)
            random.shuffle(rand_job_vec)
            rand_mach_vec = [random.randint(1, self.num_machines) for _ in range(n_ops)]
            ind = Individual(rand_job_vec, rand_mach_vec)
            self.decoder.evaluate(ind)
            population.append(ind)

        best_ind = min(population, key=lambda x: x.fitness)
        f_best = best_ind.fitness
        no_improve = 0
        q = pox_iterations + 1
        best_improved = False
        generation = 0
        start_clock = time.time()
        history = []
        stop_reason = None

        # Store generation 0 so we can see whether the starting population
        # already contains the final solution.
        if record_history:
            history.append({
                "generation": 0,
                "runtime": 0.0,
                "best_fitness": float(f_best),
                "current_best": float(f_best),
                "no_improve": 0,
                "used_pox": False,
                "improved": True,
            })

        while True:
            elapsed = time.time() - start_clock
            if elapsed >= max_time_seconds:
                stop_reason = "time_limit"
                break
            if no_improve >= no_improvement_limit:
                stop_reason = "no_improvement_limit"
                break
            if max_generations is not None and generation >= max_generations:
                stop_reason = "debug_generation_limit"
                break

            offspring = []
            use_pox = best_improved or q <= pox_iterations

            while len(offspring) < pop_size:
                p1, p2 = self.select_parents(population, tournament_rate)

                if use_pox:
                    tp1, tp2 = transform_apmx(p1.job_vector, p2.job_vector)
                    tc1, tc2, cx1, cx2 = pmx_crossover(tp1, tp2)
                    c1_job = [p1.job_vector[v - 1] for v in tc1]
                    c2_job = [p1.job_vector[v - 1] for v in tc2]
                    c1_job = apply_pox_edd(c1_job, cx1, cx2, self.ops_by_job)
                    c2_job = apply_pox_edd(c2_job, cx1, cx2, self.ops_by_job)
                    c1_mach = build_pox_machine_vector(c1_job, self.ops_by_job, self.num_machines, self.C)
                    c2_mach = build_pox_machine_vector(c2_job, self.ops_by_job, self.num_machines, self.C)
                else:
                    tp1, tp2 = transform_apmx(p1.job_vector, p2.job_vector)
                    tc1, tc2, _, _ = pmx_crossover(tp1, tp2)
                    c1_job = [p1.job_vector[v - 1] for v in tc1]
                    c2_job = [p1.job_vector[v - 1] for v in tc2]
                    c1_mach, c2_mach = self.two_point_crossover(p1.machine_vector, p2.machine_vector)

                child1, child2 = Individual(c1_job, c1_mach), Individual(c2_job, c2_mach)
                self.apply_mutation(child1, swap_mutation_prob, uniform_mutation_prob)
                self.apply_mutation(child2, swap_mutation_prob, uniform_mutation_prob)
                self.decoder.evaluate(child1)
                self.decoder.evaluate(child2)
                offspring.extend([child1, child2])

            offspring = offspring[:pop_size]

            # Paper elitism: choose the best SE parents, then randomly replace
            # SE offspring by those parents.
            se = int(elitism_rate * pop_size)
            parents_elite = sorted(population, key=lambda x: x.fitness)[:se]
            next_pop = offspring[:]
            if se > 0:
                replace_idx = random.sample(range(pop_size), se)
                for idx, elite in zip(replace_idx, parents_elite):
                    next_pop[idx] = elite

            # Immigration: replace duplicate chromosomes by random individuals.
            unique_signatures = set()
            final_pop = []
            for ind in next_pop:
                sig = ind.get_signature()
                if sig not in unique_signatures:
                    unique_signatures.add(sig)
                    final_pop.append(ind)
                else:
                    rand_job_vec = list(self.flat_ops)
                    random.shuffle(rand_job_vec)
                    rand_mach_vec = [random.randint(1, self.num_machines) for _ in range(n_ops)]
                    immigrant = Individual(rand_job_vec, rand_mach_vec)
                    self.decoder.evaluate(immigrant)
                    final_pop.append(immigrant)

            population = final_pop
            current_best = min(population, key=lambda x: x.fitness)
            improved = current_best.fitness < f_best
            if improved:
                f_best = current_best.fitness
                best_ind = current_best
                best_improved = True
                q = 1
                no_improve = 0
            else:
                best_improved = False
                q += 1
                no_improve += 1

            generation += 1
            elapsed = time.time() - start_clock

            if record_history:
                history.append({
                    "generation": generation,
                    "runtime": float(elapsed),
                    "best_fitness": float(f_best),
                    "current_best": float(current_best.fitness),
                    "no_improve": int(no_improve),
                    "used_pox": bool(use_pox),
                    "improved": bool(improved),
                })

            if verbose:
                print(
                    f"gen={generation:4d} best={f_best:.4f} "
                    f"current={current_best.fitness:.4f} "
                    f"no_improve={no_improve:2d} pox={use_pox} "
                    f"time={elapsed:.2f}s"
                )

        if stop_reason is None:
            stop_reason = "unknown"

        best_ind.generations = generation
        best_ind.runtime = time.time() - start_clock
        best_ind.stop_reason = stop_reason
        best_ind.history = history
        return best_ind

    def select_parents(self, population, tournament_rate):
        size = max(2, int(tournament_rate * len(population)))
        return min(random.sample(population, size), key=lambda x: x.fitness), min(random.sample(population, size), key=lambda x: x.fitness)

    def two_point_crossover(self, m1, m2):
        n = len(m1)
        cx1 = random.randint(0, n - 2)
        cx2 = random.randint(cx1 + 1, n - 1)
        c1, c2 = list(m1), list(m2)
        c1[cx1:cx2+1], c2[cx1:cx2+1] = m2[cx1:cx2+1], m1[cx1:cx2+1]
        return c1, c2

    def apply_mutation(self, individual, Ps, Pu):
        # Paper swap mutation: each job-vector gene is selected with
        # probability Ps and swapped with a randomly chosen gene.
        n = len(individual.job_vector)
        for idx in range(n):
            if random.random() < Ps:
                j = random.randrange(n)
                individual.job_vector[idx], individual.job_vector[j] = individual.job_vector[j], individual.job_vector[idx]

        # Paper uniform mutation: each machine-vector gene is selected
        # with probability Pu and replaced by U{1,...,|M|}.
        for idx in range(len(individual.machine_vector)):
            if random.random() < Pu:
                individual.machine_vector[idx] = random.randint(1, self.num_machines)


# =====================================================================
# 7. STRICT PH -> GA/MH -> ALNS-AOS -> ILP HYBRID ENGINE
# =====================================================================

def clone_individual(ind):
    """Create a safe copy of an Individual, including common result attributes."""
    new = Individual(ind.job_vector, ind.machine_vector)
    new.fitness = float(getattr(ind, "fitness", float("inf")))
    new.tardiness = float(getattr(ind, "tardiness", 0.0))
    new.setups = int(getattr(ind, "setups", 0))
    for attr in [
        "runtime", "generations", "stop_reason", "history", "setup_positions",
        "alns_iterations", "alns_runtime", "alns_stop_reason", "alns_history",
        "alns_destroy_weights", "alns_repair_weights", "alns_eval_cache_hits",
        "alns_eval_cache_misses", "source_label", "source_rank", "source_diversity",
    ]:
        if hasattr(ind, attr):
            setattr(new, attr, getattr(ind, attr))
    return new


def individual_distance(a, b):
    """
    Normalized distance between two chromosomes.
    50% job-order Hamming distance + 50% machine-assignment Hamming distance.
    """
    n = max(1, len(a.job_vector))
    job_diff = sum(x != y for x, y in zip(a.job_vector, b.job_vector)) / n
    mach_diff = sum(x != y for x, y in zip(a.machine_vector, b.machine_vector)) / n
    return 0.5 * job_diff + 0.5 * mach_diff


def unique_sorted_individuals(individuals):
    """Remove duplicate chromosomes and sort by objective value."""
    unique = {}
    for ind in individuals:
        sig = ind.get_signature()
        if sig not in unique or ind.fitness < unique[sig].fitness:
            unique[sig] = clone_individual(ind)
    return sorted(unique.values(), key=lambda x: (x.fitness, x.tardiness, x.setups))


def select_three_diverse_ga_solutions(
    archive,
    k=3,
    near_best_gap=0.10,
    min_candidate_pool=25,
):
    """
    Select the best GA/MH solution plus k-1 high-quality but structurally diverse solutions.

    Rule:
      1. Always choose the best solution.
      2. Prefer candidates within `near_best_gap` of the best objective.
      3. If too few candidates are inside that gap, fall back to the best `min_candidate_pool` unique solutions.
      4. Greedily add the candidate with the largest minimum chromosome distance to the already chosen set.
    """
    ranked = unique_sorted_individuals(archive)
    if not ranked:
        raise ValueError("Cannot select GA solutions: archive is empty.")

    best = ranked[0]
    threshold = best.fitness + max(abs(best.fitness) * near_best_gap, 1e-9)
    pool = [ind for ind in ranked if ind.fitness <= threshold]

    if len(pool) < k:
        pool = ranked[:max(k, min(min_candidate_pool, len(ranked)))]

    chosen = [clone_individual(best)]
    while len(chosen) < min(k, len(pool)):
        best_candidate = None
        best_key = None
        for cand in pool:
            if cand.get_signature() in {c.get_signature() for c in chosen}:
                continue
            min_dist = min(individual_distance(cand, c) for c in chosen)
            # Primary: diversity. Secondary: objective quality.
            key = (min_dist, -cand.fitness, -cand.tardiness, -cand.setups)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = cand
        if best_candidate is None:
            break
        copied = clone_individual(best_candidate)
        copied.source_diversity = float(best_key[0])
        chosen.append(copied)

    for idx, ind in enumerate(chosen, start=1):
        ind.source_rank = idx
        ind.source_label = "GA_best" if idx == 1 else f"GA_diverse_{idx}"
    return chosen


def select_best_ga_solution(archive):
    """Return only the single best unique GA/MH solution for the ALNS start."""
    ranked = unique_sorted_individuals(archive)
    if not ranked:
        raise ValueError("Cannot select GA solution: archive is empty.")

    best = clone_individual(ranked[0])
    best.source_rank = 1
    best.source_label = "GA_best"
    best.source_diversity = 0.0
    return [best]


class MatheuristicWithArchive(Matheuristic):
    """
    Same PH + GA/MH structure as the paper-replication Matheuristic, but it also
    stores an archive of high-quality GA/MH solutions so ALNS-AOS can be started
    from the best solution and two near-best diverse alternatives.
    """

    def run_with_archive(
        self,
        max_time_seconds=PAPER_MAX_TIME_SECONDS,
        no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
        pop_size=PAPER_POP_SIZE,
        tournament_rate=PAPER_TOURNAMENT_RATE,
        elitism_rate=PAPER_ELITISM_RATE,
        swap_mutation_prob=PAPER_SWAP_MUTATION_PROB,
        uniform_mutation_prob=PAPER_UNIFORM_MUTATION_PROB,
        pox_iterations=PAPER_B,
        max_generations=None,
        record_history=True,
        verbose=False,
        archive_top_per_generation=25,
        show_progress=False,
        progress_desc="GA/MH",
    ):
        population = []
        archive = []

        ph_engine = PractitionerHeuristic(self.jobs_data, self.num_machines, self.C, self.tau)
        ph_baseline = ph_engine.run()
        self.decoder.evaluate(ph_baseline)
        population.append(ph_baseline)
        archive.append(clone_individual(ph_baseline))

        n_ops = len(self.flat_ops)
        for _ in range(pop_size - 1):
            rand_job_vec = list(self.flat_ops)
            random.shuffle(rand_job_vec)
            rand_mach_vec = [random.randint(1, self.num_machines) for _ in range(n_ops)]
            ind = Individual(rand_job_vec, rand_mach_vec)
            self.decoder.evaluate(ind)
            population.append(ind)

        archive.extend(clone_individual(ind) for ind in sorted(population, key=lambda x: x.fitness)[:archive_top_per_generation])

        best_ind = min(population, key=lambda x: x.fitness)
        f_best = best_ind.fitness
        no_improve = 0
        q = pox_iterations + 1
        best_improved = False
        generation = 0
        start_clock = time.time()
        history = []
        stop_reason = None

        pbar = tqdm(
            total=max_generations if max_generations is not None else None,
            desc=progress_desc,
            leave=False,
            disable=not show_progress,
        )

        if record_history:
            history.append({
                "generation": 0,
                "runtime": 0.0,
                "best_fitness": float(f_best),
                "current_best": float(f_best),
                "no_improve": 0,
                "used_pox": False,
                "improved": True,
                "archive_size": len(unique_sorted_individuals(archive)),
            })

        while True:
            elapsed = time.time() - start_clock
            if elapsed >= max_time_seconds:
                stop_reason = "time_limit"
                break
            if no_improve >= no_improvement_limit:
                stop_reason = "no_improvement_limit"
                break
            if max_generations is not None and generation >= max_generations:
                stop_reason = "debug_generation_limit"
                break

            offspring = []
            use_pox = best_improved or q <= pox_iterations

            while len(offspring) < pop_size:
                p1, p2 = self.select_parents(population, tournament_rate)

                if use_pox:
                    tp1, tp2 = transform_apmx(p1.job_vector, p2.job_vector)
                    tc1, tc2, cx1, cx2 = pmx_crossover(tp1, tp2)
                    c1_job = [p1.job_vector[v - 1] for v in tc1]
                    c2_job = [p1.job_vector[v - 1] for v in tc2]
                    c1_job = apply_pox_edd(c1_job, cx1, cx2, self.ops_by_job)
                    c2_job = apply_pox_edd(c2_job, cx1, cx2, self.ops_by_job)
                    c1_mach = build_pox_machine_vector(c1_job, self.ops_by_job, self.num_machines, self.C)
                    c2_mach = build_pox_machine_vector(c2_job, self.ops_by_job, self.num_machines, self.C)
                else:
                    tp1, tp2 = transform_apmx(p1.job_vector, p2.job_vector)
                    tc1, tc2, _, _ = pmx_crossover(tp1, tp2)
                    c1_job = [p1.job_vector[v - 1] for v in tc1]
                    c2_job = [p1.job_vector[v - 1] for v in tc2]
                    c1_mach, c2_mach = self.two_point_crossover(p1.machine_vector, p2.machine_vector)

                child1, child2 = Individual(c1_job, c1_mach), Individual(c2_job, c2_mach)
                self.apply_mutation(child1, swap_mutation_prob, uniform_mutation_prob)
                self.apply_mutation(child2, swap_mutation_prob, uniform_mutation_prob)
                self.decoder.evaluate(child1)
                self.decoder.evaluate(child2)
                offspring.extend([child1, child2])

            offspring = offspring[:pop_size]

            se = int(elitism_rate * pop_size)
            parents_elite = sorted(population, key=lambda x: x.fitness)[:se]
            next_pop = offspring[:]
            if se > 0:
                replace_idx = random.sample(range(pop_size), se)
                for idx, elite in zip(replace_idx, parents_elite):
                    next_pop[idx] = elite

            unique_signatures = set()
            final_pop = []
            for ind in next_pop:
                sig = ind.get_signature()
                if sig not in unique_signatures:
                    unique_signatures.add(sig)
                    final_pop.append(ind)
                else:
                    rand_job_vec = list(self.flat_ops)
                    random.shuffle(rand_job_vec)
                    rand_mach_vec = [random.randint(1, self.num_machines) for _ in range(n_ops)]
                    immigrant = Individual(rand_job_vec, rand_mach_vec)
                    self.decoder.evaluate(immigrant)
                    final_pop.append(immigrant)

            population = final_pop
            archive.extend(clone_individual(ind) for ind in sorted(population, key=lambda x: x.fitness)[:archive_top_per_generation])

            current_best = min(population, key=lambda x: x.fitness)
            improved = current_best.fitness < f_best
            if improved:
                f_best = current_best.fitness
                best_ind = current_best
                best_improved = True
                q = 1
                no_improve = 0
            else:
                best_improved = False
                q += 1
                no_improve += 1

            generation += 1
            elapsed = time.time() - start_clock

            if show_progress:
                pbar.update(1)
                pbar.set_postfix(
                    best=round(float(f_best), 2),
                    noimp=int(no_improve),
                    archive=len(unique_sorted_individuals(archive)),
                )

            if record_history:
                history.append({
                    "generation": generation,
                    "runtime": float(elapsed),
                    "best_fitness": float(f_best),
                    "current_best": float(current_best.fitness),
                    "no_improve": int(no_improve),
                    "used_pox": bool(use_pox),
                    "improved": bool(improved),
                    "archive_size": len(unique_sorted_individuals(archive)),
                })

            if verbose:
                print(
                    f"gen={generation:4d} best={f_best:.4f} "
                    f"current={current_best.fitness:.4f} "
                    f"no_improve={no_improve:2d} pox={use_pox} "
                    f"time={elapsed:.2f}s archive={len(unique_sorted_individuals(archive))}"
                )

        pbar.close()

        if stop_reason is None:
            stop_reason = "unknown"

        best_ind = clone_individual(best_ind)
        best_ind.generations = generation
        best_ind.runtime = time.time() - start_clock
        best_ind.stop_reason = stop_reason
        best_ind.history = history
        best_ind.ga_archive = unique_sorted_individuals(archive)
        return best_ind, best_ind.ga_archive


def build_ops_by_job(jobs_data):
    ops_by_job = {}
    for op in jobs_data:
        job_id = int(op["job_id"])
        ops_by_job.setdefault(job_id, []).append(op)
    for job_id in ops_by_job:
        ops_by_job[job_id].sort(key=lambda x: x["op_id"])
    return ops_by_job


def run_ph_mh_alns_aos_on_file(
    case_file,
    seed=0,
    mh_time_seconds=PAPER_MAX_TIME_SECONDS,
    mh_no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
    mh_pop_size=PAPER_POP_SIZE,
    mh_max_generations=None,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    near_best_gap=0.10,
    num_alns_starts=1,
    reset_seed_per_alns_start=True,
    verbose=False,
    show_progress=True,
):
    """
    Complete hybrid pipeline:
        PH -> paper-style GA/MH -> select the best GA/MH start -> ALNS-AOS -> final ILP/TRM decoder evaluation.

    The GA/MH phase follows the same operators and stopping logic as the paper replication code.
    The ALNS-AOS phase uses the same ALNS_AOS class as the longer ALNS code.
    """
    random.seed(seed)
    np.random.seed(seed)

    jobs_data, m_case, c_case = load_actual_kmwe_instance(case_file)
    ops_by_job = build_ops_by_job(jobs_data)
    decoder = Decoder(ops_by_job, m_case, c_case, PAPER_SETUP_TIME)

    # Separate PH measurement for reporting. MH also initializes with PH internally.
    t0 = time.time()
    ph_engine = PractitionerHeuristic(jobs_data, num_machines=m_case, magazine_capacity=c_case)
    ph_solution = ph_engine.run()
    decoder.evaluate(ph_solution)
    ph_runtime = time.time() - t0

    mh_engine = MatheuristicWithArchive(jobs_data, num_machines=m_case, magazine_capacity=c_case)
    mh_solution, ga_archive = mh_engine.run_with_archive(
        max_time_seconds=mh_time_seconds,
        no_improvement_limit=mh_no_improvement_limit,
        pop_size=mh_pop_size,
        max_generations=mh_max_generations,
        record_history=True,
        verbose=verbose,
        show_progress=show_progress,
        progress_desc=f"ALNS-only seed={seed}",
    )
    decoder.evaluate(mh_solution)

    # Use only the single best GA/MH solution as the ALNS starting point.
    # near_best_gap and num_alns_starts are retained in the signature for
    # backward compatibility with existing experiment scripts.
    starts = select_best_ga_solution(ga_archive)

    alns_results = []
    start_iterator = tqdm(
        list(enumerate(starts, start=1)),
        desc=f"ALNS starts seed={seed}",
        leave=False,
        disable=not show_progress,
    )
    for idx, start_ind in start_iterator:
        if reset_seed_per_alns_start:
            # Deterministic but different random stream for each ALNS start.
            alns_seed = seed * 1000 + idx
            random.seed(alns_seed)
            np.random.seed(alns_seed)

        start_ind = clone_individual(start_ind)
        decoder.evaluate(start_ind)
        if show_progress:
            start_iterator.set_postfix(
                start=getattr(start_ind, "source_label", f"GA_start_{idx}"),
                fit=round(float(start_ind.fitness), 2),
            )

        alns_engine = ALNS_AOS(jobs_data, m_case, c_case, PAPER_SETUP_TIME)
        alns_solution = alns_engine.run(
            start_ind,
            max_time_seconds=alns_time_seconds,
            max_iterations=alns_iterations,
            no_improvement_limit=alns_no_improvement_limit,
            record_history=True,
            verbose=verbose,
            show_progress=show_progress,
            progress_desc=f"ALNS seed={seed} start={idx}/{len(starts)}",
        )
        decoder.evaluate(alns_solution)
        alns_solution.start_label = getattr(start_ind, "source_label", f"GA_start_{idx}")
        alns_solution.start_rank = idx
        alns_solution.start_fitness = float(start_ind.fitness)
        alns_solution.start_tardiness = float(start_ind.tardiness)
        alns_solution.start_setups = int(start_ind.setups)
        alns_solution.start_diversity = float(getattr(start_ind, "source_diversity", 0.0))
        alns_results.append(alns_solution)

    best_hybrid = min(alns_results, key=lambda x: (x.fitness, x.tardiness, x.setups))

    rows = []
    for sol in alns_results:
        rows.append({
            "case_file": case_file,
            "seed": seed,
            "start_rank": sol.start_rank,
            "start_label": sol.start_label,
            "start_fitness": sol.start_fitness,
            "start_tardiness": sol.start_tardiness,
            "start_setups": sol.start_setups,
            "start_diversity": sol.start_diversity,
            "ALNS_fitness": sol.fitness,
            "ALNS_tardiness": sol.tardiness,
            "ALNS_setups": sol.setups,
            "ALNS_runtime": getattr(sol, "alns_runtime", np.nan),
            "ALNS_iterations": getattr(sol, "alns_iterations", np.nan),
            "ALNS_stop_reason": getattr(sol, "alns_stop_reason", "unknown"),
            "is_best_hybrid_start": sol is best_hybrid,
        })

    summary = {
        "case_file": case_file,
        "seed": seed,
        "PH_fitness": ph_solution.fitness,
        "PH_tardiness": ph_solution.tardiness,
        "PH_setups": ph_solution.setups,
        "PH_runtime": ph_runtime,
        "MH_fitness": mh_solution.fitness,
        "MH_tardiness": mh_solution.tardiness,
        "MH_setups": mh_solution.setups,
        "MH_runtime": getattr(mh_solution, "runtime", np.nan),
        "MH_generations": getattr(mh_solution, "generations", np.nan),
        "MH_stop_reason": getattr(mh_solution, "stop_reason", "unknown"),
        "GA_archive_unique_size": len(ga_archive),
        "Hybrid_fitness": best_hybrid.fitness,
        "Hybrid_tardiness": best_hybrid.tardiness,
        "Hybrid_setups": best_hybrid.setups,
        "ILP_final_fitness": best_hybrid.fitness,
        "ILP_final_tardiness": best_hybrid.tardiness,
        "ILP_final_setups": best_hybrid.setups,
        "Hybrid_ALNS_runtime": getattr(best_hybrid, "alns_runtime", np.nan),
        "Hybrid_total_runtime": ph_runtime + getattr(mh_solution, "runtime", 0.0) + sum(getattr(s, "alns_runtime", 0.0) for s in alns_results),
        "Best_ALNS_start_label": getattr(best_hybrid, "start_label", "unknown"),
        "Best_ALNS_start_rank": getattr(best_hybrid, "start_rank", np.nan),
    }

    return summary, rows, {
        "ph_solution": ph_solution,
        "mh_solution": mh_solution,
        "ga_starts": starts,
        "alns_results": alns_results,
        "best_hybrid": best_hybrid,
        "ga_archive": ga_archive,
    }


def run_hybrid_experiments(
    case_files,
    seeds=range(10),
    output_prefix="hybrid_ph_mh_alns_aos_results",
    show_progress=True,
    **kwargs,
):
    """
    Batch runner for the project requirement: multiple independent runs per instance.
    Writes two CSV files:
      - <output_prefix>_summary.csv: one row per case/seed
      - <output_prefix>_starts.csv: one row per ALNS start solution
    """
    all_summaries = []
    all_start_rows = []

    case_iterator = tqdm(list(case_files), desc="Cases", disable=not show_progress)
    for case_file in case_iterator:
        case_iterator.set_postfix(case=os.path.basename(str(case_file)))
        seed_iterator = tqdm(list(seeds), desc=f"Seeds {os.path.basename(str(case_file))}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            msg = f"Running strict PH -> GA/MH -> ALNS-AOS -> ILP | case={case_file} | seed={seed}"
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)
            summary, start_rows, _ = run_ph_mh_alns_aos_on_file(
                case_file=case_file,
                seed=int(seed),
                show_progress=show_progress,
                **kwargs,
            )
            all_summaries.append(summary)
            all_start_rows.extend(start_rows)

            # Save after every run so partial results are not lost.

    summary_df = pd.DataFrame(all_summaries)
    starts_df = pd.DataFrame(all_start_rows)
    print("\nSummary results:")
    print(summary_df.to_string(index=False))
    return summary_df, starts_df



# User-facing strict pipeline aliases. These names make the execution order explicit.
def run_ph_ga_mh_alns_ilp_on_file(*args, **kwargs):
    """
    Strict project pipeline:
        PH -> GA/MH -> ALNS-AOS -> final ILP/TRM decoder evaluation.

    This is an alias for run_ph_mh_alns_aos_on_file(...). The last step is
    the final Decoder.evaluate(...), where the exact TRM/ILP tool-replacement
    model is called whenever capacity must be freed.
    """
    return run_ph_mh_alns_aos_on_file(*args, **kwargs)


def run_ph_ga_mh_alns_ilp_on_jobs_data(*args, **kwargs):
    """
    In-memory strict project pipeline:
        PH -> GA/MH -> ALNS-AOS -> final ILP/TRM decoder evaluation.
    """
    return run_ph_mh_alns_aos_on_jobs_data(*args, **kwargs)


def run_strict_hybrid_experiments(*args, **kwargs):
    """
    Batch strict project pipeline:
        PH -> GA/MH -> ALNS-AOS -> final ILP/TRM decoder evaluation.
    """
    return run_hybrid_experiments(*args, **kwargs)


# =====================================================================
# 8. TABLE 8 AND TABLE 14 HYBRID EXPERIMENTS
# =====================================================================

def _prepare_jobs_data_cache(jobs_data):
    """Reset the global tool-size cache for an already-loaded/sliced instance."""
    GLOBAL_TOOL_SIZES.clear()
    for op in jobs_data:
        GLOBAL_TOOL_SIZES[op["tool_set"]] = op["size"]


def run_ph_mh_alns_aos_on_jobs_data(
    jobs_data,
    num_machines,
    magazine_capacity,
    case_label="custom_instance",
    seed=0,
    mh_time_seconds=PAPER_MAX_TIME_SECONDS,
    mh_no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
    mh_pop_size=PAPER_POP_SIZE,
    mh_max_generations=None,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    near_best_gap=0.10,
    num_alns_starts=1,
    reset_seed_per_alns_start=True,
    verbose=False,
    show_progress=True,
):
    """
    Same hybrid pipeline as run_ph_mh_alns_aos_on_file(...), but works on an
    in-memory instance. This is needed for Table 8, where 6M140 is sliced into
    n = 15, 25, 30, 60, 90, 120, 140 operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    _prepare_jobs_data_cache(jobs_data)

    ops_by_job = build_ops_by_job(jobs_data)
    decoder = Decoder(ops_by_job, num_machines, magazine_capacity, PAPER_SETUP_TIME)

    t0 = time.time()
    ph_engine = PractitionerHeuristic(
        jobs_data,
        num_machines=num_machines,
        magazine_capacity=magazine_capacity,
    )
    ph_solution = ph_engine.run()
    decoder.evaluate(ph_solution)
    ph_runtime = time.time() - t0

    mh_engine = MatheuristicWithArchive(
        jobs_data,
        num_machines=num_machines,
        magazine_capacity=magazine_capacity,
    )
    mh_solution, ga_archive = mh_engine.run_with_archive(
        max_time_seconds=mh_time_seconds,
        no_improvement_limit=mh_no_improvement_limit,
        pop_size=mh_pop_size,
        max_generations=mh_max_generations,
        record_history=True,
        verbose=verbose,
        show_progress=show_progress,
        progress_desc=f"GA/MH {case_label} seed={seed}",
    )
    decoder.evaluate(mh_solution)

    # Use only the single best GA/MH solution as the ALNS starting point.
    # near_best_gap and num_alns_starts are retained in the signature for
    # backward compatibility with existing experiment scripts.
    starts = select_best_ga_solution(ga_archive)

    alns_results = []
    start_iterator = tqdm(
        list(enumerate(starts, start=1)),
        desc=f"ALNS starts {case_label} seed={seed}",
        leave=False,
        disable=not show_progress,
    )

    for idx, start_ind in start_iterator:
        if reset_seed_per_alns_start:
            alns_seed = seed * 1000 + idx
            random.seed(alns_seed)
            np.random.seed(alns_seed)

        start_ind = clone_individual(start_ind)
        decoder.evaluate(start_ind)

        if show_progress:
            start_iterator.set_postfix(
                start=getattr(start_ind, "source_label", f"GA_start_{idx}"),
                fit=round(float(start_ind.fitness), 2),
            )

        alns_engine = ALNS_AOS(jobs_data, num_machines, magazine_capacity, PAPER_SETUP_TIME)
        alns_solution = alns_engine.run(
            start_ind,
            max_time_seconds=alns_time_seconds,
            max_iterations=alns_iterations,
            no_improvement_limit=alns_no_improvement_limit,
            record_history=True,
            verbose=verbose,
            show_progress=show_progress,
            progress_desc=f"ALNS {case_label} seed={seed} start={idx}/{len(starts)}",
        )
        decoder.evaluate(alns_solution)
        alns_solution.start_label = getattr(start_ind, "source_label", f"GA_start_{idx}")
        alns_solution.start_rank = idx
        alns_solution.start_fitness = float(start_ind.fitness)
        alns_solution.start_tardiness = float(start_ind.tardiness)
        alns_solution.start_setups = int(start_ind.setups)
        alns_solution.start_diversity = float(getattr(start_ind, "source_diversity", 0.0))
        alns_results.append(alns_solution)

    best_hybrid = min(alns_results, key=lambda x: (x.fitness, x.tardiness, x.setups))

    start_rows = []
    for sol in alns_results:
        start_rows.append({
            "case_label": case_label,
            "seed": seed,
            "start_rank": sol.start_rank,
            "start_label": sol.start_label,
            "start_fitness": sol.start_fitness,
            "start_tardiness": sol.start_tardiness,
            "start_setups": sol.start_setups,
            "start_diversity": sol.start_diversity,
            "ALNS_fitness": sol.fitness,
            "ALNS_tardiness": sol.tardiness,
            "ALNS_setups": sol.setups,
            "ALNS_runtime": getattr(sol, "alns_runtime", np.nan),
            "ALNS_iterations": getattr(sol, "alns_iterations", np.nan),
            "ALNS_stop_reason": getattr(sol, "alns_stop_reason", "unknown"),
            "is_best_hybrid_start": sol is best_hybrid,
        })

    summary = {
        "case_label": case_label,
        "seed": seed,
        "PH_fitness": ph_solution.fitness,
        "PH_tardiness": ph_solution.tardiness,
        "PH_setups": ph_solution.setups,
        "PH_runtime": ph_runtime,
        "MH_fitness": mh_solution.fitness,
        "MH_tardiness": mh_solution.tardiness,
        "MH_setups": mh_solution.setups,
        "MH_runtime": getattr(mh_solution, "runtime", np.nan),
        "MH_generations": getattr(mh_solution, "generations", np.nan),
        "MH_stop_reason": getattr(mh_solution, "stop_reason", "unknown"),
        "GA_archive_unique_size": len(ga_archive),
        "Hybrid_fitness": best_hybrid.fitness,
        "Hybrid_tardiness": best_hybrid.tardiness,
        "Hybrid_setups": best_hybrid.setups,
        "ILP_final_fitness": best_hybrid.fitness,
        "ILP_final_tardiness": best_hybrid.tardiness,
        "ILP_final_setups": best_hybrid.setups,
        "Hybrid_ALNS_runtime": getattr(best_hybrid, "alns_runtime", np.nan),
        "Hybrid_total_runtime": ph_runtime + getattr(mh_solution, "runtime", 0.0) + sum(getattr(s, "alns_runtime", 0.0) for s in alns_results),
        "Best_ALNS_start_label": getattr(best_hybrid, "start_label", "unknown"),
        "Best_ALNS_start_rank": getattr(best_hybrid, "start_rank", np.nan),
    }

    return summary, start_rows, {
        "ph_solution": ph_solution,
        "mh_solution": mh_solution,
        "ga_starts": starts,
        "alns_results": alns_results,
        "best_hybrid": best_hybrid,
        "ga_archive": ga_archive,
    }



def _aggregate_hybrid_records(records, group_col):
    """Aggregate seed-level hybrid records into a paper-compatible summary table."""
    rows = []
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    for key, g in df.groupby(group_col, sort=False):
        ph = g["PH_fitness"].astype(float)
        mh = g["MH_fitness"].astype(float)
        hy = g["Hybrid_fitness"].astype(float)

        ph_mean = float(ph.mean())
        mh_mean = float(mh.mean())
        hy_mean = float(hy.mean())

        means = {"PH": ph_mean, "MH": mh_mean, "ALNS": hy_mean}
        best_method = min(means, key=means.get)

        rows.append({
            group_col: key,
            "PH_μ": round(ph_mean, 2),
            "PH_σ": round(float(ph.std(ddof=0)), 2),
            "PH_best": round(float(ph.min()), 2),
            "PH_C.T.(s)": round(float(g["PH_runtime"].mean()), 3),
            "MH_μ": round(mh_mean, 2),
            "MH_σ": round(float(mh.std(ddof=0)), 2),
            "MH_best": round(float(mh.min()), 2),
            "MH_C.T.(s)": round(float(g["MH_runtime"].mean()), 3),
            "MH_gen_μ": round(float(g["MH_generations"].mean()), 1),
            "ALNS_μ": round(hy_mean, 2),
            "ALNS_σ": round(float(hy.std(ddof=0)), 2),
            "ALNS_best": round(float(hy.min()), 2),
            "ALNS_C.T.(s)": round(float(g["Hybrid_total_runtime"].mean()), 3),
            "ALNS_phase_C.T.(s)": round(float(g["Hybrid_ALNS_runtime"].mean()), 3),
            "ALNS_it_μ": round(float(g["Hybrid_ALNS_iterations"].mean()), 1),
            "StopReasons": ",".join(sorted(set(map(str, g["MH_stop_reason"])))),
            "ALNS_StopReasons": ",".join(sorted(set(map(str, g["Hybrid_ALNS_stop_reason"])))),
            "Gap_MH_vs_PH (%)": f"{((mh_mean - ph_mean) / max(1.0, ph_mean)) * 100.0:.2f}%",
            "Gap_ALNS_vs_PH (%)": f"{((hy_mean - ph_mean) / max(1.0, ph_mean)) * 100.0:.2f}%",
            "Gap_ALNS_vs_MH (%)": f"{((hy_mean - mh_mean) / max(1.0, mh_mean)) * 100.0:.2f}%",
            "Best_Method": best_method,
            "Best_μ": round(float(means[best_method]), 2),
        })
    return pd.DataFrame(rows)


def _paper_like_table8_from_hybrid(run_records):
    """Return Table 8 with the same leading layout as Paper_replication_Lars plus hybrid columns."""
    full = _aggregate_hybrid_records(run_records, "n")
    if full.empty:
        return full
    cols = [
        "n",
        "PH_μ", "PH_σ", "PH_best", "PH_C.T.(s)",
        "MH_μ", "MH_σ", "MH_best", "MH_C.T.(s)", "MH_gen_μ",
        "StopReasons", "Gap_MH_vs_PH (%)",
        "ALNS_μ", "ALNS_σ", "ALNS_best", "ALNS_C.T.(s)", "ALNS_phase_C.T.(s)", "ALNS_it_μ",
        "ALNS_StopReasons", "Gap_ALNS_vs_PH (%)", "Gap_ALNS_vs_MH (%)",
        "Best_Method", "Best_μ",
    ]
    return full[cols]


def _paper_like_table14_from_hybrid(run_records):
    """Return Table 14 with the same leading layout as Paper_replication_Lars plus hybrid columns."""
    full = _aggregate_hybrid_records(run_records, "BaseCase")
    if full.empty:
        return full
    cols = [
        "BaseCase",
        "PH_μ", "PH_σ", "PH_best", "PH_C.T.(s)",
        "MH_μ", "MH_σ", "MH_best", "MH_C.T.(s)", "MH_gen_μ",
        "StopReasons", "Gap_MH_vs_PH (%)",
        "ALNS_μ", "ALNS_σ", "ALNS_best", "ALNS_C.T.(s)", "ALNS_phase_C.T.(s)", "ALNS_it_μ",
        "ALNS_StopReasons", "Gap_ALNS_vs_PH (%)", "Gap_ALNS_vs_MH (%)",
        "Best_Method", "Best_μ",
    ]
    out = full[cols].rename(columns={"Gap_MH_vs_PH (%)": "Net_Gap_MH (%)"})
    return out


def _enrich_summary_with_best_alns_metadata(summary, objects):
    """Add best ALNS metadata to the seed-level summary record for aggregation."""
    best_hybrid = objects["best_hybrid"]
    summary["Hybrid_ALNS_iterations"] = getattr(best_hybrid, "alns_iterations", np.nan)
    summary["Hybrid_ALNS_stop_reason"] = getattr(best_hybrid, "alns_stop_reason", "unknown")
    return summary


def run_hybrid_table8_replications(
    num_runs=10,
    output_prefix="hybrid_table8",
    mh_time_seconds=PAPER_MAX_TIME_SECONDS,
    mh_no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
    mh_pop_size=PAPER_POP_SIZE,
    mh_max_generations=None,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    near_best_gap=0.10,
    num_alns_starts=1,
    show_progress=True,
    verbose=False,
):
    """
    Produce the PH + MH + ALNS version of Table 8 with the same print style as Paper_replication_Lars.

    Defaults use 10 replications: seeds 0, 1, ..., 9.
    No CSV files are written; results are printed to the console and returned as DataFrames.
    """
    print("\n[EXACT REPLICATION: TABLE 8 - Operational Scaling Framework on 6M140]")

    case_file = resolve_kmwe_case_file("6M140")
    full_jobs_data, m_val, c_val = load_actual_kmwe_instance(case_file)
    df_sorted = pd.DataFrame(full_jobs_data).sort_values(by="r").copy()

    run_records = []
    start_records = []
    n_values = [15, 25, 30, 60, 90, 120, 140]

    n_iterator = tqdm(n_values, desc="Table 8 n-slices", disable=not show_progress)
    for n_slice in n_iterator:
        n_iterator.set_postfix(n=n_slice)
        sliced_ops = df_sorted.head(n_slice).to_dict(orient="records")
        seed_iterator = tqdm(range(num_runs), desc=f"Seeds n={n_slice}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            if show_progress:
                tqdm.write(f"Table 8 | n={n_slice} | seed={seed}")

            # The same replication seeds as Paper_replication_Lars: 0..num_runs-1.
            random.seed(int(seed))
            np.random.seed(int(seed))

            summary, starts, objects = run_ph_mh_alns_aos_on_jobs_data(
                sliced_ops,
                num_machines=m_val,
                magazine_capacity=c_val,
                case_label=f"6M140_n{n_slice}",
                seed=int(seed),
                mh_time_seconds=mh_time_seconds,
                mh_no_improvement_limit=mh_no_improvement_limit,
                mh_pop_size=mh_pop_size,
                mh_max_generations=mh_max_generations,
                alns_time_seconds=alns_time_seconds,
                alns_iterations=alns_iterations,
                alns_no_improvement_limit=alns_no_improvement_limit,
                near_best_gap=near_best_gap,
                num_alns_starts=num_alns_starts,
                verbose=verbose,
                show_progress=show_progress,
            )
            summary = _enrich_summary_with_best_alns_metadata(summary, objects)
            summary["n"] = n_slice
            for row in starts:
                row["n"] = n_slice
            run_records.append(summary)
            start_records.extend(starts)


    summary_df = _paper_like_table8_from_hybrid(run_records)
    print(summary_df.to_string(index=False))
    return summary_df, pd.DataFrame(run_records), pd.DataFrame(start_records)


def run_hybrid_table14_replications(
    num_runs=10,
    output_prefix="hybrid_table14",
    case_names=("2M38", "2M46", "6M140", "6M163"),
    mh_time_seconds=PAPER_MAX_TIME_SECONDS,
    mh_no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
    mh_pop_size=PAPER_POP_SIZE,
    mh_max_generations=None,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    near_best_gap=0.10,
    num_alns_starts=1,
    show_progress=True,
    verbose=False,
):
    """
    Produce the PH + MH + ALNS version of Table 14 with the same print style as Paper_replication_Lars.

    Defaults use 10 replications: seeds 0, 1, ..., 9.
    No CSV files are written; results are printed to the console and returned as DataFrames.
    """
    print("\n[EXACT REPLICATION: TABLE 14 - Production Base-Case Workcenters]")

    run_records = []
    start_records = []
    case_iterator = tqdm(list(case_names), desc="Table 14 cases", disable=not show_progress)

    for case_name in case_iterator:
        case_iterator.set_postfix(case=case_name)
        try:
            case_file = resolve_kmwe_case_file(case_name)
        except FileNotFoundError:
            print(f"Skipping baseline verification for {case_name}: Target CSV file missing.")
            continue

        seed_iterator = tqdm(range(num_runs), desc=f"Seeds {case_name}", leave=False, disable=not show_progress)
        for seed in seed_iterator:
            if show_progress:
                tqdm.write(f"Table 14 | case={case_name} | seed={seed}")

            # The same replication seeds as Paper_replication_Lars: 0..num_runs-1.
            random.seed(int(seed))
            np.random.seed(int(seed))

            summary, starts, objects = run_ph_mh_alns_aos_on_file(
                case_file=case_file,
                seed=int(seed),
                mh_time_seconds=mh_time_seconds,
                mh_no_improvement_limit=mh_no_improvement_limit,
                mh_pop_size=mh_pop_size,
                mh_max_generations=mh_max_generations,
                alns_time_seconds=alns_time_seconds,
                alns_iterations=alns_iterations,
                alns_no_improvement_limit=alns_no_improvement_limit,
                near_best_gap=near_best_gap,
                num_alns_starts=num_alns_starts,
                verbose=verbose,
                show_progress=show_progress,
            )
            summary = _enrich_summary_with_best_alns_metadata(summary, objects)
            summary["BaseCase"] = case_name
            for row in starts:
                row["BaseCase"] = case_name
            run_records.append(summary)
            start_records.extend(starts)


    summary_df = _paper_like_table14_from_hybrid(run_records)
    print(summary_df.to_string(index=False))
    return summary_df, pd.DataFrame(run_records), pd.DataFrame(start_records)


def run_exact_hybrid_replications(
    num_runs=10,
    mh_time_seconds=PAPER_MAX_TIME_SECONDS,
    mh_no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
    mh_pop_size=PAPER_POP_SIZE,
    mh_max_generations=None,
    alns_time_seconds=600.0,
    alns_iterations=2000,
    alns_no_improvement_limit=500,
    near_best_gap=0.10,
    num_alns_starts=1,
    show_progress=True,
    verbose=False,
    output_excel="hybrid_ph_mh_alns_seed_results.xlsx",
):
    """
    Main runner with the same console structure as Paper_replication_Lars.

    It produces Table 8 and Table 14 using 10 replications by default.
    """
    print("=" * 110)
    print(f" TRUE EXPERIMENTAL ENGINE: REPLICATING EXACT TABLES ({num_runs} SEED SAMPLES) ".center(110, "#"))
    print("=" * 110)

    table8_summary, table8_runs, table8_starts = run_hybrid_table8_replications(
        num_runs=num_runs,
        output_prefix="hybrid_table8",
        mh_time_seconds=mh_time_seconds,
        mh_no_improvement_limit=mh_no_improvement_limit,
        mh_pop_size=mh_pop_size,
        mh_max_generations=mh_max_generations,
        alns_time_seconds=alns_time_seconds,
        alns_iterations=alns_iterations,
        alns_no_improvement_limit=alns_no_improvement_limit,
        near_best_gap=near_best_gap,
        num_alns_starts=num_alns_starts,
        show_progress=show_progress,
        verbose=verbose,
    )

    table14_summary, table14_runs, table14_starts = run_hybrid_table14_replications(
        num_runs=num_runs,
        output_prefix="hybrid_table14",
        mh_time_seconds=mh_time_seconds,
        mh_no_improvement_limit=mh_no_improvement_limit,
        mh_pop_size=mh_pop_size,
        mh_max_generations=mh_max_generations,
        alns_time_seconds=alns_time_seconds,
        alns_iterations=alns_iterations,
        alns_no_improvement_limit=alns_no_improvement_limit,
        near_best_gap=near_best_gap,
        num_alns_starts=num_alns_starts,
        show_progress=show_progress,
        verbose=verbose,
    )

    export_seed_results_to_excel(
        output_excel,
        table8_summary=table8_summary,
        table8_seed_results=table8_runs,
        table8_alns_starts=table8_starts,
        table14_summary=table14_summary,
        table14_seed_results=table14_runs,
        table14_alns_starts=table14_starts,
    )

    return {
        "table8_summary": table8_summary,
        "table8_runs": table8_runs,
        "table8_starts": table8_starts,
        "table14_summary": table14_summary,
        "table14_runs": table14_runs,
        "table14_starts": table14_starts,
    }


# Backward-compatible alias: running this file behaves like the paper replication script,
# but with the added ALNS columns.
def run_exact_paper_replications(num_runs=10, output_excel="hybrid_ph_mh_alns_seed_results.xlsx"):
    return run_exact_hybrid_replications(num_runs=num_runs, output_excel=output_excel)


if __name__ == "__main__":
    # Final default: exactly 10 replications, using seeds 0..9, like the paper-style runner.
    # This can take a long time because each seed runs PH + MH/GA + 1 ALNS-AOS start.
    run_exact_hybrid_replications(
        num_runs=10,
        mh_time_seconds=3600.0,
        mh_no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
        alns_time_seconds=600.0,
        alns_iterations=2000,
        alns_no_improvement_limit=500,
        show_progress=True,
    )
