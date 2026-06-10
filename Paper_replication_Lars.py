import os
import glob
import re
import random
import time
import numpy as np
import pandas as pd

# Global cache for tool sizes to optimize search speed in the knapsack solver
GLOBAL_TOOL_SIZES = {}

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

    This version solves the TRM as an actual binary ILP using SciPy/HiGHS,
    rather than enumerating all subsets. No additional tie-breaking objective
    is added, so the model matches equations (23)-(25) as closely as possible.
    """
    if needed_capacity <= 0:
        return []

    tools = list(tools_in_magazine)
    if not tools:
        return []

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for the ILP-based TRM solver. "
            "Install it with: pip install scipy"
        ) from exc

    c = np.array([float(scores.get(t, 0.0)) for t in tools], dtype=float)
    sizes = np.array([float(tool_sizes[t]) for t in tools], dtype=float)

    # Constraint: sum(phi_t * lambda_t) >= needed_capacity
    constraints = LinearConstraint(
        A=sizes.reshape(1, -1),
        lb=np.array([float(needed_capacity)]),
        ub=np.array([np.inf]),
    )

    # Binary variables: 0 <= lambda_t <= 1, lambda_t integer
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
# 6. REPLICATION EXECUTION HUB
# =====================================================================
# Synthetic/mock-data experiments were intentionally removed.
# All experiment runners below load real KMWE CSV files through
# load_actual_kmwe_instance(...).


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
    
    # 1. Extract metadata from the 5-line header
    try:
        with open(filepath, 'r') as f:
            for _ in range(5):
                line = f.readline().strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    if parts[0].strip() == 'M':
                        num_machines = int(parts[1])
                    elif parts[0].strip() == 'C':
                        magazine_capacity = int(parts[1])
    except Exception as e:
        print(f"Header parsing warning on {filepath}: {e}")

    # 2. Read the structured operations block
    df = pd.read_csv(filepath, skiprows=5)
    df.columns = ['job_id', 'op_id', 'r', 'p', 'd', 'tool_set', 'size']
    jobs_data = df.to_dict(orient='records')

    # 3. Cache global tool dimensions for the TRM knapsack solver
    for op in jobs_data:
        GLOBAL_TOOL_SIZES[op['tool_set']] = op['size']
        
    return jobs_data, num_machines, magazine_capacity


# =====================================================================
# EXACT REPLICATION ENGINE HUB
# =====================================================================
def run_exact_paper_replications(num_runs=20):
    print("="*110)
    print(f" TRUE EXPERIMENTAL ENGINE: REPLICATING EXACT TABLES ({num_runs} SEED SAMPLES) ".center(110, "#"))
    print("="*110)

    # -------------------------------------------------------------
    # PARSING & REPLICATING TABLE 8 (Variable Scaling 'n' on 6M140)
    # -------------------------------------------------------------
    print("\n[EXACT REPLICATION: TABLE 8 - Operational Scaling Framework on 6M140]")
    
    # Locate the base 6M140 file (handling potential nesting variations)
    base_6m140_paths = [
        os.path.join("6M140", "6M140.csv"),
        os.path.join("6M140", "Base 6M140.csv"),
        "6M140.csv"
    ]
    
    target_file = None
    for path in base_6m140_paths:
        if os.path.exists(path):
            target_file = path
            break
            
    if target_file is None:
        print("CRITICAL: '6M140.csv' base data file not detected in work directory.")
        print("Please check your local folder paths.")
        return
        
    # Load and sequence slice configurations precisely as defined by the paper's methodology
    full_jobs_data, m_val, c_val = load_actual_kmwe_instance(target_file)
    df_base = pd.DataFrame(full_jobs_data)
    df_sorted = df_base.sort_values(by='r').copy() # Sorted chronologically by release time
    
    t8_metrics = []
    for n_slice in [15, 25, 30, 60, 90, 120, 140]:
        ph_objectives, mh_objectives = [], []
        ph_times, mh_times = [], []
        mh_generations, mh_stop_reasons = [], []
        
        # Take the first 'n' operations after sorting chronologically
        sliced_ops = df_sorted.head(n_slice).to_dict(orient='records')
        
        for run_seed in range(num_runs):
            random.seed(run_seed)
            np.random.seed(run_seed)
            
            # PH Baseline Model
            t0 = time.time()
            ph_eng = PractitionerHeuristic(sliced_ops, num_machines=m_val, magazine_capacity=c_val)
            ph_res = ph_eng.run()
            ph_times.append(time.time() - t0)
            ph_objectives.append(ph_res.fitness)
            
            # MH Metaheuristic Model: defaults are the paper parameters.
            t0 = time.time()
            mh_eng = Matheuristic(sliced_ops, num_machines=m_val, magazine_capacity=c_val)
            mh_res = mh_eng.run()
            mh_times.append(time.time() - t0)
            mh_objectives.append(mh_res.fitness)
            mh_generations.append(getattr(mh_res, "generations", np.nan))
            mh_stop_reasons.append(getattr(mh_res, "stop_reason", "unknown"))
            
        mean_ph = np.mean(ph_objectives)
        mean_mh = np.mean(mh_objectives)
        gap = ((mean_mh - mean_ph) / max(1.0, mean_ph)) * 100.0
        
        t8_metrics.append({
            "n": n_slice,
            "PH_μ": round(mean_ph, 2), "PH_σ": round(np.std(ph_objectives), 2), "PH_C.T.(s)": round(np.mean(ph_times), 3),
            "MH_μ": round(mean_mh, 2), "MH_σ": round(np.std(mh_objectives), 2), "MH_C.T.(s)": round(np.mean(mh_times), 3),
            "MH_gen_μ": round(np.nanmean(mh_generations), 1),
            "StopReasons": ",".join(sorted(set(mh_stop_reasons))),
            "Gap (%)": f"{gap:.2f}%"
        })
    print(pd.DataFrame(t8_metrics).to_string(index=False))


    # -------------------------------------------------------------
    # PARSING & REPLICATING TABLE 14 (Full Industrial Benchmarks)
    # -------------------------------------------------------------
    print("\n[EXACT REPLICATION: TABLE 14 - Production Base-Case Workcenters]")
    t14_metrics = []
    
    cases_config = ["2M38", "2M46", "6M140", "6M163"]
    
    for case_name in cases_config:
        # Search for base cases dynamically in target directories
        possible_paths = [
            os.path.join(case_name, f"{case_name}.csv"),
            os.path.join(case_name, f"Base {case_name}.csv"),
            f"{case_name}.csv"
        ]
        
        case_file = None
        for path in possible_paths:
            if os.path.exists(path):
                case_file = path
                break
                
        if not case_file:
            print(f"Skipping baseline verification for {case_name}: Target CSV file missing.")
            continue
            
        case_ops, m_case, c_case = load_actual_kmwe_instance(case_file)
        ph_objectives, mh_objectives = [], []
        ph_times, mh_times = [], []
        mh_generations, mh_stop_reasons = [], []
        
        for run_seed in range(num_runs):
            random.seed(run_seed)
            np.random.seed(run_seed)
            
            # PH Optimization Run
            t0 = time.time()
            ph_eng = PractitionerHeuristic(case_ops, num_machines=m_case, magazine_capacity=c_case)
            ph_res = ph_eng.run()
            ph_times.append(time.time() - t0)
            ph_objectives.append(ph_res.fitness)
            
            # MH Optimization Run: defaults are the paper parameters.
            t0 = time.time()
            mh_eng = Matheuristic(case_ops, num_machines=m_case, magazine_capacity=c_case)
            mh_res = mh_eng.run()
            mh_times.append(time.time() - t0)
            mh_objectives.append(mh_res.fitness)
            mh_generations.append(getattr(mh_res, "generations", np.nan))
            mh_stop_reasons.append(getattr(mh_res, "stop_reason", "unknown"))
            
        mean_ph = np.mean(ph_objectives)
        mean_mh = np.mean(mh_objectives)
        gap = ((mean_mh - mean_ph) / max(1.0, mean_ph)) * 100.0
        
        t14_metrics.append({
            "BaseCase": case_name,
            "PH_μ": round(mean_ph, 2), "PH_σ": round(np.std(ph_objectives), 2), "PH_C.T.(s)": round(np.mean(ph_times), 3),
            "MH_μ": round(mean_mh, 2), "MH_σ": round(np.std(mh_objectives), 2), "MH_C.T.(s)": round(np.mean(mh_times), 3),
            "MH_gen_μ": round(np.nanmean(mh_generations), 1),
            "StopReasons": ",".join(sorted(set(mh_stop_reasons))),
            "Net_Gap (%)": f"{gap:.2f}%"
        })
    print(pd.DataFrame(t14_metrics).to_string(index=False))


def run_no_premature_stop_check(
    case_file,
    seed=0,
    max_time_seconds=PAPER_MAX_TIME_SECONDS,
    normal_no_improvement_limit=PAPER_NO_IMPROVEMENT_LIMIT,
    extended_no_improvement_limit=10**9,
    export_prefix="mh_verification",
):
    """
    Diagnostic experiment to verify whether the paper stopping rule
    terminates before meaningful improvements are exhausted.

    It runs the same instance twice with the same seed:
      1. normal paper mode: no-improvement limit = 20
      2. extended mode: no-improvement limit effectively disabled

    Both runs still respect the same max_time_seconds. The function exports
    convergence-history CSV files so you can inspect best_fitness over time.
    """
    case_ops, m_case, c_case = load_actual_kmwe_instance(case_file)

    results = []
    for label, gilimit in [
        ("paper_stop", normal_no_improvement_limit),
        ("extended_stop_check", extended_no_improvement_limit),
    ]:
        random.seed(seed)
        np.random.seed(seed)
        mh_eng = Matheuristic(case_ops, num_machines=m_case, magazine_capacity=c_case)
        res = mh_eng.run(
            max_time_seconds=max_time_seconds,
            no_improvement_limit=gilimit,
            max_generations=None,
            record_history=True,
            verbose=False,
        )

        history_df = pd.DataFrame(res.history)
        history_path = f"{export_prefix}_{label}_seed{seed}.csv"
        history_df.to_csv(history_path, index=False)

        results.append({
            "mode": label,
            "seed": seed,
            "fitness": res.fitness,
            "tardiness": res.tardiness,
            "setups": res.setups,
            "runtime_seconds": res.runtime,
            "generations": res.generations,
            "stop_reason": res.stop_reason,
            "history_csv": history_path,
        })

    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    # Paper uses 10 runs for reported means and standard deviations.
    run_exact_paper_replications(num_runs=10)
    