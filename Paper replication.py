import os
import glob
import re
import random
import time
import pandas as pd

# Global cache for tool sizes to optimize search speed in the knapsack solver
GLOBAL_TOOL_SIZES = {}

# =====================================================================
# 1. THE EXACT TOOL REPLACEMENT METHOD (TRM) SOLVER
# =====================================================================
def solve_trm_knapsack(tools_in_magazine, tool_sizes, scores, needed_capacity):
    best_cost = float('inf')
    best_subset = []
    n = len(tools_in_magazine)
    
    tools_sorted = sorted(
        tools_in_magazine, 
        key=lambda x: scores.get(x, 0) / max(1, tool_sizes.get(x, 1))
    )

    def backtrack(index, current_weight, current_cost, current_subset):
        nonlocal best_cost, best_subset
        if current_weight >= needed_capacity:
            if current_cost < best_cost:
                best_cost = current_cost
                best_subset = list(current_subset)
            return
        if index >= n:
            return
        if current_cost >= best_cost:
            return
            
        tool = tools_sorted[index]
        
        # Branch 1: Evict this tool
        current_subset.append(tool)
        backtrack(index + 1, current_weight + tool_sizes[tool], current_cost + scores.get(tool, 0), current_subset)
        current_subset.pop()
        
        # Branch 2: Keep this tool
        backtrack(index + 1, current_weight, current_cost, current_subset)
        
    backtrack(0, 0, 0, [])
    return best_subset


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
    def __init__(self, ops_by_job, num_machines, magazine_capacity, setup_time=1.0, weights=(1.0, 1.0)):
        self.ops_by_job = ops_by_job
        self.num_machines = num_machines
        self.C = magazine_capacity
        self.tau = setup_time
        self.w_d, self.w_s = weights

    def evaluate(self, individual):
        job_vec = individual.job_vector
        mach_vec = individual.machine_vector
        n = len(job_vec)
        
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
        
        # Step B: Run standard simulation
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
            r_ij = op_data['r']
            p_ij = op_data['p']
            d_ij = op_data['d']
            
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
                        allocated_space = 0
                        for t in zero_score_tools:
                            T_m[m_id].remove(t)
                            allocated_space += GLOBAL_TOOL_SIZES[t]
                            if allocated_space >= needed_space:
                                break
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
            
            if occ == 0:
                start_time = max(r_ij, a_m[m_id])
            else:
                prev_finish = job_finish_times.get((j_id, occ - 1), 0.0)
                start_time = max(r_ij, a_m[m_id], prev_finish)
            end_time = start_time + p_ij + (self.tau * z_ijt)
            
            a_m[m_id] = end_time
            job_finish_times[(j_id, occ)] = end_time
            
            tardiness_ij = max(0.0, end_time - d_ij)
            total_tardiness += tardiness_ij
            
        individual.tardiness = total_tardiness
        individual.setups = total_setups
        individual.fitness = (self.w_d * total_tardiness) + (self.w_s * self.tau * total_setups)


# =====================================================================
# DYNAMIC PRACTITIONER HEURISTIC (ALGORITHM 4)
# =====================================================================
class PractitionerHeuristic:
    def __init__(self, jobs_data, num_machines, magazine_capacity, tool_setup_time=1.0, theta_m=72.0, weights=(1.0, 1.0)):
        self.O = jobs_data
        self.M = list(range(1, num_machines + 1))
        self.C = magazine_capacity
        self.tau = tool_setup_time  
        self.theta_m = theta_m      
        self.wd, self.ws = weights  
        
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
        
        job_vector = []
        machine_vector = []
        
        for op in O_hat:
            job_id, op_id = op['job_id'], op['op_id']
            t_ij = op['tool_set']
            phi_t = op['size']
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
                    removed_tool = random.choice(list(self.T_m[m_star]))
                    self.T_m[m_star].remove(removed_tool)
                    phi_s = phi_t - (self.C - self.get_magazine_size(m_star))
                    
                self.T_m[m_star].add(t_ij)
                total_setups += 1
            
            start_time = calc_xi(m_star)
            end_time = start_time + p_ij + (self.tau * z_ijt)
            self.a_m[m_star] = end_time
            job_finish_times[(job_id, occ)] = end_time
            
            tardiness = max(0.0, end_time - d_ij)
            total_tardiness += tardiness
            
            job_vector.append(job_id)
            machine_vector.append(m_star)
            
        objective_val = total_tardiness + (self.tau * total_setups)
        
        ind = Individual(job_vector, machine_vector)
        ind.fitness = objective_val
        ind.tardiness = total_tardiness
        ind.setups = total_setups
        
        return ind


# =====================================================================
# 3. GENETIC OPERATORS
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
            while val1 in map1:
                val1 = map1[val1]
            child1[i] = val1
            
            val2 = p2[i]
            while val2 in map2:
                val2 = map2[val2]
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
            op_data = ops_by_job[job][occ]
            outside_elements.append((job, op_data, idx))
            
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
        t_ij = op_data['tool_set']
        phi_t = op_data['size']
        p_ij = op_data['p']
        
        m_T_list = [m for m in range(1, num_machines + 1) if t_ij in T_m[m]]
        if m_T_list:
            m_star = m_T_list[0]
        else:
            M_C = []
            for m in range(1, num_machines + 1):
                current_size = sum(GLOBAL_TOOL_SIZES[t] for t in T_m[m])
                if magazine_capacity - current_size >= phi_t:
                    M_C.append(m)
            if M_C:
                m_star = min(M_C, key=lambda m: p_m[m])
                T_m[m_star].add(t_ij)
            else:
                m_star = min(range(1, num_machines + 1), key=lambda m: p_m[m])
                
        p_m[m_star] += p_ij
        mach_vec.append(m_star)
        
    return mach_vec


# =====================================================================
# 4. THE PROPOSED MATHEURISTIC (MH) RESOLUTION
# =====================================================================
class Matheuristic:
    def __init__(self, jobs_data, num_machines, magazine_capacity, setup_time=1.0):
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

    def run(self, max_generations=30, pop_size=100):
        gamma_1 = 0.20   
        gamma_2 = 0.10   
        Ps = 0.01        
        Pu = 0.01        
        B = 1            

        population = []
        
        # Exact Dynamic PH baseline loaded as first seed
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
            
        f_best = min(ind.fitness for ind in population)
        best_ind = min(population, key=lambda x: x.fitness)
        
        q = 1
        
        for generation in range(1, max_generations + 1):
            offspring = []
            while len(offspring) < pop_size:
                p1, p2 = self.select_parents(population, gamma_1)
                
                if q <= B:
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
                    
                child1 = Individual(c1_job, c1_mach)
                child2 = Individual(c2_job, c2_mach)
                
                self.apply_mutation(child1, Ps, Pu)
                self.apply_mutation(child2, Ps, Pu)
                
                self.decoder.evaluate(child1)
                self.decoder.evaluate(child2)
                
                offspring.extend([child1, child2])
                
            offspring = offspring[:pop_size]
            population.sort(key=lambda x: x.fitness)
            offspring.sort(key=lambda x: x.fitness)
            
            n_elitism = int(gamma_2 * pop_size)
            next_pop = offspring[:pop_size - n_elitism] + population[:n_elitism]
            
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
            if current_best.fitness < f_best:
                f_best = current_best.fitness
                best_ind = current_best
                q = 1
            else:
                q += 1
                
        return best_ind

    def select_parents(self, population, tournament_rate):
        size = max(2, int(tournament_rate * len(population)))
        group1 = random.sample(population, size)
        group2 = random.sample(population, size)
        return min(group1, key=lambda x: x.fitness), min(group2, key=lambda x: x.fitness)

    def two_point_crossover(self, m1, m2):
        n = len(m1)
        cx1 = random.randint(0, n - 2)
        cx2 = random.randint(cx1 + 1, n - 1)
        c1 = list(m1)
        c2 = list(m2)
        c1[cx1:cx2+1] = m2[cx1:cx2+1]
        c2[cx1:cx2+1] = m1[cx1:cx2+1]
        return c1, c2

    def apply_mutation(self, individual, Ps, Pu):
        if random.random() < Ps:
            i, j = random.sample(range(len(individual.job_vector)), 2)
            individual.job_vector[i], individual.job_vector[j] = individual.job_vector[j], individual.job_vector[i]
        for idx in range(len(individual.machine_vector)):
            if random.random() < Pu:
                individual.machine_vector[idx] = random.randint(1, self.num_machines)


# =====================================================================
# 5. DATA LOADING AND METRICS EVALUATION
# =====================================================================
def run_practitioner_heuristic_standalone(filepath):
    num_machines = 2
    magazine_capacity = 80
    
    try:
        with open(filepath, 'r') as f:
            for _ in range(5):
                line = f.readline().strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    if parts[0] == 'M':
                        num_machines = int(parts[1])
                    elif parts[0] == 'C':
                        magazine_capacity = int(parts[1])
    except Exception:
        pass

    df = pd.read_csv(filepath, skiprows=5)
    df.columns = ['job_id', 'op_id', 'r', 'p', 'd', 'tool_set', 'size']
    jobs_data = df.to_dict(orient='records')

    for op in jobs_data:
        GLOBAL_TOOL_SIZES[op['tool_set']] = op['size']

    ph_engine = PractitionerHeuristic(jobs_data, num_machines, magazine_capacity)
    ph_ind = ph_engine.run()
    
    mh = Matheuristic(jobs_data, num_machines, magazine_capacity)
    
    return ph_ind, mh


# =====================================================================
# REPLICATION ENGINE FOR ALL TABLES
# =====================================================================
def run_paper_replications():
    
    aggregated_results = {
        "2M": {"S1": [], "S2": [], "S3": []},
        "6M": {"S1": [], "S2": [], "S3": []}
    }

    def record_metric(case_name, scenario_key, ph_val, mh_val):
        machine_type = "2M" if case_name.startswith("2M") else "6M"
        diff_hours = mh_val - ph_val
        pct_gap = (diff_hours / max(1.0, ph_val)) * 100.0
        aggregated_results[machine_type][scenario_key].append((diff_hours, pct_gap))

    # ==========================================
    # REPLICATING TABLE 8 (Varying n for 6M140)
    # ==========================================
    print("="*100)
    print(" REPLICATING TABLE 8: Comparative analysis on 6M140 ".center(100, "#"))
    print("="*100)
    
    base_6m140 = os.path.join("6M140", "6M140.csv")
    if os.path.exists(base_6m140):
        df_base = pd.read_csv(base_6m140, skiprows=5)
        df_base.columns = ['job_id', 'op_id', 'r', 'p', 'd', 'tool_set', 'size']
        df_sorted = df_base.sort_values(by='r').copy()
        
        t8_rows = []
        for n_val in [15, 25, 30, 60, 90, 120, 140]:
            sub_df = df_sorted.head(n_val)
            sub_ops = sub_df.to_dict(orient='records')
            
            for op in sub_ops:
                GLOBAL_TOOL_SIZES[op['tool_set']] = op['size']
                
            ph_engine = PractitionerHeuristic(sub_ops, 6, 80)
            t0 = time.time()
            ph_ind = ph_engine.run()
            ph_time = time.time() - t0
            
            mh_engine = Matheuristic(sub_ops, 6, 80)
            t0 = time.time()
            mh_ind = mh_engine.run(max_generations=30, pop_size=100)
            mh_time = time.time() - t0
            
            gap = ((mh_ind.fitness - ph_ind.fitness) / max(1.0, ph_ind.fitness)) * 100.0
            
            t8_rows.append({
                "n": n_val,
                "PH_Obj": round(ph_ind.fitness, 2),
                "PH_C.T.(s)": round(ph_time, 4),
                "MH_Obj": round(mh_ind.fitness, 2),
                "MH_C.T.(s)": round(mh_time, 4),
                "Gap (%)": f"{gap:.2f}%"
            })
        print(pd.DataFrame(t8_rows).to_string(index=False))

    # ==========================================================
    # REPLICATING TABLES 9, 10, 11 (Sensitivity on 6M140)
    # ==========================================================
    for s_idx, tbl_num, s_key, label in [(1, 9, "S1", "Tool Ratio"), (2, 10, "S2", "Magazine Capacity"), (3, 11, "S3", "Deadline Shift")]:
        print("\n" + "="*100)
        print(f" REPLICATING TABLE {tbl_num}: Sensitivity on 6M140 ({label}) ".center(100, "#"))
        print("="*100)
        
        scenario_dir = os.path.join("6M140", f"Scenario{s_idx}")
        if os.path.exists(scenario_dir):
            files = glob.glob(os.path.join(scenario_dir, "*.csv"))
            files = sorted(files, key=get_scenario_sort_key)
            
            tbl_rows = []
            for filepath in files:
                t0 = time.time()
                ph_ind, mh_engine = run_practitioner_heuristic_standalone(filepath)
                ph_time = time.time() - t0
                
                t0 = time.time()
                mh_ind = mh_engine.run(max_generations=30, pop_size=100)
                mh_time = time.time() - t0
                
                gap_hours = mh_ind.fitness - ph_ind.fitness
                gap_pct = (gap_hours / max(1.0, ph_ind.fitness)) * 100.0
                record_metric("6M140", s_key, ph_ind.fitness, mh_ind.fitness)
                
                param = os.path.basename(filepath).split('_')[1].replace(".csv", "")
                
                tbl_rows.append({
                    "Param": param,
                    "PH_Obj": round(ph_ind.fitness, 2),
                    "PH_C.T.(s)": round(ph_time, 4),
                    "MH_Obj": round(mh_ind.fitness, 2),
                    "MH_C.T.(s)": round(mh_time, 4),
                    "Gap (Hours)": round(gap_hours, 2),
                    "Gap (%)": f"{gap_pct:.2f}%"
                })
            print(pd.DataFrame(tbl_rows).to_string(index=False))

    # ==========================================================
    # REPLICATING TABLES 15, 16, 17 (Sensitivity on other cases)
    # ==========================================================
    for s_idx, tbl_num, s_key, label in [(1, 15, "S1", "Tool Ratio"), (2, 16, "S2", "Magazine Capacity"), (3, 17, "S3", "Deadline Shift")]:
        print("\n" + "="*100)
        print(f" REPLICATING TABLE {tbl_num}: Sensitivity on 2M38, 2M46, 6M163 ({label}) ".center(100, "#"))
        print("="*100)
        
        tbl_rows = []
        for case in ["2M38", "2M46", "6M163"]:
            scenario_dir = os.path.join(case, f"Scenario{s_idx}")
            if os.path.exists(scenario_dir):
                files = glob.glob(os.path.join(scenario_dir, "*.csv"))
                files = sorted(files, key=get_scenario_sort_key)
                
                for filepath in files:
                    t0 = time.time()
                    ph_ind, mh_engine = run_practitioner_heuristic_standalone(filepath)
                    ph_time = time.time() - t0
                    
                    t0 = time.time()
                    mh_ind = mh_engine.run(max_generations=30, pop_size=100)
                    mh_time = time.time() - t0
                    
                    gap_hours = mh_ind.fitness - ph_ind.fitness
                    gap_pct = (gap_hours / max(1.0, ph_ind.fitness)) * 100.0
                    record_metric(case, s_key, ph_ind.fitness, mh_ind.fitness)
                    
                    param = os.path.basename(filepath).split('_')[1].replace(".csv", "")
                    
                    tbl_rows.append({
                        "Case": case,
                        "Param": param,
                        "PH_Obj": round(ph_ind.fitness, 2),
                        "PH_C.T.(s)": round(ph_time, 4),
                        "MH_Obj": round(mh_ind.fitness, 2),
                        "MH_C.T.(s)": round(mh_time, 4),
                        "Gap (Hours)": round(gap_hours, 2),
                        "Gap (%)": f"{gap_pct:.2f}%"
                    })
        print(pd.DataFrame(tbl_rows).to_string(index=False))

    # ==========================================
    # REPLICATING TABLE 12 (Results Summary)
    # ==========================================
    print("\n" + "="*100)
    print(" REPLICATING TABLE 12: Summary of results of sensitivity analysis ".center(100, "#"))
    print("="*100)
    
    t12_rows = []
    scenarios_meta = [
        ("S1", "Tool ratio (Scenario 1)"),
        ("S2", "Magazine capacity (Scenario 2)"),
        ("S3", "Deadline shift (Scenario 3)")
    ]
    
    for s_key, s_label in scenarios_meta:
        m2_data = aggregated_results["2M"][s_key]
        m2_hours = sum(item[0] for item in m2_data) / max(1, len(m2_data))
        m2_pct = sum(item[1] for item in m2_data) / max(1, len(m2_data))
        
        m6_data = aggregated_results["6M"][s_key]
        m6_hours = sum(item[0] for item in m6_data) / max(1, len(m6_data))
        m6_pct = sum(item[1] for item in m6_data) / max(1, len(m6_data))
        
        t12_rows.append({
            "Scenario": s_label,
            "2M Gap (Hours)": round(m2_hours, 2),
            "2M Gap (%)": f"{m2_pct:.2f}%",
            "6M Gap (Hours)": round(m6_hours, 2),
            "6M Gap (%)": f"{m6_pct:.2f}%"
        })
        
    print(pd.DataFrame(t12_rows).to_string(index=False))


def get_scenario_sort_key(filepath):
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if len(parts) >= 2:
        variable_part = parts[1]
        match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", variable_part)
        if match:
            return float(match.group())
    return 0.0


if __name__ == "__main__":
    run_paper_replications()