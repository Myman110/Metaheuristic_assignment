import pandas as pd
import requests
import io
import random

# =====================================================================
# 1. THE HEURISTIC ENGINE (From our previous blueprint)
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

    def phase_1_initial_allocation(self):
        # Sort operations using the EDD (Earliest Due Date) rule
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
        return O_hat

    def phase_2_scheduling(self, O_hat):
        total_tardiness = 0
        total_setups = 0
        schedule_log = {m: [] for m in self.M}
        
        for op in O_hat:
            job_id, op_id = op['job_id'], op['op_id']
            t_ij = op['tool_set']
            phi_t = op['size']
            r_ij, p_ij, d_ij = op['r'], op['p'], op['d']
            
            m_P = min(self.M, key=lambda m: self.a_m[m])
            m_T_list = [m for m in self.M if t_ij in self.T_m[m]]
            m_T = m_T_list[0] if m_T_list else None
            
            def calc_xi(machine):
                return max(r_ij, self.a_m[machine])
            
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
            
            tardiness = max(0.0, end_time - d_ij)
            total_tardiness += tardiness
            
            schedule_log[m_star].append({
                'job': f"J{job_id}(Op{op_id})", 
                'start': round(start_time, 2), 
                'end': round(end_time, 2), 
                'tardiness': round(tardiness, 2),
                'tool_switch': bool(z_ijt)
            })
            
        objective_val = (self.wd * total_tardiness) + (self.ws * self.tau * total_setups)
        return objective_val, total_tardiness, total_setups, schedule_log

# =====================================================================
# 2. REMOTE DATA FETCHING & EXECUTION
# =====================================================================
def load_and_run_benchmark(instance_name, num_machines, magazine_capacity=80):
    # Base URL pointing to the raw repository files
    base_url = "https://raw.githubusercontent.com/vinhise/pmstr-basecases/master/"
    file_url = f"{base_url}{instance_name}"
    
    print(f"Fetching case file: {file_url}...")
    response = requests.get(file_url)
    
    if response.status_code != 200:
        raise FileNotFoundError(f"Could not retrieve instance '{instance_name}'. Check name format.")

    # Parse dataset (handling variable whitespaces/tabs often present in TXT matrices)
    df = pd.read_csv(io.StringIO(response.text), sep=r'\s+', header=None)
    df.columns = ['job_id', 'op_id', 'r', 'p', 'd', 'tool_set', 'size']
    
    # Convert dataframes into structural row dictionaries
    jobs_data = df.to_dict(orient='records')
    print(f"Successfully loaded {len(jobs_data)} scheduling operations.")

    # Initialize and execute the engine
    # Default settings matching KMWE baseline configurations mentioned in the paper:
    # Setup time = 1 hour, Threshold = 72 hours
    heuristic = PractitionerHeuristic(
        jobs_data=jobs_data, 
        num_machines=num_machines, 
        magazine_capacity=magazine_capacity,
        tool_setup_time=1.0, 
        theta_m=72.0
    )
    
    print("\n--- Running Phase 1: Initial Tooling Allocation ---")
    sorted_ops = heuristic.phase_1_initial_allocation()
    
    print("--- Running Phase 2: Timeline Sequencing ---")
    obj, tardy, setups, schedule = heuristic.phase_2_scheduling(sorted_ops)
    
    print("\n================ RESULTS SUMMARY ================")
    print(f"Instance Handled: {instance_name}")
    print(f"Total Combined Objective Score: {obj:.2f} hours")
    print(f" -> Accumulated Tardiness Costs: {tardy:.2f} hours")
    print(f" -> Forced Tool Setup Adjustments: {setups} switches")
    print("=================================================")
    
    # Let's inspect what Machine 1's queue looks like as a small snapshot
    print(f"\nSample Pipeline Timeline for Machine 1 (First 5 jobs):")
    for task in schedule[1][:5]:
        print(f"  {task['job']} | Window: [{task['start']} -> {task['end']}] | Tardiness: {task['tardiness']} | Tool Switch: {task['tool_switch']}")

# Run the 2-Machine 38-Operation benchmark baseline instance
load_and_run_benchmark(instance_name="2M38_0.5R", num_machines=2, magazine_capacity=80)
