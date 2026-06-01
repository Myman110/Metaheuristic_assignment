import os
import pandas as pd
import random

# =====================================================================
# 1. THE HEURISTIC ENGINE
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
# 2. LOCAL FILE LOADING & EXECUTION
# =====================================================================
def load_and_run_local_benchmark(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file path does not exist: {filepath}")

    # Step A: Parse metadata rows at the top of the CSV (Lines 1-4)
    num_machines = 2       # Default fallback
    magazine_capacity = 80 # Default fallback
    
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
    except Exception as e:
        print(f"Could not parse file metadata: {e}. Proceeding with defaults.")

    # Step B: Read the operations data. Skip first 5 metadata lines.
    df = pd.read_csv(filepath, skiprows=5)
    
    # Rename dataset columns to align with internal engine variables
    df.columns = ['job_id', 'op_id', 'r', 'p', 'd', 'tool_set', 'size']
    jobs_data = df.to_dict(orient='records')
    
    print(f"\nFile: {os.path.basename(filepath)}")
    print(f"Loaded {len(jobs_data)} scheduling operations.")
    print(f"Configuration -> Machines: {num_machines} | Magazine Capacity: {magazine_capacity}")

    # Step C: Initialize Heuristic
    heuristic = PractitionerHeuristic(
        jobs_data=jobs_data, 
        num_machines=num_machines, 
        magazine_capacity=magazine_capacity,
        tool_setup_time=1.0, 
        theta_m=72.0
    )
    
    sorted_ops = heuristic.phase_1_initial_allocation()
    obj, tardy, setups, schedule = heuristic.phase_2_scheduling(sorted_ops)
    
    print("---------------------------------------------")
    print(f"Combined Objective Score: {obj:.2f} hours")
    print(f"Accumulated Tardiness:    {tardy:.2f} hours")
    print(f"Tool Setup Adjustments:   {setups} switches")
    print("---------------------------------------------")
    
    # Timeline overview for Machine 1
    if 1 in schedule and schedule[1]:
        print("Sample timeline (Machine 1 - First 3 jobs):")
        for task in schedule[1][:3]:
            print(f"  {task['job']} | Window: [{task['start']} -> {task['end']}] | Tardiness: {task['tardiness']} | Switch: {task['tool_switch']}")
    print("=============================================\n")
    return obj, tardy, setups


# =====================================================================
# 3. RUNNING EXAMPLES
# =====================================================================
if __name__ == "__main__":
    # Define mapped paths
    paths = {
        "Base Case (2M38)": r"2M38\2M38.csv",
        "Scenario 1 (Tool Ratio)": r"2M38\Scenario1\2M38_0.37R.csv",
        "Scenario 2 (Capacity)": r"2M38\Scenario2\2M38_53C.csv",
        "Scenario 3 (Deadline Shift)": r"2M38\Scenario3\2M38_+1D.csv"
    }

    # Execute all defined cases
    for name, filepath in paths.items():
        print(f"Evaluating: {name}")
        try:
            load_and_run_local_benchmark(filepath)
        except Exception as e:
            print(f"Failed to process case due to: {e}\n")