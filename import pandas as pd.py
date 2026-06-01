import os
import glob
import re
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
        
        for op in O_hat:
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
            
        objective_val = (self.wd * total_tardiness) + (self.ws * self.tau * total_setups)
        return objective_val, total_tardiness, total_setups


# =====================================================================
# 2. FILE DETECTOR & SORTING UTILITIES
# =====================================================================
def get_scenario_sort_key(filepath):
    """
    Extracts the variable numerical value from the filename suffix for logical sorting.
    e.g., '6M163_-2D.csv' -> splits to '-2D.csv' -> extracts -2.0
    """
    basename = os.path.basename(filepath)
    parts = basename.split('_')
    if len(parts) >= 2:
        variable_part = parts[1]
        # Match signed float or integer
        match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", variable_part)
        if match:
            return float(match.group())
    return 0.0


def run_heuristic_file(filepath):
    """
    Parses a single local CSV case file, reads its metadata, and runs the PH.
    """
    if not os.path.exists(filepath):
        return None

    # Reset seed for consistent evaluation
    random.seed(42)

    num_machines = 2       
    magazine_capacity = 80 
    
    # Try parsing machine and capacity config from metadata headers
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

    heuristic = PractitionerHeuristic(
        jobs_data=jobs_data, 
        num_machines=num_machines, 
        magazine_capacity=magazine_capacity,
        tool_setup_time=1.0, 
        theta_m=72.0
    )
    
    sorted_ops = heuristic.phase_1_initial_allocation()
    obj, tardy, setups = heuristic.phase_2_scheduling(sorted_ops)
    
    # Extract parametric label (e.g. "0.24R")
    basename = os.path.basename(filepath)
    param_label = "Base"
    parts = basename.split('_')
    if len(parts) >= 2:
        param_label = parts[1].replace(".csv", "")

    return {
        "Instance": param_label,
        "File": basename,
        "M": num_machines,
        "C": magazine_capacity,
        "Objective": round(obj, 2),
        "Tardiness": round(tardy, 2),
        "Setups": setups
    }


# =====================================================================
# 3. DIRECTORY BATCH RUNNER
# =====================================================================
if __name__ == "__main__":
    # Base path where all problem directories are placed
    problem_folders = ["2M38", "2M46", "6M140", "6M163"]

    for folder_name in problem_folders:
        folder_path = folder_name
        if not os.path.exists(folder_path):
            print(f"Directory omitted (does not exist): {folder_path}")
            continue
            
        print("=" * 80)
        print(f" PROCESSING BASE CASE AND SCENARIOS FOR: {folder_name} ".center(80, "#"))
        print("=" * 80)

        # 1. Run Base Instance
        base_file_pattern = os.path.join(folder_path, f"{folder_name}.csv")
        base_files = glob.glob(base_file_pattern)
        if base_files:
            print("\n[Base Instance]")
            base_res = run_heuristic_file(base_files[0])
            if base_res:
                print(pd.DataFrame([base_res]).to_string(index=False))
        
        # 2. Run Scenarios (Scenario1, Scenario2, Scenario3)
        for s_idx in [1, 2, 3]:
            scenario_dir = os.path.join(folder_path, f"Scenario{s_idx}")
            if not os.path.exists(scenario_dir):
                continue
                
            # Discover and sort scenario CSV files
            csv_files = glob.glob(os.path.join(scenario_dir, "*.csv"))
            sorted_files = sorted(csv_files, key=get_scenario_sort_key)
            
            results = []
            for filepath in sorted_files:
                res = run_heuristic_file(filepath)
                if res:
                    results.append(res)
            
            if results:
                print(f"\n[Scenario {s_idx} - {folder_name}]")
                df_scenario = pd.DataFrame(results)
                print(df_scenario.to_string(index=False))
        
        print("\n" + "=" * 80 + "\n")