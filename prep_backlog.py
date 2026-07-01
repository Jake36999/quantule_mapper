import csv
import json
import os

def generate_backlog(csv_file, output_json):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    backlog = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Extract physics parameters into a flat dictionary
                job_params = {
                    "param_D": float(row.get("param_D", 1.0)),
                    "param_eta": float(row.get("param_eta", 0.1)),
                    "param_rho_vac": float(row.get("param_rho_vac", 0.0))
                }
                # Add a, s, f if they are expected by the solver (default to 0.0)
                job_params["param_a"] = float(row.get("param_a", 0.0))
                job_params["param_s"] = float(row.get("param_s", 0.0))
                job_params["param_f"] = float(row.get("param_f", 0.0))
                
                backlog.append(job_params)
            except ValueError as e:
                print(f"Skipping invalid row: {e}")
                continue

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(backlog, f, indent=2)
    
    print(f"Successfully generated {len(backlog)} flat backlog jobs in {output_json}")

if __name__ == "__main__":
    # Ensure this points to your specific long-run data backup path
    csv_path = r"E:\Development_back_up_folder_2026\long_run data back up\merged_parameters_results.csv"
    output_path = "phase1_input.json"
    generate_backlog(csv_path, output_path)