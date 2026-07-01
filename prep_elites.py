import csv
import json
import os

def extract_elites():
    # Update this path if necessary!
    csv_path = r"E:\Development_back_up_folder_2026\long_run data back up\merged_parameters_results.csv"
    output_path = "elite_128_payload.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    runs = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # The SSE column in your CSV is 'log_prime_sse'
                sse = float(row.get("log_prime_sse", 9999.0))
                
                # Only keep the absolute elites
                if sse < 5.0:
                    params = {
                        "param_D": float(row.get("param_D", 1.0)),
                        "param_eta": float(row.get("param_eta", 0.1)),
                        "param_rho_vac": float(row.get("param_rho_vac", 0.0)),
                        "param_a_coupling": float(row.get("param_a_coupling", 0.0)),
                        "param_splash_coupling": float(row.get("param_splash_coupling", 0.0)),
                        "param_splash_fraction": float(row.get("param_splash_fraction", 0.0))
                    }
                    runs.append((sse, params))
            except ValueError:
                continue

    # Sort by error ascending (lowest SSE goes first)
    runs.sort(key=lambda x: x[0])
    
    elite_payload = [r[1] for r in runs]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(elite_payload, f, indent=2)
        
    print(f"✅ Successfully extracted {len(elite_payload)} Elite runs (SSE < 5) to {output_path}")
    print(f"🏆 Best Run SSE: {runs[0][0]}")

if __name__ == "__main__":
    extract_elites()