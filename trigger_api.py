import json
import urllib.request

stage_data = {
    "hunt_name": "INGEST_563", "generations": 563, "batch_size": 1,
    "population_size": 563, "seeds_per_candidate": 1, "n_grid": 128,
    "t_steps": 1200, "dt": 0.005, "origin": "UI_CONTROL",
    "mode": "backlog", "backlog_source": "phase1_input.json"
}

print("1. Sending STAGE payload to API...")
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/control/stage",
    data=json.dumps(stage_data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    resp_data = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    staged_path = resp_data.get("staged_path")
    print("✅ Scaled parameters generated at:", staged_path)

    print("\n=======================================================")
    print("NEXT STEP -> Paste this EXACT command into Terminal 2:")
    print(f"g:/quantule_mapper/.venv/Scripts/python.exe orchestrator/orchestrator_service.py --config {staged_path}")
    print("=======================================================\n")

except Exception as e:
    print("❌ API Request Failed:", e)
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))
