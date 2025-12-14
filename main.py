import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] {script_name} finished in {elapsed:.2f} seconds.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_name} failed with exit code {e.returncode}.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {script_name}")
        sys.exit(1)

if __name__ == "__main__":
    scripts = [
        "01_Data_Prep.py",
        "02_Visualizations.py",
        "03_Model_XGBoost.py",
        "04_Deep_Insights.py"
    ]
    
    total_start = time.time()
    
    for script in scripts:
        run_script(script)
        
    total_time = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"All tasks completed successfully in {total_time:.2f} seconds!")
    print(f"{'#'*60}")