import subprocess

commands = [
    ".\\venv\\Scripts\\python.exe src/run_all_experiments.py",
    ".\\venv\\Scripts\\python.exe src/run_lr_scaled.py",
    ".\\venv\\Scripts\\python.exe src/run_boundary_fidelity.py",
    ".\\venv\\Scripts\\python.exe src/run_calibration.py"
]

for cmd in commands:
    print(f"Running {cmd}...")
    subprocess.run(cmd.split())

print("All scripts finished!")
