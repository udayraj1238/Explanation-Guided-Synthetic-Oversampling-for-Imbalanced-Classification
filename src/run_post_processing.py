import subprocess

commands = [
    ".\\venv\\Scripts\\python.exe src/statistical_tests.py",
    ".\\venv\\Scripts\\python.exe src/gen_table2_mean_ranks.py",
    ".\\venv\\Scripts\\python.exe src/make_figures.py",
    ".\\venv\\Scripts\\python.exe src/run_tuning_stats.py",
    ".\\venv\\Scripts\\python.exe src/run_fidelity_stats.py",
    ".\\venv\\Scripts\\python.exe src/run_calibration_stats.py"
]

for cmd in commands:
    print(f"Running {cmd}...")
    subprocess.run(cmd.split())

print("All post-processing finished!")
