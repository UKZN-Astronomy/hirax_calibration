#!/bin/sh
#SBATCH --job-name=many_corrcal_runs   # Job name
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=20
#SBATCH --mem=64000
#SBATCH --time=24:00:00
#SBATCH --output=array_%A-%a.out    # Standard output and error log
#SBATCH --array=1-5                 # Array range

#Set the number of runs that each SLURM task should do
PER_TASK=100

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
  python run_corrcal.py $run
done

python collect_arrays.py

