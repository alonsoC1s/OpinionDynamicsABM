#!/bin/zsh
#SBATCH --job-name=benchmarks_abm     # create a short name for your job
#SBATCH --chdir=/home/htc/amartine/OpinionDynamicsABM.jl
#SBATCH --output=out/slurm-%A.%a.out # stdout file
#SBATCH --error=out/err/slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=2:50:50          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=martinez@zib.de
#SBATCH --mail-type=all          # send email on job start, end and fault

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)
echo "\n\n\n"
# -r $(git log --pretty=format:'%h' -n 7 | tr '\n' ',') \

/scratch/htc/amartine/julia/bin/benchpkg \
    -r vUseFill,vReuseBuffers,vTmpBuffer,v@inbounds,vNoNorm,vNoReturn,vExpFast \
    -o tmp/benchresults \
    --exeflags="--threads auto -O3" \
    --tune

/scratch/htc/amartine/julia/bin/benchpkgplot OpinionDynamicsABM \
    -r vUseFill,vReuseBuffers,vTmpBuffer,v@inbounds,vNoNorm,vNoReturn,vExpFast \
    -i tmp/benchresults \
    -o tmp/ \
