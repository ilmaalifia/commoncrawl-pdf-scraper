#!/bin/bash
#SBATCH -J pdf-scraper
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email@example.com>
#SBATCH --output=pdf_scraper_%j.log

export HTTP_PROXY=<proxy>
export HTTPS_PROXY=<proxy>
export http_proxy=<proxy>
export https_proxy=<proxy>

# Create a directory for this job on the node
ScratchDir="/local/${SLURM_JOBID}"
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' already found!"
else
   echo "'$ScratchDir' not found, creating!"
   mkdir $ScratchDir
fi
cd $ScratchDir

# Copy input and executable to the node
cp -r ${SLURM_SUBMIT_DIR}/* $ScratchDir
cp ${SLURM_SUBMIT_DIR}/.env $ScratchDir
ls -a $ScratchDir # debug

# Manage environment
module load miniconda3/3.8
conda env list # debug

# It's nice to have some information logged for debugging
echo "Date              : $(date)"
echo "Hostname          : $(hostname -s)"
echo "Working Directory : $(pwd)"
echo "Starting worker   : "

caseName=${PWD##*/} # to distinguish several log files
# Run the job -- make sure that it terminates itself before time is up
# Do not submit into the background (i.e. no & at the end of the line).
srun -N1 -n1 --exclusive python src/main.py --index-name=CC-MAIN-2025-13 --url="*.gov" &
srun -N1 -n1 --exclusive python src/main.py --index-name=CC-MAIN-2025-13 --url="*.com"

# Copy output back to the master
cp *.json ${SLURM_SUBMIT_DIR}/pdf-scraper

# Clean up on the compute node!
cd ~
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' found and removing files."
   rm -rf $ScratchDir
else
   echo "Warning: '$ScratchDir' NOT found."
fi