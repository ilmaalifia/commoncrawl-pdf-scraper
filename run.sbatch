#!/bin/bash
SBATCH -J pdf-scraper
SBATCH -c 8        
# SBATCH --mail-type=ALL       
# SBATCH --mail-user=<your email>
SBATCH --output=job_test_%j.log   

# Create a directory for this job on the node
ScratchDir="/local/${SLURM_JOBID}"
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' already found !"
else
   echo "'$ScratchDir' not found, creating !"
   mkdir $ScratchDir
fi
cd $ScratchDir

# Copy input and executable to the node
cp -r ${SLURM_SUBMIT_DIR}/pdf-scraper/* $ScratchDir

# Manage environment 
module load miniconda3/3.8
if [ ! -d "$ScratchDir/.conda/envs/pdf-scraper" ]; then
   conda create -n pdf-scraper python=3.13.0 -y
   source $ScratchDir/.bashrc
fi
conda activate pdf-scraper
pip install -r --no-cache-dir $ScratchDir/requirements.txt

# It's nice to have some information logged for debugging
echo "Date              : $(date)"
echo "Hostname          : $(hostname -s)"
echo "Working Directory : $(pwd)"
echo "Starting worker   : "

caseName=${PWD##*/} # to distinguish several log files
# Run the job -- make sure that it terminates itself before time is up
# Do not submit into the background (i.e. no & at the end of the line).
python src/main.py --index-name=CC-MAIN-2025-13

# Copy output back to the master, comment with # if not used
cp failed_urls.json ${SLURM_SUBMIT_DIR}/pdf-scraper
mv output ${SLURM_SUBMIT_DIR}/pdf-scraper

# Clean up on the compute node!
cd ~
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' found and removing files."
   rm -rf $ScratchDir
else
   echo "Warning: '$ScratchDir' NOT found."
fi