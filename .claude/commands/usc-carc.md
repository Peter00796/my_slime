**Purpose**: USC CARC (Center for Advanced Research Computing) cluster guide — SSH access, job submission, storage, modules, and common workflows.

---

## Cluster Overview

USC operates two HPC clusters managed by CARC:
- **Discovery** — general research cluster
- **Endeavour** — condo/partition-based cluster (used by our group)

Documentation: https://www.carc.usc.edu/user-information/user-guides

---

## SSH Access

### Login Nodes
```
Host: usc-endeavour
HostName: endeavour2.usc.edu
User: muruzhan
```
- **Requires**: USC VPN active (Cisco AnyConnect), or internal IP `10.72.0.16`
- Login nodes are for file management and job submission only — do NOT run compute on login nodes

### SSH Config (local `~/.ssh/config`)
```
Host usc-endeavour
    HostName endeavour2.usc.edu
    User muruzhan
    ForwardAgent yes
```

### Connecting
```bash
ssh usc-endeavour
```

---

## File Systems & Storage

| Path | Purpose | Quota | Backed Up |
|------|---------|-------|-----------|
| `/home1/muruzhan/` | Home directory | 100 GB | Yes |
| `/project2/swabhas_1625/` | Project storage (group shared) | 10 TB | No |
| `/scratch1/muruzhan/` | Scratch (fast, temporary) | 10 TB | No, purged after 14 days |

- **Use project storage** (`/project2/`) for datasets, checkpoints, and code
- **Use scratch** for large temporary files (intermediate outputs, logs)
- Check quota: `myquota` or `lfs quota -u muruzhan /project2/`

---

## Module System

CARC uses Lmod for software management.

```bash
module avail              # List all available modules
module load gcc/11.3.0    # Load a specific module
module load cuda/11.8.0   # Load CUDA
module list               # Show loaded modules
module purge              # Unload all modules
```

### Common Modules
```bash
module load gcc/11.3.0
module load cuda/11.8.0
module load cudnn/8.6.0.163-11.8
module load python/3.11.3
```

### Conda / Micromamba
If using micromamba (preferred for speed):
```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate <env_name>
```

---

## SLURM Job Scheduler

### Key Commands
```bash
squeue -u muruzhan          # View your jobs
squeue -p nlp               # View partition queue
scancel <jobid>             # Cancel a job
scancel -u muruzhan         # Cancel all your jobs
sacct -j <jobid>            # Job accounting info
sinfo -p nlp                # Partition node status
```

### Interactive Session (srun)
```bash
srun --partition=nlp \
     --gres=gpu:4 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=02:00:00 \
     --exclude=c05-05,c06-05 \
     --pty bash
```

### Batch Job (sbatch)
```bash
sbatch my_script.sh
```

#### sbatch Script Template
```bash
#!/bin/bash
#SBATCH --partition=nlp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=my-job
#SBATCH --exclude=c05-05,c06-05
#SBATCH --output=slurm_%j.log

# Environment setup
eval "$(micromamba shell hook --shell bash)"
micromamba activate <env_name>

# Your commands here
python train.py
```

### Common SBATCH Directives
| Directive | Example | Description |
|-----------|---------|-------------|
| `--partition` | `nlp`, `gpu`, `main` | Which partition to use |
| `--gres` | `gpu:4`, `gpu:a100:2` | GPU resource request |
| `--cpus-per-task` | `8` | CPU cores per task |
| `--mem` | `64G` | Memory request |
| `--time` | `04:00:00` | Wall time limit (HH:MM:SS) |
| `--nodes` | `1` | Number of nodes |
| `--ntasks` | `1` | Number of tasks |
| `--exclude` | `c05-05,c06-05` | Nodes to avoid |
| `--nodelist` | `d05-08` | Request specific node |
| `--output` | `slurm_%j.log` | Stdout file (%j = job ID) |
| `--error` | `slurm_%j.err` | Stderr file |
| `--mail-type` | `END,FAIL` | Email notifications |

---

## Partitions (Endeavour)

| Partition | GPUs | Max Time | Notes |
|-----------|------|----------|-------|
| `nlp` | A100 (40GB/80GB) | varies | Group condo, may preempt |
| `gpu` | Mixed | 48h | General GPU partition |
| `main` | None (CPU only) | 48h | CPU-only jobs |

- **Preemption**: The `nlp` partition can preempt lower-priority jobs. If your job gets killed, just resubmit.
- Check availability: `sinfo -p nlp -N -l`

---

## GPU Usage

### Check GPU availability
```bash
sinfo -p nlp -N -l          # Node status in partition
nvidia-smi                    # On a compute node only
```

### Known Bad Nodes
- **c05-05, c06-05** — CUDA Error 802 (broken drivers). Always exclude these:
  ```
  #SBATCH --exclude=c05-05,c06-05
  ```

### SSH to Allocated Node
If you already have an allocation, you can SSH directly to the compute node:
```bash
squeue -u muruzhan            # Find your node (e.g., d05-08)
ssh d05-08                    # SSH to it
```

---

## Data Transfer

### SCP (from local machine)
```bash
scp local_file.tar.gz muruzhan@endeavour2.usc.edu:/project2/swabhas_1625/muruzhan/
```

### rsync (preferred for large transfers)
```bash
rsync -avz --progress local_dir/ muruzhan@endeavour2.usc.edu:/project2/swabhas_1625/muruzhan/remote_dir/
```

### Within cluster
```bash
cp -r /scratch1/muruzhan/data /project2/swabhas_1625/muruzhan/data
```

---

## Common Workflows

### Submit and monitor a job
```bash
sbatch my_script.sh                    # Submit
squeue -u muruzhan                     # Check status
tail -f slurm_<jobid>.log             # Watch output
scancel <jobid>                        # Cancel if needed
```

### Check job history and efficiency
```bash
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize
```

### Environment variables to set
```bash
export HF_HOME=/project2/swabhas_1625/muruzhan/<your_dir>/hf_cache   # Hugging Face cache
export WANDB_DIR=/project2/swabhas_1625/muruzhan/<your_dir>/wandb     # WandB logs
export TMPDIR=/scratch1/muruzhan/tmp                                    # Temp directory
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA Error 802` | Bad node — add to `--exclude` and resubmit |
| Job preempted | Normal on shared partitions — resubmit |
| OOM (Out of Memory) | Increase `--mem` or reduce batch size |
| GPU OOM | Reduce model/batch size, enable gradient checkpointing |
| SSH timeout | Check VPN is connected |
| `module not found` | Run `module avail` to find correct name/version |
| Disk quota exceeded | Clean up `/home1/` or request more project space |
| Job pending forever | Check `sinfo -p nlp` — nodes may be down or fully allocated |
