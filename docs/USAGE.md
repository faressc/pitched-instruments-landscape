<!--
Copyright 2024 tu-studio
This file is licensed under the Apache License, Version 2.0.
See the LICENSE file in the root of this project for details.
-->

# User Guide

## Adding new data to your DVC remote

To track and add new data inputs (e.g., `data/raw`):

```sh
dvc add data/raw
dvc push
```

>**Note**: Only necessary if you want to track new data inputs that are not already declared in the [dvc.yaml](../dvc.yaml) file as outputs of a stage.

> **Info**: The files added with DVC should be Git-ignored, but adding with DVC will automatically create .gitignore files. What Git tracks are references with a .dvc suffix (e.g. data/raw.dvc). Make sure you add and push the .dvc files to the Git remote.

## Update Docker Image / Dependencies

As your requirements change, always update [requirements.txt](../requirements.txt) with your fixed versions:

```sh
pip freeze > requirements.txt
```

Docker images are automatically rebuilt and pushed to Docker Hub by the GitHub workflow when [requirements.txt](../requirements.txt), [Dockerfile](../Dockerfile), or [docker_image.yml](../.github/workflows/docker_image.yml) are updated and pushed to GitHub. If you trigger an image build, ensure it is completed and pushed to Docker Hub before proceeding.

> **Note**: For the free `docker/build-push-action`, there is a 14GB storage limit for free public repositories on GitHub runners ([About GitHub runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners)). Therefore, the Docker image must not exceed this size.

On the HPC cluster, the Docker image is then automatically pulled and converted to a Singularity image when running the [slurm_job.sh](../slurm_job.sh) and no image is found within the repository.

```sh
sbatch slurm_job.sh
```

 If you want to force the update of the Singularity image, you can use the flag `--rebuild-container` or delete the existing image on the cluster when submitting the job.

```sh
sbatch slurm_job.sh --rebuild-container
```

## Launch ML Pipeline

### Locally natively or with Docker

To run the entire pipeline locally or locally within a docker container, execute the following commands:

```sh
# natively
./exp_workflow.sh
# with Docker
docker run --rm \
  --mount type=bind,source="$(pwd)",target=/home/app \
  --mount type=volume,source=ssh-config,target=/root/.ssh \
  <your_image_name> \
  /home/app/exp_workflow.sh
```

### On the HPC Cluster

Log into the High-Performance Computing (HPC) cluster using your SSH config and key and navigate to your repository. Substitute the appropriate names for the placeholders `<username>` and `<repository>`:

```sh
ssh hpc
cd /scratch/<username>/<repository>
git pull # optionally pull the latest changes you have done locally
```

Launch pipeline jobs either individually or in parallel. To launch multiple trainings at once with parameter grids or predefined parameter sets, modify `multi_submission.py`:

```sh
# submit a single Slurm job:
sbatch slurm_job.sh # optional args for slurm_job.sh
# submit multiple Slurm jobs at once:
venv/bin/python multi_submission.py # optional args for slurm_job.sh
```

## Monitoring and Logs

### SLURM Job Monitoring

Check the status of all jobs associated with your user account:

```sh
squeue -u <user_name>
```

Monitor SLURM logs in real-time:

```sh
cd logs/slurm
tail -f slurm-<slurm_job_id>.out
```

To kill a single job using the SLURM job id or all jobs per user:

```sh
# Per job id
scancel <slurm_job_id>
# Per user
scancel -u <user_name>
```

### Local Monitoring with TensorBoard

Use the [sync_logs.sh](../sync_logs.sh) script to sync logs on your local machine in the `logs/` directory every 30 seconds:

```sh
./sync_logs.sh
```

Then open a new terminal, launch TensorBoard to monitor experiments:

```sh
tensorboard --logdir=logs/tensorboard
```

Access TensorBoard via your browser at:

```text
localhost:6006
```

> **Tip**: You can also view TensorBoard logs in VSCode using the official extension.

### Remote Monitoring with TensorBoard

To start TensorBoard remotely on the SSH Host and access it in your browser:

```sh
tensorboard --logdir=<TUSTU_TENSORBOARD_HOST_DIR>/<TUSTU_PROJECT_NAME>/logs/tensorboard --path_prefix=/tb1
```

> **Note**: For an overview of all DVC experiments, it is important to start TensorBoard on the collected logs folder tensorboard/, where all experiments are organized in subdirectories.

Access TensorBoard via your browser at:

```text
<your_domain>/tb1
```

## Troubleshooting

If the [exp_workflow.sh](../exp_workflow.sh) did not run through all steps, the temporary subdirectory in `tmp/` in the root of the repository, will not be deleted. If for example the `dvc exp push origin` failed, you can `cd` into the subdirectory in `tmp/` and manually try to push the experiment again:

```sh
cd tmp/<experiment_subdirectory>
dvc exp push origin
```

## DVC Experiment Retrieval

Each time we run the pipeline, DVC creates a new experiment. These are saved as custom Git references that can be retrieved and applied to your workspace. These references do not appear in the Git log, but are stored in the `.git/refs/exps` directory and can be pushed to the remote Git repository. This is done automatically at the end of the [exp_workflow.sh](../exp_workflow.sh) with `dvc exp push origin`. All outputs and dependencies are stored in the `.dvc/cache` directory and pushed to the remote DVC storage when the experiment is pushed. Since we create a new temporary copy of the repository for each pipeline run (and delete it at the end), the experiments will not automatically appear in the main repository.

To retrieve, view, and apply an experiment, do the following (either locally or on the HPC cluster):

```sh
# Get all experiments from remote
dvc exp pull origin
# List experiments
dvc exp show
# Apply a specific experiment
dvc exp apply <dvc_exp_name>
```

> **Note**: By default, experiments are tied to the specific Git commit of their execution. Therefore the commands `dvc exp pull origin` and `dvc exp show` only work for experiments associated with the same commit used when the experiment was created. To pull and show experiments from a different or all commits, you can use specific flags as outlined in the [DVC documentation](https://dvc.org/doc/command-reference/experiments).

> **Tip**: You can also get the Git ref hash of the experiment from `dvc exp show` and do a `git diff`.

## Clean Up

To clean up copies of repositories of failed experiments, use this command from the root of your repository:

```sh
rm -rf tmp/
```

For information on cleaning up the DVC cache, refer to the [DVC Documentation](https://dvc.org/doc/command-reference/gc).

> **Note**: Be careful with this, as we are using a shared cache between parallel experiment runs.

## Other Notes

- If you are using multiple workes in a `torch.data.utils.DataLoader`, there may be issues because of too many open files descriptors in your process. You can increase the limit of open files descriptors with the following command:

```sh
# View current limit
ulimit -n
# Increase limit
ulimit -n 262144
```

> Note: This is a temporary change and will be reset after the session ends. On the hpc you can only change this once per session to a higher limit.