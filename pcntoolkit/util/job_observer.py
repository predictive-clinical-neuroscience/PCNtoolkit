import copy
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List

from IPython.display import clear_output

from pcntoolkit.util.output import Messages, Output, Warnings


@dataclass
class JobStatus:
    job_id: str
    name: str
    state: str
    time: str
    nodes: str
    success_file_exists: bool = False


class JobObserver:
    def __init__(
        self,
        active_job_ids: Dict[str, str],
        job_type: str = "local",
        log_dir: str = "logs",
        task_id: str = "",
    ):
        self.all_job_ids = copy.deepcopy(active_job_ids)
        self.active_job_ids = copy.deepcopy(active_job_ids)
        self.job_type = job_type
        self.log_dir = log_dir
        self.task_id = task_id
        # Reverse mapping from job_id to job_name for looking up success files
        self.job_id_to_name = {v: k for k, v in active_job_ids.items()}

    def check_success_file(self, job_name: str) -> bool:
        """Check if a success file exists for the given job name."""
        success_file = os.path.join(self.log_dir, f"{job_name}.success")
        max_retries = 10
        retry_delay = 1  # seconds

        for _ in range(max_retries):
            if os.path.exists(success_file):
                try:
                    # Try to open the file to ensure it's fully written
                    with open(success_file, "r") as f:
                        f.read()
                    return True
                except (IOError, OSError):
                    pass
            time.sleep(retry_delay)
        return False

    def get_job_statuses(self) -> List[JobStatus]:
        """Get status of all tracked jobs."""
        # Get all job statuses at once
        if self.job_type == "slurm":
            process = subprocess.Popen(
                ["squeue", "--format=%i|%j|%T|%M|%N", "--noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()
        elif self.job_type == "torque":
            process = subprocess.Popen(
                ["qstat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()
            stdout = "\n".join(stdout.split("\n")[2:])

        statuses = []
        if process.returncode is None or process.returncode == 0:
            # Split output into lines and filter out empty lines
            lines = [line.strip() for line in stdout.split("\n") if line.strip()]
            for line in lines:
                try:
                    if self.job_type == "slurm":
                        job_id, name, state, time, nodes = line.split("|")
                    elif self.job_type == "torque":
                        job_id, name, _, time, state, _ = line.split(" ")
                        nodes = "?"
                    if job_id in list(self.all_job_ids.values()):
                        job_name = self.job_id_to_name.get(job_id)
                        if state not in ["RUNNING", "PENDING", "COMPLETING"]:
                            success_exists = (
                                job_name is not None
                            ) and self.check_success_file(job_name)
                        else:
                            success_exists = False
                        statuses.append(
                            JobStatus(
                                job_id=job_id,
                                name=name,
                                state=state,
                                time=time,
                                nodes=nodes,
                                success_file_exists=success_exists,
                            )
                        )
                except ValueError as e:
                    Output.warning(
                        Warnings.ERROR_PARSING_JOB_STATUS_LINE, line=line, error=e
                    )
        elif stderr:
            Output.warning(Warnings.ERROR_GETTING_JOB_STATUSES, stderr=stderr)

        # Any jobs not in the process list are assumed to be completed.
        # Check if they have a success file
        job_ids_set = set(self.all_job_ids.values())
        found_job_ids = {status.job_id for status in statuses}
        for job_id in job_ids_set - found_job_ids:
            job_name = self.job_id_to_name.get(job_id)
            if job_name:
                success_exists = self.check_success_file(job_name)
                state = "COMPLETED" if success_exists else "FAILED"
                statuses.append(
                    JobStatus(
                        job_id=job_id,
                        name=job_name,
                        state=state,
                        time="",
                        nodes="",
                        success_file_exists=success_exists,
                    )
                )
        return statuses

    def wait_for_jobs(self, check_interval=1):
        """Wait for all submitted jobs to complete.

        Args:
            check_interval (int): Time in seconds between job status checks
        """
        # Detect if we're running in a Jupyter notebook
        in_notebook = True
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell not in ["ZMQInteractiveShell", "Shell"]:
                in_notebook = False
        except NameError:
            in_notebook = False

        # Give the jobs time to start
        time.sleep(5)
        statuses = self.get_job_statuses()
        while any(
            status.state in ["RUNNING", "PENDING", "COMPLETING"] for status in statuses
        ):
            self.show_job_status_monitor(in_notebook, statuses)
            time.sleep(check_interval)
            statuses = self.get_job_statuses()
        time.sleep(check_interval)
        statuses = self.get_job_statuses()
        self.show_job_status_monitor(in_notebook, statuses)

    def show_job_status_monitor(self, in_notebook, statuses):
        show_pid = Output.get_show_pid()
        show_messages = Output.get_show_messages()
        show_timestamp = Output.get_show_timestamp()

        Output.set_show_pid(False)
        Output.set_show_messages(True)
        Output.set_show_timestamp(False)
        if in_notebook:
            clear_output(wait=True)
        Output.print(Messages.JOB_STATUS_MONITOR, task_id=self.task_id)
        for status in sorted(statuses, key=lambda x: x.job_id):
            Output.print(
                Messages.JOB_STATUS_LINE,
                status.job_id,
                status.name,
                status.state,
                status.time,
                status.nodes,
            )

            # Count completed, failed, and active jobs
        completed_jobs, failed_jobs, active_jobs = 0, 0, 0
        for job_name, job_id in sorted(
            list(self.active_job_ids.items()), key=lambda x: x[0]
        ):
            matching_statuses = [s for s in statuses if s.job_id == job_id]
            if len(matching_statuses) > 1:
                Output.warning(
                    Warnings.MULTIPLE_JOBS_FOUND_FOR_JOB_ID,
                    job_id=job_id,
                    job_name=job_name,
                )
            elif len(matching_statuses) == 1:
                my_status = matching_statuses[0]
                if my_status.state == "COMPLETED":
                    completed_jobs += 1
                elif my_status.state == "FAILED":
                    failed_jobs += 1
                else:
                    active_jobs += 1

        Output.print(
            Messages.JOB_STATUS_SUMMARY,
            total_completed_jobs=completed_jobs,
            total_active_jobs=active_jobs,
            total_failed_jobs=failed_jobs,
        )

        if not any(
            status.state in ["RUNNING", "PENDING", "COMPLETING"] for status in statuses
        ):
            Output.print(Messages.NO_MORE_RUNNING_JOBS)
        else:
            if not in_notebook:
                print(f"\033[{len(statuses) + 16}A", end="")

        Output.set_show_messages(show_messages)
        Output.set_show_pid(show_pid)
        Output.set_show_timestamp(show_timestamp)
