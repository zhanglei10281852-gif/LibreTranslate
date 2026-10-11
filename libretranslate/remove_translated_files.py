import atexit
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from libretranslate import file_jobs

# How long published translation results are kept available for download.
FILE_RETENTION_MINUTES = 30


def remove_translated_files(upload_dir: str):
    """Reclaim expired/orphaned file translation products.

    Even though every worker schedules this job, :func:`file_jobs.sweep`
    elects a single reaper, so concurrent workers never perform duplicate or
    conflicting deletions.
    """
    file_jobs.sweep(upload_dir)


def setup(upload_dir):
    file_jobs.init_store(upload_dir)

    scheduler = BackgroundScheduler(daemon=True, timezone='UTC')
    scheduler.add_job(
        remove_translated_files,
        "interval",
        minutes=FILE_RETENTION_MINUTES,
        kwargs={'upload_dir': upload_dir},
        # Reclaim leftovers from previous runs as soon as a worker starts.
        next_run_time=datetime.now(),
    )
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
