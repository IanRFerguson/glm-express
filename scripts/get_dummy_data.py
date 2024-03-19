from nilearn.datasets import fetch_development_fmri
import logging
import os

logger = logging.getLogger(__name__)

##########


def main(n_subjects: int, data_dir: str):
    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory [path={data_dir}]")

    fetch_development_fmri(n_subjects=n_subjects, data_dir=data_dir)
    logger.info(f"Successfully retrieved {n_subjects}")


#####

if __name__ == "__main__":
    SUBJECTS = os.environ.get("SUBJECTS", 10)
    DIRECTORY = os.path.abspath(os.environ.get("DIRECTORY", "../project_data"))

    main(n_subjects=SUBJECTS, data_dir=DIRECTORY)
