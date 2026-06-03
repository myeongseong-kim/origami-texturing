"""Project configuration objects."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from origami_texturing import paths


@dataclass(frozen=True)
class ProjectConfig:
    """Common paths used by notebooks and scripts."""

    root_dir: Path = paths.ROOT_DIR
    data_dir: Path = paths.DATA_DIR
    raw_data_dir: Path = paths.RAW_DATA_DIR
    processed_data_dir: Path = paths.PROCESSED_DATA_DIR
    external_dir: Path = paths.EXTERNAL_DIR
    inputs_dir: Path = paths.INPUTS_DIR
    output_dir: Path = paths.OUTPUT_DIR


def get_openai_client() -> OpenAI:
    """Return an OpenAI client configured from environment variables."""

    load_dotenv(paths.ROOT_DIR / ".env")
    return OpenAI()
