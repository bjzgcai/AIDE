from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """
    The class to store the dataset information from HF Hub.
    """

    name: str
    subset: str  # some datasets have subset name, e.g. "coco" in "coco-2017"
    number_of_rows: (
        int | str
    )  # sometimes can be N/A or '10M<n<100M', or just an integer
    columns: list[str]
    description: str
    hf_tags: list[str]
    modalities: list[str]
    license: str
    n_likes: int
    n_downloads_last_month: int
