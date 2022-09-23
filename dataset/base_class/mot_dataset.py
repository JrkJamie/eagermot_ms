from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Set

import dataset.base_class.mot_sequence as mot_sequence


class MOTDataset(ABC):
    def __init__(self, work_dir: str, det_source: str, seg_source: str):
        """
        base class for the dataset, used to detect whether dataset split exist

        Args:
            work_dir (str): path to workspace output directory
            det_source (str): source of 3D detections
            seg_source (str): source of 2D detections

        Examples:
            >>> super().__init__(work_dir, det_source, seg_source)  # recommended to inherit this class
        """
        self.work_dir = work_dir
        self.det_source = det_source  # see dataset specific classes e.g. mot_kitti
        self.seg_source = seg_source  # see dataset specific classes e.g. mot_kitti
        self.splits: Set[str] = set()

    def assert_split_exists(self, split: str) -> None:
        """
        verify whether split exists in dataset

        Args:
            split (str): input split

        Returns:
            None
        """
        assert split in self.splits, f"There is no split {split}"

    def assert_sequence_in_split_exists(self, split: str, sequence_name: str) -> None:
        """
        verify whether sequence exists in dataset

        Args:
            split (str): split name
            sequence_name (str): sequence_name

        Returns:
            None
        """
        self.assert_split_exists(split)
        assert sequence_name in self.sequence_names(split), f"There is no sequence {sequence_name} in split {split}"

    @abstractmethod
    def sequence_names(self, split: str) -> List[str]:
        """
        Return list of sequences in the split
        """
        pass

    @abstractmethod
    def get_sequence(self, split: str, sequence_name: str) -> mot_sequence.MOTSequence:
        """
        Return a sequence object by split-name combo
        """
        pass

    @abstractmethod
    def save_all_mot_results(self, folder_name: str) -> None: pass
