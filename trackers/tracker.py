""" Definition of object tracking abstractions """

from typing import Iterable, Optional, Type, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import supervision as sv


class NoPredictSample(Exception):
    pass

class NoPredictFrames(Exception):
    pass


class Object(ABC):
    @classmethod
    def from_json(cls, x: dict | list[dict]) -> "Object":
        pass

    def serialize(self) -> dict | list[dict]:
        pass

    def draw(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        pass


@dataclass
class TrackingResults:
    predictions: list[Object] = field(default_factory=lambda: [])
    sample_predictions: list[Object] = field(default_factory=lambda: [])
    counter: int = 0

    def load(self, predictions: list[Object]) -> None:
        self.predictions = predictions
        self.sample_predictions = []
        self.counter: int = 0

    def update(self, predictions: list[Object]) -> None:
        self.predictions += predictions
        self.sample_predictions = predictions
        self.counter += 1

    def restart(self) -> None:
        self.predictions = []
        self.sample_predictions = []
        self.counter = 0

    def __len__(self) -> int:
        return len(self.predictions)
    
    def __getitem__(self, i: int) -> Object:
        return self.predictions[i]
    
    def __iter__(self) -> Iterable[Object]:
        return (pred for pred in self.predictions)


class Tracker(ABC):

    batch_size : int

    def __init__(
        self, 
        load_path: Optional[str | Path] = None,
        save_path: Optional[str | Path] = None,
    ) -> None:
        
        self.results = TrackingResults()
        self.load_path = load_path
        self.save_path = save_path

        # Load predictions if load_path is not None
        self.load_predictions()

    @property
    def DEVICE(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


    @abstractmethod
    def video_info_post_init(self, video_info: sv.VideoInfo) -> "Tracker":
        pass

    @abstractmethod
    def object(self) -> Type[Object]:
        pass

    @abstractmethod
    def draw_kwargs(self) -> dict:
        pass
    

    @abstractmethod
    def restart(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.results)

    @abstractmethod
    def __str__(self) -> str:
        pass

    def save_predictions(self) -> None:
        if self.save_path:

            print(f"{self.__str__()}: saving predictions")
            
            parsable_predictions = [
                object_cls.serialize()
                for object_cls in self.results.predictions
            ]
            
            with open(self.save_path, "w") as f:
                json.dump(parsable_predictions, f)
            
            print(f"{self.__str__()}: {self.__len__()} predictions saved")
    
    def load_predictions(self) -> None:
        if self.load_path and Path(self.load_path).exists():

            print(f"{self.__str__()}: loading predictions")

            with open(self.load_path, "r") as f:
                parsable_detections = json.load(f)
            
            predictions = [
                self.object().from_json(obj_json)
                for obj_json in parsable_detections
            ]

            self.results.load(predictions)
        
        print(f"{self.__str__()}: {self.__len__()} predictions loaded")

    def to(self, device: Literal["cuda", "cpu", "mps"]) -> None:
        pass

    @abstractmethod
    def predict_sample(self, sample: Iterable[np.ndarray], **kwargs) -> Optional[list[Object]]:
        pass

    @abstractmethod
    def predict_frames(self, frame_generator: Iterable[np.ndarray], **kwargs) -> Optional[list[Object]]:
        pass

    def predict_and_update(self, frame_generator: Iterable[np.ndarray], **kwargs) -> list[Object]:

        def sampler(
            generator: Iterable[np.ndarray],
            sequence_length: int,
        ) -> Iterable[list[np.ndarray]]:
            w = []
            for x in generator:
                w.append(x)

                if len(w) == sequence_length:
                    yield w
                    w = []

            if w != []:
                yield w
        
        try:
            predictions = self.predict_frames(frame_generator, **kwargs)
            self.results.predictions = predictions
        except NoPredictFrames:
            for sample in tqdm(
                sampler(
                    frame_generator,
                    sequence_length=self.batch_size,
                )
            ):
                predictions = self.predict_sample(sample, **kwargs)
                self.results.update(predictions)

        print(f"{self.__str__()}: {len(self.results)}")
            
        return self.results






    


        



