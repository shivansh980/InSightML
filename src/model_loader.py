from abc import ABC, abstractmethod
import pickle
from tensorflow import keras
import torch

# Abstract base class
class ModelLoader(ABC):
    @abstractmethod
    def load_model(self, file_path):
        pass

# Concrete classes for loading models
class PickleModelLoader(ModelLoader):
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

class KerasModelLoader(ModelLoader):
    def load_model(self, file_path):
        return keras.models.load_model(file_path)

class TorchModelLoader(ModelLoader):
    def load_model(self, file_path):
        model = torch.load(file_path)
        model.eval()
        return model

# Factory Class for model selection
class ModelLoaderFactory:
    @staticmethod
    def get_model_loader(file_path):
        if file_path.endswith('.pkl'):
            return PickleModelLoader()
        elif file_path.endswith('.h5') or file_path.endswith('.keras'):
            return KerasModelLoader()
        elif file_path.endswith('.pt') or file_path.endswith('.pth'):
            return TorchModelLoader()
        else:
            raise ValueError("Unsupported model format")
