import bentoml
from bentoml.io import NumpyNdarray
import numpy as np

# Load the model with BentoML
model_ref = bentoml.keras.get('cifar10_ann_model:latest')
model_runner = model_ref.to_runner()

# Define a BentoML service
svc = bentoml.Service('cifar10_ann_service', runners=[model_runner])

# Define the inference API
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_data: np.ndarray) -> np.ndarray:
    return model_runner.run(input_data)
