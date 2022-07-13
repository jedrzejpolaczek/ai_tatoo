from typing import Callable
from keras.applications import inception_v3
from keras import backend as K

def load_model() -> Callable:
    # Setting model to not train it. Turning off all training operations.
    K.set_learning_phase(0)

    model = inception_v3.InceptionV3(
        weights='imagenet',
        include_top=False
    )

    return model
