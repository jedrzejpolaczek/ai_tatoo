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

def configure_pretrained_model() -> dict:
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.0,
        'mixed4': 2.0,
        'mixed5': 1.5,
    }

    return layer_contributions

def max_loss_value(model, layer_contributions) -> None:
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    loss = K.variable(0)

    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output

        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
