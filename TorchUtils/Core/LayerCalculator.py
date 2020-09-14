def calculate_listed_layer(layers, x):
    for layer in layers:
        x = layer(x)

    return x
