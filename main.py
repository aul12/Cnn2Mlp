from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import xml.etree.ElementTree as et
from tensorflow.python import pywrap_tensorflow

def conv2dense(kernel, bias):
    #TODO create weight and bias matrices and flatten them
    #TODO the input sizes need to be asked from the user probably
    return kernel, bias

def get_order(layers_map):
    #TODO request order via input()

def map_layers(matrixes):
    num_of_layers = int(len(matrixes) / 2)
    print("Assuming that the MLP consists of %d layers" % num_of_layers)
    layer_names = set()
    for name,_ in matrixes.items():
        layer_name = name.split("/", 1)[0]
        layer_names.add(layer_name)
    print(layer_names)

    layers_map = dict()
    for layer_name in layer_names:
        kernel = matrixes[layer_name+"/kernel"]
        bias = matrixes[layer_name+"/bias"]
        if "conv" in layer_name:
            kernel, bias = conv2dense(kernel, bias)
        layers_map[layer_name] = (kernel, bias)

    return get_order(layers_map)

def save_to_opencv(layers, output_file_name):
    layer_sizes_text = ""
    input_scale_text = ""
    output_scale_text = ""
    weights_text = list()

    for layer in layers:
        weight, bias = layer
        weight, _ = weight
        bias, _ = bias

        layer_sizes_text += str(weight.shape[0]) + " "

        w = np.transpose(np.array(weight))
        b = np.transpose(np.array([bias]))
        homogeneous = np.concatenate((w, b), axis=1)
        flattened = homogeneous.flatten('C')
        weight_text = ""
        for c in range(0, len(flattened)):
            weight_text += str(flattened[c])
            weight_text += "\n" if c % 2 != 0 else " "
        weights_text.append(weight_text)

    output_size = layers[len(layers) - 1][0][0].shape[1]
    input_size = layers[0][0][0].shape[0]

    layer_sizes_text += str(output_size)

    for _ in range(0, input_size):
        input_scale_text += "1. 0. "

    for _ in range(0, output_size):
        output_scale_text += "1. 0. "

    root = et.Element("opencv_storage")
    mlp = et.SubElement(root, "opencv_ml_ann_mlp")
    format = et.SubElement(mlp, "format")
    layer_sizes = et.SubElement(mlp, "layer_sizes")
    activation_function = et.SubElement(mlp, "activation_function")
    f_param1 = et.SubElement(mlp, "f_param1")
    f_param2 = et.SubElement(mlp, "f_param2")
    min_val = et.SubElement(mlp, "min_val")
    max_val = et.SubElement(mlp, "max_val")
    min_val1 = et.SubElement(mlp, "min_val1")
    max_val1 = et.SubElement(mlp, "max_val1")
    training_params = et.SubElement(mlp, "training_params")
    train_method = et.SubElement(training_params, "train_method")
    dw_scale = et.SubElement(training_params, "dw_scale")
    moment_scale = et.SubElement(training_params, "moment_scale")
    term_criteria = et.SubElement(training_params, "term_criteria")
    epsilon = et.SubElement(term_criteria, "epsilon")
    iterations = et.SubElement(term_criteria, "iterations")
    input_scale = et.SubElement(mlp, "input_scale")
    output_scale = et.SubElement(mlp, "output_scale")
    inv_output_scale = et.SubElement(mlp, "inv_output_scale")
    weights = et.SubElement(mlp, "weights")
    layer_weights = list()
    for _ in layers:
        layer_weights.append(et.SubElement(weights, "_"))

    format.text = "3"  # OpenCv Version 2 or 3, 4 is treated as 3
    layer_sizes.text = layer_sizes_text
    activation_function.text = "SIGMOID_SYM"  # Sigmoid, which is actually tanh
    f_param1.text = "1"  # Use a normalied opencv-sigmoid which is tanh(x/2)
    f_param2.text = "1"
    min_val.text = "0."
    max_val.text = "0."
    min_val1.text = "0."
    max_val1.text = "0."
    train_method.text = "BACKPROP"
    dw_scale.text = "1.0e-03"
    moment_scale.text = "0."
    epsilon.text = "1.0e-02"
    iterations.text = "1000"
    input_scale.text = input_scale_text
    output_scale.text = output_scale_text
    inv_output_scale.text = output_scale_text
    for c in range(0, len(layers)):
        layer_weights[c].text = weights_text[c]

    file = open(output_file_name, "wb")
    et.ElementTree(root).write(file)


def main():
    if len(sys.argv) != 3:
        print("usage: main.py model.ckpt output.xml")
        exit(1)

    checkpoint_file = sys.argv[1]
    output_file = sys.argv[2]

    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)

    var_to_shape_map = reader.get_variable_to_shape_map()
    weights = dict()
    for key in sorted(var_to_shape_map):
        weights[key] = reader.get_tensor(key)
        print("%s (%s)" % (key, str(reader.get_tensor(key).shape)))
    layers = map_layers(weights)
    #save_to_opencv(layers, output_file)


if __name__ == "__main__":
    main()
