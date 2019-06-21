from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import xml.etree.ElementTree as et
from tensorflow.python import pywrap_tensorflow

size_map = dict(conv2d=(80, 80, 3), conv2d_1=(40, 40, 32), conv2d_2=(20, 20, 64))


# kernel \in R^{width height channels filters}
def conv2dense(kernel, bias, input_size, stride):
    input_size_flat = input_size[0] * input_size[1] * input_size[2]
    output_size_flat = (input_size[0] // stride) * (input_size[1] // stride) * kernel.shape[3]
    weights = np.zeros((input_size_flat, output_size_flat))

    for f in range(0, kernel.shape[2]):
        for chi in range(0, input_size[0], stride):
            for upsilon in range(0, input_size[1], stride):
                kx = (kernel.shape[0] - 1) // 2
                ky = (kernel.shape[1] - 1) // 2
                w = np.zeros((input_size[0] // stride, input_size[1] // stride, kernel.shape[3]))
                for c in range(0, kernel.shape[2]):
                    for x in range(-kx, kx):
                        for y in range(-ky, ky):
                            ix = chi // stride + x
                            iy = upsilon // stride + y
                            if 0 <= ix < w.shape[0] and 0 <= iy < w.shape[1]:
                                w[ix][iy][c] = kernel[x + kx][y + ky][c][f]
                w_flat = np.reshape(w, -1)
                index = f * input_size[0] * input_size[1] + chi * input_size[1] + upsilon
                weights[index] = w_flat

    bias_vect = np.zeros(output_size_flat)
    per_bias_count = (input_size[0] // stride) * (input_size[1] // stride)
    for i in range(0, bias.shape[0]):
        for c in range(0, per_bias_count):
            bias_vect[c + i * per_bias_count] = bias[i]

    return weights, bias_vect


def get_order(layers_map):
    print("Retrieving order...")
    ret = list()
    for key in sorted(layers_map):
        ret.append(layers_map[key])
    print("...finished!")
    return ret


def map_layers(matrixes):
    print("Mapping layers...")
    num_of_layers = int(len(matrixes) / 2)
    print("Assuming that the net consists of %d layers" % num_of_layers)
    layer_names = set()
    for name, _ in matrixes.items():
        layer_name = name.split("/", 1)[0]
        layer_names.add(layer_name)

    layers_map = dict()
    for layer_name in layer_names:
        kernel = matrixes[layer_name + "/kernel"]
        bias = matrixes[layer_name + "/bias"]
        if "conv" in layer_name:
            kernel, bias = conv2dense(kernel, bias, size_map[layer_name], 3)
        layers_map[layer_name] = (kernel, bias)

    print("...finished!")
    return get_order(layers_map)


def save_to_opencv(layers, output_file_name):
    print("Saving to opencv...")
    layer_sizes_text = ""
    input_scale_text = ""
    output_scale_text = ""

    output_size = layers[len(layers) - 1][0].shape[1]
    input_size = layers[0][0].shape[0]

    for c in range(0, len(layers)):
        weight, bias = layers[c]
        layer_sizes_text += str(weight.shape[0]) + " "

    layer_sizes_text += str(output_size)

    for _ in range(0, input_size):
        input_scale_text += "1. 0. "

    for _ in range(0, output_size):
        output_scale_text += "1. 0. "

    print("...building xml...")
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
        layer_weights[c].text = "__SPLIT__"

    print("...splitting...")
    xml_str = et.tostring(root, encoding='utf8', method='xml')
    parts = str(xml_str).split("__SPLIT__")
    for c in range(0, len(parts)):
        file = open("part_%02d.xml" % (c * 2), "w")
        file.write(parts[c])
        file.close()

    print("...writing matrices...")
    for c in range(0, len(layers)):
        weight, bias = layers[c]
        homogeneous = np.concatenate((weight, [bias]), axis=0)
        flattened = homogeneous.flatten('A')

        print("w")
        file = open("part_%02d.xml" % (c * 2 + 1), "w")
        for w in flattened:
            file.write("%d " % w)
        file.close()

    print("...finished!")


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
    layers = map_layers(weights)
    save_to_opencv(layers, output_file)


if __name__ == "__main__":
    main()
