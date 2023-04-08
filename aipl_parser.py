import os
import sys
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflowjs as tfjs
import tensorflow.lite as tflite
import pandas as pd
import tensorflow as tf

class AIPLParser:
    def __init__(self, code):
        self.code = code
        self.models = {}
        self.training_params = {}
        self.current_model = None

    def parse(self):
        for line in self.code.split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            self._parse_line(line)

    def _parse_line(self, line):
        if line.startswith('ARCHITECTURE'):
            _, model_name = line.split(' ')
            self.current_model = Sequential(name=model_name)
            self.models[model_name] = self.current_model
        elif re.match(r'LAYER .*', line):
            self._parse_layer(line)
        elif line.startswith('LOSS_FUNCTION'):
            _, loss_function = line.split(' ')
            self.training_params[self.current_model.name]['loss_function'] = self._parse_loss_function(loss_function)
        elif line.startswith('DATASET'):
            train_dataset, val_dataset = self._parse_dataset(line)
            self.training_params[self.current_model.name]['train_dataset'] = train_dataset
            self.training_params[self.current_model.name]['val_dataset'] = val_dataset
        # ... the rest of the _parse_line method
        else:
            raise ValueError(f"Unsupported line: {line}")

    def _parse_layer(self, line):
        _, layer_type, *layer_args = line.split(' ')
        layer = self._create_layer(layer_type, layer_args)
        self.current_model.add(layer)

    def _create_layer(self, layer_type, layer_args):
        if layer_type == 'Input':
            return InputLayer(int(layer_args[0]))
        elif layer_type == 'Dense':
            units = int(layer_args[0])
            activation = self._get_kwarg(layer_args, 'activation', default='linear')
            kernel_initializer = self._get_kwarg(layer_args, 'kernel_initializer', default='glorot_uniform')
            kernel_regularizer = self._get_kwarg(layer_args, 'kernel_regularizer')
            return Dense(units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        elif layer_type == 'Conv2D':
            filters = int(layer_args[0])
            kernel_size = tuple(map(int, layer_args[1].split(',')))
            activation = self._get_kwarg(layer_args, 'activation', default='linear')
            padding = self._get_kwarg(layer_args, 'padding', default='valid')
            strides = tuple(map(int, self._get_kwarg(layer_args, 'strides', default='1,1').split(',')))
            return Conv2D(filters, kernel_size, activation=activation, padding=padding, strides=strides)
        elif layer_type == 'LSTM':
            units = int(layer_args[0])
            activation = self._get_kwarg(layer_args, 'activation', default='tanh')
            return LSTM(units, activation=activation)
        elif layer_type == 'Embedding':
            input_dim = int(layer_args[0])
            output_dim = int(layer_args[1])
            input_length = int(self._get_kwarg(layer_args, 'input_length', default=None))
            return Embedding(input_dim, output_dim, input_length=input_length)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    
    def train_models(self):
        (x_train, y_train), (x_val, y_val) = self._load_data()
        
        for model_name, model in self.models.items():
            if model_name not in self.training_params:
                print(f"Skipping training for {model_name}: Training parameters not found.")
                continue
            
            params = self.training_params[model_name]
            model.compile(
                optimizer=params['optimizer'],
                loss=params['loss_function'],
                metrics=['accuracy']
            )

            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size']
            )

    def _load_data(self):
        # You can add support for custom datasets here. In this example, we use the MNIST dataset
        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        x_train = x_train.reshape((-1, 784)) / 255.0
        x_val = x_val.reshape((-1, 784)) / 255.0
        y_train = to_categorical(y_train, 10)
        y_val = to_categorical(y_val, 10)
        return (x_train, y_train), (x_val, y_val)
    
    def save_model(self, model_name, file_path, save_format='h5'):
        model = self.models[model_name]
        model.save(file_path, save_format=save_format)
        print(f"Model '{model_name}' saved to {file_path} in {save_format} format")

    def load_model(self, file_path, custom_objects=None):
        loaded_model = load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")
        return loaded_model

    def export_model(self, model_name, file_path, export_format='tfjs'):
        model = self.models[model_name]
        
        if export_format == 'tfjs':
            tfjs.converters.save_keras_model(model, file_path)
            print(f"Model '{model_name}' exported as TensorFlow.js at {file_path}")
        elif export_format == 'tflite':
            converter = tflite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(file_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Model '{model_name}' exported as TensorFlow Lite at {file_path}")
        # Add other export formats here (e.g., ONNX)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
    def _load_csv_dataset(self, file_path, target_column, batch_size=32, validation_split=0.2, shuffle=True, seed=None, *args):
        df = pd.read_csv(file_path)
        target = df.pop(target_column)
        dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df), seed=seed)

        train_size = int((1 - validation_split) * len(df))
        train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset

    def _load_json_dataset(self, file_path, target_column, batch_size=32, validation_split=0.2, shuffle=True, seed=None, *args):
        df = pd.read_json(file_path)
        target = df.pop(target_column)
        dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df), seed=seed)

        train_size = int((1 - validation_split) * len(df))
        train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset

    def _load_image_dataset(self, directory, batch_size=32, validation_split=0.2, seed=None, *args):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset='training',
            seed=seed,
            batch_size=batch_size
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset='validation',
            seed=seed,
            batch_size=batch_size
        )
        
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

        return train_ds, val_ds
    
    def _parse_dataset(self, line):
        tokens = line.split(' ')
        dataset_type = tokens[1]
    
        # Parse the remaining tokens as a dictionary of keyword arguments
        kwargs = {}
        for i in range(2, len(tokens), 2):
            key, value = tokens[i], tokens[i + 1]
            # Convert numerical values to integers or floats
            if value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
        
            kwargs[key] = value
    
        if dataset_type == 'csv':
            return self._load_csv_dataset(**kwargs)
        elif dataset_type == 'json':
            return self._load_json_dataset(**kwargs)
        elif dataset_type == 'image':
            return self._load_image_dataset(**kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


def interpret_aipl_code(code):
    parser = AIPLParser(code)
    parser.parse()
    return parser

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python aipl_parser.py <aipl_file>")
        exit(1)

    with open(sys.argv[1], 'r') as f:
        aipl_code = f.read()

    parser = interpret_aipl_code(aipl_code)
    for model_name, model in parser.models.items():
        print(f"Model: {model_name}")
        print(model.summary())

    parser.train_models()