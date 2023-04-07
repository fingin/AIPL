import re
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class AIPLParser:
    def __init__(self, code):
        self.code = code
        self.models = {}
        self.current_model = None
        self.training_params = {}

    def parse(self):
        for line in self.code.split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            self._parse_line(line)

    def _parse_layer(self, line):
        layer_type, *params = line.split(' ')
        layer_params = {}
        for param in params:
            key, value = param.split('=')
            layer_params[key] = value

        if layer_type == 'Input':
            self.current_model.add(InputLayer(input_shape=(int(layer_params['nodes']),)))
        elif layer_type == 'Dense':
            self.current_model.add(Dense(int(layer_params['nodes']), activation=layer_params['activation']))
        # Add other layers as needed

    def _parse_line(self, line):
        if line.startswith('ARCHITECTURE'):
            _, model_name = line.split(' ')
            self.current_model = Sequential(name=model_name)
            self.models[model_name] = self.current_model
        elif re.match(r'LAYER .*', line):
            self._parse_layer(line)
        # Add other keywords and parsing logic as needed

def interpret_aipl_code(code):
    parser = AIPLParser(code)
    parser.parse()
    return parser.models

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python aipl_parser.py <aipl_file>")
        exit(1)

    with open(sys.argv[1], 'r') as f:
        aipl_code = f.read()

    models = interpret_aipl_code(aipl_code)
    for model_name, model in models.items():
        print(f"Model: {model_name}")
        print(model.summary())
