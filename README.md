## AIPL
AIPL (Artificial Intelligence Programming Language) Interpreter is a Python-based parser for a domain-specific language designed to simplify the definition of deep learning model architectures and training parameters. By leveraging popular deep learning frameworks like TensorFlow, AIPL streamlines AI model development and deployment.

# Features
Concise syntax for defining model architectures and training parameters
Support for various layer types, loss functions, and optimizers
Extensibility to accommodate additional layers and components
Integration with popular deep learning frameworks like TensorFlow

# Installation
To use AIPL Interpreter, you'll need to install TensorFlow:

pip install tensorflow

Next, clone the AIPL Interpreter repository:

git clone https://github.com/fingin/AIPL.git
cd AIPL

# Usage
Create an AIPL file with your model architecture and training parameters. For example, example.aipl:


ARCHITECTURE SimpleNN
  LAYER Input 784
  LAYER Dense 128 activation=ReLU
  LAYER Dense 10 activation=Softmax
 
Run the AIPL Interpreter with your AIPL file:

python aipl_parser.py example.aipl

This command will parse the AIPL file, build the TensorFlow model, and display the model summary.

# Extending AIPL Interpreter
To support additional layer types, loss functions, and optimizers, you can modify the aipl_parser.py script. Add new parsing functions and update the _parse_line and _parse_layer methods accordingly.

# Contributing
Contributions to AIPL Interpreter are welcome. To contribute, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Commit your changes
4. Push your changes to the branch
5. Create a Pull Request targeting the main branch

# License
AIPL Interpreter is released under the MIT License. See the LICENSE file for details.
