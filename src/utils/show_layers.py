from torchinfo import summary
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.deeplearning.cnn import NeuralNetwork

model = NeuralNetwork(num_classes=2)

summary(model, (1, 8))
