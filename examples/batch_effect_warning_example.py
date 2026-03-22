"""
This example demonstrates the scenario where a model is trained
on data with multiple batch effects and then transferred to data
with fewer batch effects.

NOTE:
This is a simplified illustration. The actual warning is triggered
inside the transfer pipeline when using properly structured NormData.
"""
from pcntoolkit.normative_model import NormativeModel
import numpy as np

print("Running batch effect warning example")

# Fake minimal data
X_train = np.random.rand(10, 1)
batch_train = np.array(["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"])

X_transfer = np.random.rand(5, 1)
batch_transfer = np.array(["A", "A", "A", "A", "A"])  # fewer batches

model = NormativeModel(template_regression_model="linear")

# Manually inject batch effects (hack for example)
model.batch_effects = batch_train

# Simulate transfer scenario
model.transfer_batch_effects = batch_transfer

print("If batch effects differ, warning should trigger here")