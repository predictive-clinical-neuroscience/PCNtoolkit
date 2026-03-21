"""
Example script demonstrating the warning when transfer data
contains fewer batch effects than training data.
"""
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.dataio import NormData

# Example demonstrating missing batch effect warning

# --- Step 1: Training data with MORE batch effects ---
# (e.g., data collected from multiple sites)
train_data = NormData("train_dataset_with_multiple_batches")

model = NormativeModel()
model.fit(train_data)

# --- Step 2: Transfer data with FEWER batch effects ---
# (e.g., subset of sites)
transfer_data = NormData("transfer_dataset_with_fewer_batches")

# --- Step 3: Transfer (should trigger warning)
model.transfer(transfer_data)