# coir



# simple use
'''
import coir
from coir.models import YourCustomDEModel

model_name = "intfloat/e5-base-v2"

# Load the model
model = YourCustomDEModel(model_name=model_name)

# Get tasks
tasks = coir.get_tasks(tasks=["stackoverflow-qa"])

# Initialize evaluation
evaluation = coir.COIR(tasks=tasks)

# Run evaluation
results = evaluation.run(model, output_folder=f"results/{model_name}")
print(results)

'''
