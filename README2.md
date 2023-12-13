all module are imported as src. assuming they are run outside the src folder.
this is achieved using @ in makefile
# Yes, you are right, this is a little bit confusing. This is because of the way Hydra works. We are using their decorator in our decorator to read the config file. And their decorator works with relative path to where it was defined. Because we are using Hydra's decorator under utils module, the relative path to the config's directory is ../configs

USE OF CONTEXT MANAGER
from contextlib import contextmanager

@contextmanager
def example_context_manager():
    # Code to run before entering the 'with' block
    print("Entering the context manager")

    # The 'yield' statement is where the code within the 'with' block will run
    yield

    # Code to run after exiting the 'with' block
    print("Exiting the context manager")

# Using the context manager with the 'with' statement
with example_context_manager():
    print("Inside the 'with' block")

Entering the context manager
Inside the 'with' block
Exiting the context manager


---------------
OPTIONAL AND TYPING

This line imports the Optional and Any types from the typing module, allowing you to use them in your code. Here's a brief explanation of these types:

Optional: This is used to indicate that a variable or function argument can be either of a specified type or None. It is often used when a value is optional.

Any: This is a special type that represents any value. It is used when the type of a variable or function argument can be of any type.



sudo docker run --init --rm --gpus all --ipc host --user root --hostname Rahul --privileged --log-driver=gcplogs -e PYTHONHASHSEED=1234 -e MLFLOW_TRACKING_URI=http://project-mlflow.europe-west4-a.c.mlendtoend.internal:6100 -e TOKENIZERS_PARALLELISM=false europe-west4-docker.pkg.dev/mlendtoend/mlflow/project-model:train-B013AFA0-DB21-41CF-8F1A-F7FBC929DFF9 torchrun --nnodes=1 --nproc_per_node='gpu' src/run_tasks.py

docker run --rm -it --gpus all europe-west4-docker.pkg.dev/mlendtoend/mlflow/project-model:train-B013AFA0-DB21-41CF-8F1A-F7FBC929DFF9 bash

import torch
torch.__version__
torch.cuda.is_available()