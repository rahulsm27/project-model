# E2E ML project using MLflow, Pytorch Lightning, GCP cloud, Docker

Project on Twitter Tweets sentiment analysis
The project consists of three main parts

1. Generate config file using src/generate_final_config.py. The config file contains parameters for the entire project and experiment tracking. The file is saved at
src/configs/automatically_generated/config.yaml
2. Launch Job on GCP - It creates an instance group on GCP based on the parameters specific in the config file
3. Run task - The code is containerized and run on the GCP instance created in step 2. It uses pytorch lightning for distributed training. The parameters are logged in the MLflow tracking server.

Note: The mlflow tracking server is created as part of another repo and the same is utilised here

The project creates an instance group on the GCP cloud.
The code has been written to read the data from GCP storage
The code uses Pytorch lightning for distributed training
It uses MLFlow for experiment tracking
The config file is stored under the configs->automatically_generated folder

# Key Open source tools used

1. Hydra for config management -> It helps keep track of our hyperparameters.
2. GCP cloud infrastructure to develop instance groups for distributed training.
3. Pytorch Lightning -> For distributed training on the infrastructure created on GCP
4. MLflow for experiment tracking and logging, docker for containerization of code
5. Makefile for automating repeated commands
6. Poetry for building dependencies
7. Docker for containerization