from azureml.core import ScriptRunConfig, Workspace, Experiment, Environment

azorel_workspace = Workspace.from_config()

experiment = Experiment(azorel_workspace, "first-try-bscdegree")
env = Environment.from_conda_specification('azure_env', r'.azureml/azure-env.yaml')

config = ScriptRunConfig(source_directory= './src', script='train.py', compute_target='gpu-cluster')
config.run_config.environment = env

run = experiment.submit(config)

print(run.get_portal_url())