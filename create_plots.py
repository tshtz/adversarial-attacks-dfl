import mlflow
import os 
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
sns.set_theme(rc={'figure.figsize':(6,4)})


def create_boxplot(df_dict, filename, metric="rel_regrets"): 

    sns.boxplot(data=df_dict, x="Method", y=metric, hue="Attack", fliersize=3, showfliers=True, whis=(0,1), showmeans=True, meanline=True)
    plt.savefig(filename + metric + ".png")


# Create the mlflow client
#client = mlflow.MlflowClient(tracking_uri=os.path.abspath(os.path.join(os.getcwd(), "ModelCreators/mlruns"))) ###changed
client = mlflow.MlflowClient()  

# Experiments you want to visualise 
model_experiment_ids = ["685193085963208866"]

# Create a temp dir
temp_dir = "temp/"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

for model_experiment_id in model_experiment_ids:

    list_of_run_ids = ["4035fbf47730436c8854dced79fa7d5f", "a29a484b9ddb444499004d1096d5b588"]

    dfs = []
    for run_id in list_of_run_ids:
        
        # Get model_name
        model_run = client.get_run(run_id)
        model_name = model_run.data.params["attacked_models_name"]

        attacker = model_run.data.params["attacker"]

        # Load artifacts 
        client.download_artifacts(run_id, path="error_metrics.csv", dst_path=temp_dir)
        df = pd.read_csv("temp/error_metrics.csv")

        # Get model name and attack type
        df["Methods"] = model_name
        df["Attack"] = attacker

        # Get exp, epsilon, dataset and seed for filename 
        experiment = model_run.data.params["attacked_models_experiment"]
        eps = model_run.data.params["epsilon"]
        dataset = model_run.data.params["dataset"]
        seed = model_run.data.params["seed"]
        filename = f"{experiment}_{dataset}_{eps}_{seed}" 
        
        # Create a list of all dfs 
        dfs.append(df)

    combined_df = pd.concat(dfs)

    # Plot relative regret
    create_boxplot(combined_df, filename, metric="rel_regrets")
    # Plot absolute regret
    create_boxplot(combined_df, filename, metric="abs_regrets")
    # Plot accuracy error 
    create_boxplot(combined_df, filename, metric="acc_errors")
    



    


        


