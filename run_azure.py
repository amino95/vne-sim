import os
import sys

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import AmlCompute, Environment, CommandJob
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
except ImportError:
    sys.exit(
        "Erreur: Les modules Azure ne sont pas installés.\n"
        "Exécutez la commande suivante : pip install azure-ai-ml azure-identity"
    )


def main():
    # ---------------------------------------------------------
    # Configuration - A REMPLIR AVEC VOS INFOS AZURE
    # ---------------------------------------------------------
    # Vous pouvez trouver ces infos dans la vue d'ensemble de votre Workspace sur le portail Azure
    SUBSCRIPTION_ID = "963ef9cb-ad6c-45b0-8c5e-d8197659f3a5"
    TENANT_ID = "6ff8a032-8fe0-4533-a046-8c0104312bbd"

    RESOURCE_GROUP = "Simulator"
    WORKSPACE_NAME = "vne-sim"

    # Nom du cluster de calcul
    COMPUTE_NAME = "gpu-cluster"
    # Taille de la VM (Standard_DS3_v2 est bien pour du CPU général)
    # Si vous voulez utiliser le GPU pour le DRL, utilisez une série NC (ex: Standard_NC6)
    VM_SIZE = "Standard_NC4as_T4_v3"

    # ---------------------------------------------------------
    # Connexion à Azure ML
    # ---------------------------------------------------------
    print("Connexion à Azure ML...")
    # Note: Assurez-vous d'avoir installé Azure CLI et exécuté 'az login'
    try:
        credential = InteractiveBrowserCredential(tenant_id=TENANT_ID)
        ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
        print(f"Connecté au workspace : {WORKSPACE_NAME}")
    except Exception as e:
        print(f"Erreur de connexion: {e}")
        return

    # ---------------------------------------------------------
    # 1. Création ou Récupération du Cluster de Calcul
    # ---------------------------------------------------------
    try:
        compute_target = ml_client.compute.get(COMPUTE_NAME)
        print(f"Le cluster '{COMPUTE_NAME}' existe déjà.")
    except Exception:
        print(f"Création du cluster '{COMPUTE_NAME}'...")
        compute_target = AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size=VM_SIZE,
            min_instances=0,
            max_instances=4,  # Nombre max de nœuds
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(compute_target).result()

    # ---------------------------------------------------------
    # 2. Définition de l'Environnement
    # ---------------------------------------------------------
    # Fichier temporaire pour la définition Conda
    conda_file = "conda_env_azure.yaml"

    # Définition des dépendances basée sur votre projet
    conda_yaml = """
        name: vne-sim-env
        channels:
        - nvidia
        - pytorch
        - conda-forge
        dependencies:
        - python=3.8
        - numpy
        - networkx
        - matplotlib
        - pytorch
        - pytorch-cuda=11.8
        - pip
        - pip:
            - simpy
            - termcolor
            - dgl
            - azure-ai-ml
        """
    with open(conda_file, "w") as f:
        f.write(conda_yaml)

    print("Création de l'environnement...")
    env_name = "vne-sim-env"
    job_env = Environment(
        name=env_name,
        description="Environnement pour VNE Simulator",
        conda_file=conda_file,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    ml_client.environments.create_or_update(job_env)

    # ---------------------------------------------------------
    # 3. Soumission du Job
    # ---------------------------------------------------------
    print("Configuration du job...")
    job = CommandJob(
        code="./",  # Télécharge le répertoire courant (votre code) vers Azure
        command="python main.py",
        environment=f"{env_name}@latest",
        compute=COMPUTE_NAME,
        display_name="vne-sim-execution",
        experiment_name="vne-sim-experiment",
    )

    print("Soumission du job...")
    returned_job = ml_client.jobs.create_or_update(job)
    print("Job soumis avec succès !")
    print(f"Suivez l'avancement ici : {returned_job.studio_url}")

    # Nettoyage du fichier temporaire
    if os.path.exists(conda_file):
        os.remove(conda_file)


if __name__ == "__main__":
    main()
