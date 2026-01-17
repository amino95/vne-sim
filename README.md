# VNF Placement and Scalability Simulator

This simulator is designed to perform virtual network function (VNF) placement and scalability studies within the context of network slicing in beyond 5G networks. The simulator leverages GCNs combined with Deep Reinforcement Learning (DRL) methods to handle VNF placement dynamically, accounting for future scalability needs and resource constraints.

## Overview

The simulator tackles the challenges of Virtual Network Embedding (VNE), focusing on VNF placement and scalability in a dynamic network environment. It uses a GCN-DRL (Graph Convolutional Network - Deep Reinforcement Learning) framework to predict optimal VNF placement, maximizing network efficiency while ensuring sufficient resources are available for future scalability.

Key components of the simulator include:

1. *GCN-based state extraction:* Extracts topological information of the substrate network and VNF graphs.
2. *A2C-based DRL agent:* Learns optimal placement strategies, incorporating both immediate resource requirements and future scalability.
3. *Reward metrics:* Guides placement to maximize efficiency, considering resource cost ratios (R2C) and scalability load (Pload) for effective resource distribution.

## Installation

1. Create a virtual environment (recommended) and activate it.
    
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

2. Install dependencies by running:
```bash
pip install -r requirements.txt
```
3. Install DGL (Deep Graph Library):
```bash
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

    #If you encounter any issues with DGL, try uninstalling and reinstalling it. You can refer to DGL's installation guide for the latest version;  https://www.dgl.ai/pages/start.html
```

## Usage

This simulator is configured to accept VNF requests dynamically. VNF requests are modeled as virtual network requests (VNRs), each containing a specific chain of VNFs and link requirements. During each simulation episode, the simulator:

* Receives VNF placement requests.
* Evaluates network resource availability.
* Calculates optimal placement of VNFs.
* Manages resource scaling requests for placed VNFs.

## Running the Simulator

To run the simulator, execute the main simulation script:
```bash
python main.py
```

## Simulator Configuration

The simulation parameters can be configured in the configuration file (parameters.json).
:

* **OUTPUT_PATH:**  Specifies the directory path where simulation outputs are saved. You should always make sure that the folder does not already exist to avoid deleting existing results. The code will generate an error if the folder exists.

* **SIM_TIME**: Defines the total simulation time, in Simpy Timesteps, for each episode.

* **REPEAT_EXPERIENCE**: Number of times to repeat the same simulation experience for consistency in results.

* **MTBA** (Mean Time Between Arrivals): Average time between incoming VNRs

* **MLT** (Mean Life Time): Average life duration of each VNR in the network. It’s defined as a list to allow variation across classes.

* **vnr_classes**: Defines the probability distribution for different VNR classes based on lifetime and scalability needs.

* **MTBS** (Mean Time Between Scalability): Average time between scalability requests for each VNR, represented as a list to accommodate varying scalability levels for each VNR class.

* **flavors**: The flavor_tab is a list that determines the lengths of flavor options for each VNR, with each VNR having its own flavor list based on this tab.

* **p_flavors**: Probability distribution of different flavor sizes, correlating with each entry in flavors.

* **cpu_range**: Defines the minimum and maximum CPU resources available for substrate nodes in the network.

* **numnodes**: Number of substrate nodes in the network topology.

* **bw_range**: Bandwidth range for links in the substrate network, given as minimum and maximum values.

* **vnfs_range**: Specifies the minimum and maximum number of VNFs allowed in each VNR.

* **vcpu_range**: Range of initial CPU requests for each VNF.

* **vbw_range**: Range of initial bandwidth requests for each VNF link.

* **vlt_range**: Latency range between VNFs within each VNR.

* **episode_duration**: Number of time steps per episode.

* **episode_per_file**: Number of episodes to be saved in each output file.

* **solvers**: A list of solvers used in the simulation, each with specific configurations:
    - **name**: Name of the solver configuration.
    - **type**: Type of solver (e.g., GNNDRL or Firstfit).
    - **sigma**: Scaling factor in the reward function that controls the balance between placement and scalability.
    - **gamma**: Discount factor for future rewards in reinforcement learning.
    - **rejection_penalty**: Penalty value for rejected placements.
    - **learning_rate**: Learning rate for the GNNDRL model.
    - **epsilon**: Initial exploration rate for the reinforcement learning algorithm.
    - **memory_size**: Size of memory for storing experiences in reinforcement learning.
    - **batch_size**: Batch size for training in reinforcement learning.
    - **num_inputs_sn**: Number of features for substrate nodes.
    - **num_inputs_vnr**: Number of features for VNFs in each VNR.
    - **hidden_size**: Hidden layer size in the neural network.
    - **GCN_out**: Output size for the Graph Convolutional Network (GCN).
    - **num_actions**: Number of possible actions (or placements) which is the number of nodes in the SN
    - **max_itteration**: Maximum number of iterations allowed per action.
    - **eps_min**: Minimum exploration rate in epsilon-greedy policy.
    - **eps_dec**: Epsilon decay rate per time step.

This section configures the simulator to handle diverse scenarios by varying resource demands, VNF properties, and solver behaviors. These options enable controlled experimentation across various placement and scalability strategies.


