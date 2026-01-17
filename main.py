from generator import *
from mano import ManoSimulator
from solver import *
from substrate import SN
import matplotlib.pyplot as plt
from termcolor import colored
import simpy
import pickle
import networkx as nx
import copy
import json
from controller import Global_controller as Controller
import time

# to measure exec time 
from timeit import default_timer as timer 

# Opening JSON file 
with open('parameters.json', 'r') as openfile: 
    # Reading from json file 
    json_object = json.load(openfile)  
    
OUTPUT_PATH=json_object['OUTPUT_PATH']
SIM_TIME=json_object['SIM_TIME']
REPEAT_EXPERIENCE=json_object['REPEAT_EXPERIENCE']

# Substrate related parameters
beta=json_object['beta']
cpu_range = json_object['cpu_range']
numnodes = json_object['numnodes']
bw_range = json_object['bw_range']
lt_range = json_object['lt_range']
# Virtual network related parameters
num_reqs = json_object['num_reqs'] # number of simultaneous requests for the vnr generator
vnfs_range =json_object['vnfs_range']
vcpu_range =json_object['vcpu_range']
vbw_range =json_object['vbw_range']
vlt_range = json_object['vlt_range']
start_mean_calculation=json_object['start_mean_calculation']
MTBA = json_object['MTBA']     # Mean Time Between Arrival
MLT=json_object['MLT']         # Mean life time of each cass of VNR
MTBS= json_object['MTBS']      # Mean time between scale demande of each class of VNR
vnr_classes=json_object['vnr_classes'] #Proportion of vnr classes 
p_flavors=json_object['p_flavors']
flavor_tab=json_object['flavors']
solvers_inputs=json_object["solvers"]
episode_duration=json_object["episode_duration"]
episode_per_file=json_object["episode_per_file"]
max_vnfs=vnfs_range[1]-1
results_location=json_object["OUTPUT_PATH"]
np.random.seed(seed=100)
Seeds = [317805,7671309,222111,310320]#np.random.randint(42947296, size=REPEAT_EXPERIENCE)




# Create a substrate environment
topology  = 'generated_network.matrix'
topology =  nx.Graph(np.loadtxt(topology, dtype=int))
old_subNet= SN(numnodes, cpu_range, bw_range,lt_range,topology)

old_subNet.drawSN(edege_label=True)
old_subNet.msg()



with open('topology.pkl', 'wb') as output:
    pickle.dump(old_subNet, output, pickle.HIGHEST_PROTOCOL)
    
print(colored('Experience started', 'green'))
for j in range(len(MTBA)):
    print('%d' % j)
    np.random.seed(seed=np.random.randint(Seeds[j], size=REPEAT_EXPERIENCE)) #Seeds[i]
    
    env = simpy.Environment()
    solvers=[]
    solvers_names=[]
    sns=[]
    for i in range(len(solvers_inputs)):
        sns.append(dc(old_subNet))
        solvers_names.append(solvers_inputs[i]["name"])
        if solvers_inputs[i]["type"]=="FF":
            solvers.append(FirstFit(solvers_inputs[i]["sigma"],solvers_inputs[i]["rejection_penalty"]))
        if solvers_inputs[i]["type"]=="GNNDRL":
            solvers.append(GNNDRL(solvers_inputs[i]['sigma'],solvers_inputs[i]["gamma"],solvers_inputs[i]["rejection_penalty"],  solvers_inputs[i]["learning_rate"], solvers_inputs[i]["epsilon"], solvers_inputs[i]["memory_size"], solvers_inputs[i]["batch_size"], solvers_inputs[i]["num_inputs_sn"], solvers_inputs[i]["num_inputs_vnr"], solvers_inputs[i]["hidden_size"], solvers_inputs[i]["GCN_out"], solvers_inputs[i]["num_actions"],solvers_inputs[i]["max_itteration"],solvers_inputs[i]["eps_min"] , solvers_inputs[i]["eps_dec"]  ))
        if solvers_inputs[i]["type"]=="GNNDRL2":
            solvers.append(GNNDRL2(solvers_inputs[i]['sigma'],solvers_inputs[i]["gamma"],solvers_inputs[i]["rejection_penalty"],  solvers_inputs[i]["learning_rate"], solvers_inputs[i]["epsilon"], solvers_inputs[i]["memory_size"], solvers_inputs[i]["batch_size"], solvers_inputs[i]["num_inputs_sn"], solvers_inputs[i]["num_inputs_vnr"], solvers_inputs[i]["hidden_size"], solvers_inputs[i]["GCN_out"], solvers_inputs[i]["num_actions"],solvers_inputs[i]["max_itteration"],solvers_inputs[i]["eps_min"] , solvers_inputs[i]["eps_dec"] ))
        if solvers_inputs[i]["type"]=="GNNDRLPPO":
            solvers.append(GNNDRLPPO(
                solvers_inputs[i]['sigma'],
                solvers_inputs[i]["gamma"],
                solvers_inputs[i]["rejection_penalty"],  
                solvers_inputs[i]["learning_rate"], 
                solvers_inputs[i]["epsilon"], 
                solvers_inputs[i]["memory_size"], 
                solvers_inputs[i]["batch_size"], 
                solvers_inputs[i]["num_inputs_sn"], 
                solvers_inputs[i]["num_inputs_vnr"], 
                solvers_inputs[i]["hidden_size"], 
                solvers_inputs[i]["GCN_out"], 
                solvers_inputs[i]["num_actions"],
                solvers_inputs[i]["max_itteration"],
                solvers_inputs[i]["eps_min"], 
                solvers_inputs[i]["eps_dec"],
                solvers_inputs[i].get("clip_ratio", 0.2),
                solvers_inputs[i].get("ppo_epochs", 4),
                solvers_inputs[i].get("entropy_coef", 0.01)
            ))
        
    controller=Controller(solvers_names,sns,env,episode_duration,results_location,episode_per_file,max_vnfs)
    global_solver=GlobalSolver(solvers)
    manoSimulator=ManoSimulator(global_solver,solvers_names,sns,env,controller)
    start=time.time()
    generator=Generator(vnr_classes, MLT, MTBS, MTBA[j], vnfs_range, vcpu_range, vbw_range,vlt_range, flavor_tab, p_flavors,len(solvers_inputs))  
    env.process(generator.VnrGenerator_poisson(env,manoSimulator))
    env.process(controller.simulation_controller())
    # Execute!
    env.run(until=SIM_TIME[j])
    print(time.time()-start)
    
    
    

