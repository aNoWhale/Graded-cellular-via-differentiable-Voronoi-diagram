import os
import random
import time
from typing import Callable, Union, List

from deap import base, creator, tools, algorithms
import numpy as np
import jax.numpy as jnp
from softVoronoi import generate_gene_random
from linear_fem import linear_fem
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.softVoronoi import generate_voronoi

##### support for numpy
def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwriting in the swap operation. It prevents
    """
    size = len(ind1) if len(ind1.shape)==1 else ind1.shape[1]
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    if len(ind1.shape)!=1:
        ind1[:,cxpoint1:cxpoint2], ind2[:,cxpoint1:cxpoint2] \
            = ind2[:,cxpoint1:cxpoint2].copy(), ind1[:,cxpoint1:cxpoint2].copy()
    else:
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


def GA(optimizationParams, generate_p_random:Callable, objective, construction, load:np.array, location_fns: List, fixed_location: Callable):
    # define problem and individual
    Nx,Ny=optimizationParams["Nx"],optimizationParams["Ny"]
    Lx,Ly=optimizationParams["Lx"],optimizationParams["Ly"]
    # creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))#多目标优化
    # creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))#多目标优化
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    # low=np.concatenate((np.repeat(np.array([0-margin,0-margin]),sites_num),
    #                     np.repeat(np.array(([1,0],[0,1])).flatten(),sites_num),
    #                     np.repeat(np.array(([0-margin,0-margin])),sites_num))).tolist()
    # up=np.concatenate((np.repeat(np.array([Nx+margin,Ny+margin]),sites_num),
    #                     np.repeat(np.array(([5,5],[5,5])).flatten(),sites_num),
    #                     np.repeat(np.array(([Nx+margin,Ny+margin])),sites_num))).tolist()

    low=optimizationParams["bound_low"][optimizationParams["paras_at"][0],optimizationParams["paras_at"][1]].tolist()
    up=optimizationParams["bound_up"][optimizationParams["paras_at"][0],optimizationParams["paras_at"][1]].tolist()
    toolbox = base.Toolbox()
    #generate_gene_random(op,Nx,Ny)->np.ndarray:
    """"""
    toolbox.register("p_random", generate_p_random,optimizationParams)
    toolbox.register("individualCreator", tools.initIterate, creator.Individual,toolbox.p_random )
    toolbox.register("population",tools.initRepeat,list,toolbox.individualCreator)
    toolbox.register("select", tools.selTournament, tournsize=3) #锦标赛
    toolbox.register("mate", cxTwoPointCopy) #交叉
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.02) #翻转位 不使用
    toolbox.register("mutate",tools.mutUniformInt,low=low,up=up,indpb=0.05) #int!

    def FitnessCalculationFunction(individual,gen):
        """算给定个体的适应度"""
        ri=random.randint(100,10000)
        gen_i=(gen+ (optimizationParams["topo_i"] if "topo_i" in optimizationParams else 0))
        x=linear_fem(Nx, Ny, Lx, Ly, optimizationParams, individual,f"{gen_i}/x_{gen_i}_{ri}", load=load,
                     location_fns=location_fns, fixed_location=fixed_location, gen=gen_i)
        individual.id=ri
        """需要引入目标函数"""
        result=objective(x)
        return (result,)
    #将evaluate注册为someFitnessCalculationFunction()的别名
    toolbox.register("evaluate",FitnessCalculationFunction)

    # 设置遗传算法参数
    population = toolbox.population(n=10)  # 种群大小
    c_table=["b","c","g","k","m","r","w","y"]
    ngen = optimizationParams["maxIters"] # 迭代次数
    for gen in tqdm(range(ngen),leave=False,desc="Generation iterating"):
        # 评估种群
        gen_array=np.repeat([gen],len(population))
        fitnesses = list(map(toolbox.evaluate, population,gen_array))
        i=0
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            plt.scatter(gen,fit[0],c=c_table[i%8],marker="+")
            # plt.scatter(gen,fit[1],c=c_table[i%8],marker="^")
            i+=1
        # 选择、交叉和变异
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        if gen < ngen-1:

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.5:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

        # 更新种群
        population[:] = offspring

    # 获取最佳个体
    fits=[]
    for ind in population:
        fits.append(ind.fitness.values)
    best_ind = tools.selBest(population, 1,"fitness")
    print(f"best fitness: {best_ind[0].fitness.values},id:{best_ind[0].id}")

if __name__ == "__main__":
    start_time = time.time()
    plt.figure()

    Nx, Ny = 256, 256
    Lx, Ly = 256., 256.
    sites_num = 50
    dim = 2
    margin = 10
    coordinates = np.indices((Nx, Ny))
    optimizationParams = {"coordinates": coordinates, "sites_num": sites_num, "Dm_dim": dim, "margin": margin}
    GA(optimizationParams)