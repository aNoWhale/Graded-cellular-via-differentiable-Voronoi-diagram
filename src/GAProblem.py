import os
import random
import time

from deap import base, creator, tools, algorithms
import numpy as np
import jax.numpy as jnp
from softVoronoi import generate_gene_random
from linear_fem import linear_fem
from tqdm import tqdm

from src.softVoronoi import generate_voronoi


##### support for numpy
def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwriting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
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

# 定义问题和个体
start_time=time.time()

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)


Nx,Ny=100,100
Lx,Ly=100.,100.
sites_num=50
dim=2
margin=10
coordinates = np.indices((Nx, Ny))
optimizationParams = {"coordinates":coordinates,"sites_num":sites_num,"Dm_dim":dim,"margin":margin}
low=np.concatenate((np.repeat(np.array([0-margin,0-margin]),sites_num),
                    np.repeat(np.array(([1,0],[0,1])).flatten(),sites_num),
                    np.repeat(np.array(([0-margin,0-margin])),sites_num))).tolist()
up=np.concatenate((np.repeat(np.array([Nx+margin,Ny+margin]),sites_num),
                    np.repeat(np.array(([5,5],[5,5])).flatten(),sites_num),
                    np.repeat(np.array(([Nx+margin,Ny+margin])),sites_num))).tolist()
toolbox = base.Toolbox()
#generate_gene_random(op,Nx,Ny)->np.ndarray:
toolbox.register("p_random", generate_gene_random,optimizationParams,Nx,Ny)
toolbox.register("individualCreator", tools.initIterate, creator.Individual,toolbox.p_random )
toolbox.register("population",tools.initRepeat,list,toolbox.individualCreator)
toolbox.register("select", tools.selTournament, tournsize=3) #锦标赛
toolbox.register("mate", cxTwoPointCopy) #交叉
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.02) #翻转位 不使用
toolbox.register("mutate",tools.mutUniformInt,low=low,up=up,indpb=0.05) #int!


def FitnessCalculationFunction(individual,gen):
    """算给定个体的适应度"""
    def load_location_x(point):
         return jnp.isclose(point[0], Lx, atol=1e-5)
    def load_location_y(point):
         return jnp.isclose(point[1], Ly, atol=1e-5)
    def fixed_location_x(point):
        return jnp.isclose(point[0], 0., atol=1e-5)
    def fixed_location_y(point):
        return jnp.isclose(point[1], 0., atol=1e-5)
    ri=random.randint(0,1000)
    # compute compliance
    x=linear_fem(Nx,Ny,Lx,Ly,optimizationParams,individual,f"{gen}/x_{gen}_{ri}",load=np.array([100.,0.]),load_location=load_location_x,fixed_location=fixed_location_x)
    y=linear_fem(Nx,Ny,Lx,Ly,optimizationParams,individual,f"{gen}/y_{gen}_{ri}",load=np.array([0.,100.]),load_location=load_location_y,fixed_location=fixed_location_y)
    return x,y
#将evaluate注册为someFitnessCalculationFunction()的别名
toolbox.register("evaluate",FitnessCalculationFunction)

# 设置遗传算法参数
population = toolbox.population(n=10)  # 种群大小

ngen = 10  # 迭代次数
for gen in tqdm(range(ngen),leave=False,desc="Generation iterating"):
    # 评估种群
    gen_array=np.repeat([gen],len(population))
    fitnesses = list(map(toolbox.evaluate, population,gen_array))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

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
print(f"最优值: {best_ind[0].fitness.values}")
best_voronoi=generate_voronoi(optimizationParams,best_ind[0])
data_path = os.path.join(os.path.dirname(__file__), 'data')
best_dir = 'data/best'
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
np.save(f'{best_dir}/best_voronoi.npy', best_voronoi)
np.save(f"{best_dir}/op.npy",np.array(optimizationParams))
end_time=time.time()
print(f"time spent: {end_time-start_time}")