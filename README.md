
### Here we supported 3 entrances of code:
* #### [manual.py](src/manual.py)
* #### [topo_mulitscale_principal.py](src/topo_mulitscale_principal.py)
* #### [topo_mutilscale.py](src/topo_mutilscale.py) (recommend)
######  (be careful with spelling mistakes)
#### [manual.py](src/manual.py) was used to generate samples for compression test.
#### The samples created by [topo_mulitscale_principal.py](src/topo_mulitscale_principal.py) are inserted with pseudo site points, along the first principal stress.
#### The [topo_mutilscale.py](src/topo_mutilscale.py) is deprecated, provides a very early version of the algorithm.

### THE JAX-FEM was modified and employed for this project.
#### You can follow [README.md](..%2Fjax-fem-voronoi%2FREADME.md) to install the requirements of JAX-FEM.
#### But you don't have to do the Option steps, as the modified jax-fem is already put at path [jax-fem-voronoi](..%2Fjax-fem-voronoi).
