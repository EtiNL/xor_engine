# Cuda_wrapper
- look at dependences between node graphs / conditionnal nodes (if/while/switch)
- look at cuStreamBeginCapture_v2 to tell cuda which nodes can be launched concurently
# Ray marching
- Compare in gpu multiple spp (megakernel) vs one kernel launch per spp
- make a stat based adaptative sampler

# Ray tracing
- implement a first ray tracing kernel
- glTF importer

# Path tracing

# running time edition feature
- make a running time edition feature to create scenes in real time (need ECS.4 taskpoint)
- make a scene uploader/saver
- permettre d'éditer les arbres de dépendances/contraintes/interaction entre entities

# ECS
- put sample per pixel in the camera
- put light in ecs and make it customizable
- add some basic physics simulations
- faire des arbres de dependances/contraintes/interactions entre entities

# main
- make the code more sustainable over time
- add sdf deformation/union/intersection... 
- add lattice boundaries
- integrate solid color to renderables for them to use either a texture or a color and then other materials like procedural patterns, metallic/roughness by evolving the Material enum and the GPU representation (e.g., a small material descriptor struct).




# AI features
- denoiser
- neural sdf
- PINN based simulations

