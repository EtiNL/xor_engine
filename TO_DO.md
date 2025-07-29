# Cuda_wrapper
- look at dependences between node graphs / conditionnal nodes (if/while/switch)
- look at cuStreamBeginCapture_v2 to tell cuda which nodes can be launched concurently
- add sdf deformation/union/intersection... lattice creation

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
- add the spawn to the ecs update logic
- faire des arbres de dependances/contraintes/interactions entre entities
- change one texture_buffer per SdfObject to possibly shared texture buffer

# main
- make the code more sustainable over time

# AI features
- denoiser
- neural sdf
- PINN based simulations

