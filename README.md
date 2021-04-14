# CardiacMeshConstruction
Generate tetrahedral meshes of adult human heart, using the statistical shape model.

## Requirements
Deformetrica: http://www.deformetrica.org/

Gmsh: https://gmsh.info/

## Adjustments
File CardiacMeshConstruction/DataInput/Gmsh/meshing.sh contains path to gmsh and path to temporary files with
tetrahedral meshes (line 11). These need to be adjusted manually. Sincere apologies.

## How2use
The file Weights.csv in DataInput/PCA should contain a table with desired weights of each of the 18 modes
provided as rows.

In pipeline.py set up the output path and the indices of meshes to be tetrahedralized and run the script.

