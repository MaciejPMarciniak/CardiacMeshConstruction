# CardiacMeshConstruction
Generate tetrahedral meshes of adult human heart, using the statistical shape model.

## Requirements
Deformetrica: http://www.deformetrica.org/

Gmsh: https://gmsh.info/

## How2use
The file Weights.csv in DataInput/PCA should contain a table with desired weights of each of the 18 modes
provided as rows.

In pipeline.py set up the output path and the indices of meshes to be tetrahedralized and run the script.

## Additional information
If you would like to just download already existing meshes, check out

1. https://zenodo.org/record/4590294 for the original hearts on which the statistical shape model was based.
2. https://zenodo.org/record/4593739 for the meshes with extreme variation along the modes.
3. https://zenodo.org/record/4506930 for a set of 1000 meshes with randomly generated weights.