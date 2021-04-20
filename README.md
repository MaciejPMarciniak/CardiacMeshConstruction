# CardiacMeshConstruction
Generate tetrahedral meshes of adult human heart, using the statistical shape model.

## Requirements
Deformetrica: http://www.deformetrica.org/

Gmsh: https://gmsh.info/

## How2use
The file Weights.csv in DataInput/PCA should contain a table with desired weights of each of the 18 modes
provided as rows.

Pipeline.py conatins all the relevant functions. To generate heart meshes:
* provide the path to your gmsh executable,
* set up the output path and the indices of meshes to be tetrahedralized and run the script.

## Additional information
If you would like to just download already existing meshes, check out

1. https://zenodo.org/record/4590294 for the original hearts on which the statistical shape model was based.
2. https://zenodo.org/record/4593739 for the meshes with extreme variation along the modes.
3. https://zenodo.org/record/4506930 for a set of 1000 meshes with randomly generated weights.

## Credits

Please quote the following publication:

Rodero C, Strocchi M, Marciniak M, Longobardi S, Whitaker J, et al. (2021) Linking statistical shape models and simulated function in the healthy adult human heart. PLOS Computational Biology 17(4): e1008851. https://doi.org/10.1371/journal.pcbi.1008851

## License

The code is openly available. If this tool has been useful in your research, please reference this site https://github.com/MaciejPMarciniak/CardiacMeshConstruction