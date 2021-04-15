// ---Input file
Merge "/home/mat/Python/code/CardiacMeshConstruction/DataInput/Gmsh/temp/Shooting_0__GeodesicFlow__IVC__tp_10__age_1.00.vtk";

// ---Change mesh parameters. The full list is available at https://gitlab.onelab.info/gmsh/gmshblob/master/Common/DefaultOptions.h ---
Mesh.Algorithm3D=1;// (1) delaunay (tetgen); (4) frontal (netgen - does not take CharacteristicLengthFactor into consideration - only propagates info from the surface)
Mesh.CharacteristicLengthFactor=0.85; // the bigger this value, the coarser the mesh. Optimal values for RV: 0.85
Mesh.CharacteristicLengthMax=1.0;
//Mesh.CharacteristicLengthMin=0.7;
Mesh.QualityType=1; //(0): SICN~signed inverse condition number; (1): SIGE~signed inverse gradient error; (2): gamma~vol/sum_face/max_edge, (3): Disto~minJ/maxJ
Mesh.OptimizeThreshold=0.3;
Mesh.OptimizeNetgen=1;
Mesh.Optimize=2; // Ideally there should be one more tetgen optimization than Netgen, to see the results at the end of the run.
Mesh.Format = 16; // (1) default, .msh object; (16) vtk object;

// ---Build the volume---
Surface Loop(1) = {1}; // Value in () is arbitrary, but {} has to point to the object created with the merge (only 1 in this case)
//+
Volume(1) = {1}; // Value in () is arbitrary, value {} points to the Surface Loop object
//+
Physical Volume(2)={1}; // Value in () has to be sequential (i.e. 1+1) [TO AVOID CONFLICT], value in {} points to Volume object
