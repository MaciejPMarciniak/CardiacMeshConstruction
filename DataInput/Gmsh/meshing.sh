#!/bin/bash
cd "$(dirname "$0")"

FILES="geofiles/*.geo"

for f in $FILES
do
  d=${f##*/}
  d=${d%_*_*}
  echo "File: $f, chamber: $d"

  /home/mat/gmsh-4.0.7-Linux64/bin/gmsh "$f" -3 -o /home/mat/Python/code/CardiacMeshConstruction/DataInput/Gmsh/tetra/"$d"_tetra.vtk

done
