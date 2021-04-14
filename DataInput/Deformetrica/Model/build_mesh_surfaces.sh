#!/usr/bin/env bash

echo $PWD
cd DataInput/Deformetrica/Model
echo $PWD

deformetrica compute predef_model_shooting.xml -p optimization_parameters.xml -o output_shooting
rm output_shooting/*ControlPoints*
rm output_shooting/*Momenta*

mv output_shooting/Shooting*tp_10* ../../../DataOutput/Deformetrica/MorphedModels
rm -r output_shooting
cd ../../..
echo $PWD