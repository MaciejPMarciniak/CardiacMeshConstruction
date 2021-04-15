import os
import pandas as pd
import numpy as np
import subprocess

from decomposition_pca import PcaWithScaling
from read_momenta import HandleMomenta
from create_model_and_dataset_files import ModelShooting
from MeshGeneration import MeshTetrahedralization


def pca_get_predefined_combinations(path_to_data=os.path.join('DataInput', 'PCA'),
                                    data_filename='OriginalDataSet.csv',
                                    weights_filename='Weights.csv',
                                    output_path=os.path.join('DataOutput', 'PCA', 'WeightedModes')):

    momenta_pca = PcaWithScaling(path_to_data, data_filename, output_path, number_of_components=18)
    momenta_pca.decompose_with_pca()
    weights = pd.read_csv(os.path.join(path_to_data, weights_filename))
    weighted_momenta = np.zeros((len(weights), momenta_pca.components.shape[1]))

    for single_set_of_weights in weights.iterrows():
        weighted_momenta[single_set_of_weights[0], :] = momenta_pca.get_n_main_modes(single_set_of_weights[1].values)

    momenta_pca.save_result('Weighted_momenta.csv', weighted_momenta)


def generate_predefined_weighted_modes(path_to_momenta=os.path.join('DataOutput', 'PCA', 'WeightedModes'),
                                       output_path=os.path.join('DataInput', 'Deformetrica', 'Template')):

    momenta_vec = 'Weighted_momenta.csv'
    momenta_new = 'Weighted_momenta.txt'
    momenta = HandleMomenta(path_to_momenta, momenta_vec,  output_path=output_path, output_filename=momenta_new,
                            configuration='extreme')
    momenta.save_momenta_matrix_in_deformetrica_format()


def prep_predefined_cohort_for_reconstruction(source_path=os.path.join('DataInput', 'PCA'),
                                              template_path=os.path.join('DataInput', 'Deformetrica', 'Template'),
                                              output_path=os.path.join('DataInput', 'Deformetrica', 'Model')):

    mom_mdl = ModelShooting(source=source_path,
                            template_path=template_path,
                            output_path=output_path,
                            key_word='Template',  # Template copied to Decomposition folder
                            momenta_filename='Weighted_momenta.txt',
                            deformation_kernel_width=10)
    mom_mdl.build_with_lxml_tree()
    mom_mdl.write_xml(output_path)
    os.rename(os.path.join(output_path, 'model.xml'), os.path.join(output_path, 'predef_model_shooting.xml'))


def perform_deformation(bash_file_path=os.path.join('DataInput', 'Deformetrica', 'Model'),
                        bash_file='build_mesh_surfaces.sh'):
    subprocess.call(os.path.join(bash_file_path, bash_file))


def write_meshing_file(gmsh_exe_path,
                       tetra_dir_path=os.path.join(
                           os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
                           'DataInput', 'Gmsh', 'tetra')):

    with open(os.path.join('DataInput', 'Gmsh', 'meshing.sh'), 'w') as f:
        f.writelines(['#!/bin/bash\n',
                      'cd "$(dirname "$0")"\n\n',
                      'FILES="geofiles/*.geo"\n\n',
                      'for f in $FILES\n',
                      'do\n',
                      '  d=${f##*/}\n',
                      '  d=${d%_*_*}\n',
                      '  echo "File: $f, chamber: $d"\n\n',
                      '  {} "$f" -3 -o {}/"$d"_tetra.vtk\n\n'.format(gmsh_exe_path, tetra_dir_path),
                      'done'])


def merged_shapes_generation(id_from=0,
                             id_to=20,
                             models_path=os.path.join('DataOutput', 'Deformetrica', 'MorphedModels'),
                             output_path='/media/mat/BEDC-845B/FullPipeline',
                             merged_type='tetra'):

    assert id_from < id_to, 'Insert proper range of IDs to create meshes from'
    assert os.path.exists(models_path), 'Provide proper path to models'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for j in range(id_from, id_to):

        sup = MeshTetrahedralization(main_path=os.path.join('DataInput', 'Gmsh'),
                                     models_path=models_path,
                                     geo_dir='geofiles',
                                     temp_dir='temp',
                                     output_path=output_path,
                                     k_model=j,
                                     template=False)
        if merged_type == 'surface':
            sup.pipeline_aggr_surf_mesh()
        elif merged_type == 'tetra':
            sup.pipeline_surf_2_tetra_mesh()
        else:
            exit('Provide proper model generation type: "tetra" or "surface"')


def pipeline(gmsh_exe_path,
             tetrahedralized_mesh_output_path='/media/mat/BEDC-845B/FullPipeline',
             run_only_tetra=False):

    if not run_only_tetra:
        print('PCA on 19 cases')
        pca_get_predefined_combinations()

        print('Generating momenta')
        generate_predefined_weighted_modes()

        print('Prepare files necessary for deformation')
        prep_predefined_cohort_for_reconstruction()

        print('Generate tetrahedralization file')
        write_meshing_file(gmsh_exe_path)

        print('Run deformetrica')
        perform_deformation()

    print('Perform tetrahedralization')
    merged_shapes_generation(0, 1, output_path=tetrahedralized_mesh_output_path)


if __name__ == '__main__':

    absolute_path_to_gmsh_exe = r'/home/mat/gmsh-4.0.7-Linux64/bin/gmsh'
    pipeline(absolute_path_to_gmsh_exe)
