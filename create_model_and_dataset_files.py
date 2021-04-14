import glob
import os
from pathlib import Path
from lxml.etree import Element, SubElement, tostring
import regex as re


class BuildXML:
    """
    In deformetrica files (model.xml, data-set.xml, optimization-parameters.xml) many of the fields depend on the files
    used to build the model. This superclass finds all the relevant files (i.e. the file names) and puts them in
    a format easy to parse with xml tree. It also asserts that if a model is split into elements, all elements
    of a given model exist.

    :param source: path to a directory with files relevant for building models
    :param key_word: a label to put in 'object_id' field if models are treated as single shapes
    :param list_of_elements: list provided to find all elements belonging to a model. If not provided, model(s)
    is(are) searched.
    """

    _list_of_elements_24 = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'MV', 'TV', 'AV', 'PV',
                            'APP', 'LSPV', 'LIPV', 'RIPV', 'RSPV', 'SVC', 'IVC',
                            'AB', 'RIPVB', 'LIPVB', 'LSPVB', 'RSPVB', 'SVCB', 'IVCB']

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def __init__(self, source='data', output=None, key_word=None,  list_of_elements=None):

        self.source = source
        self.output = self.source if output is None else output
        self.list_of_elements = self._list_of_elements_24 if list_of_elements is None else list_of_elements
        self.key_word = key_word
        self.element_files = []
        self.elem = ''
        self.files = None
        self.current_id = None
        self.top = None

    def find_elements(self):

        list_of_elements_with_id = []
        for f in self.files:
            if self.current_id in f:
                list_of_elements_with_id.append(f)
        return list_of_elements_with_id
    #  -------------------------------------------

    def find_ids(self):

        list_of_ids = []
        for filename in self.files:
            file_id = re.findall(r'\d+$', filename)  # Takes all numbers from the filename, if $ ommitted
            list_of_ids.append(''.join(file_id),)
        return list_of_ids
    #  -------------------------------------------

    def get_subject_filename(self):
        return glob.glob1(self.source, '*'+self.current_id+'*')[0]
    #  -------------------------------------------

    def get_element_filename(self):

        found_element_file = []
        for element_file in self.element_files:
            if self.elem in element_file:
                found_element_file = element_file
                return found_element_file
        if not found_element_file:
            exit('Element {} in subject {} is missing!'.format(self.elem, self.current_id))
    #  -------------------------------------------

    def write_xml(self, output=None):

        if output is not None:
            self.output = output
        filename = os.path.join(self.output, (self.top.tag+'.xml').replace('-', '_'))
        pretty_top = tostring(self.top, pretty_print=True, xml_declaration=True)
        with open(filename, 'wb') as f:
            f.write(pretty_top)
    #  -------------------------------------------


class DataSet(BuildXML):
    """Class for building the data-set.xml file. Saved as data-set.xml in the 'source' directory.

    :param source key_word list_of_elements: parameters passed to the superclass BuildXML
    """
    def __init__(self, source='data', key_word=None,  list_of_elements=None):

        super().__init__(source, key_word, list_of_elements)

        self.files = glob.glob(os.path.join(self.source, '*.vtk'))
        self.files.sort()
        self.number_of_cases = len(self.files)
        self.ids = list(set(self.find_ids()))
        self.ids.sort()
        self.current_id = self.ids[0]
        self.element_files = []
        self.elem = ''
        self.structure = ''

    def build_with_lxml_tree(self):
        self.top = Element('data-set')
        for self.current_id in self.ids:
            self.element_files = self.find_elements()

            subject = SubElement(self.top, 'subject', id='sub{}'.format(self.current_id))
            visit = SubElement(subject, 'visit', id='varifold')

            if self.list_of_elements:
                for element in self.list_of_elements:
                    self.elem = '_' + element
                    filename = SubElement(visit, 'filename', object_id=element)
                    filename.text = self.get_element_filename()
            else:
                filename = SubElement(visit, 'filename', object_id=self.key_word)
                filename.text = os.path.join(self.source, self.get_subject_filename())


class ModelAtlas(BuildXML):
    """Class for building the model.xml file that is used for atlas construction. Saved as model.xml in the 'source'
    directory. Model.xml contains all the parametrization of the atlas construction process. The kernel widths, types of
    objects and attachments and default values for parameters not included in this class are described on the wiki page
    of deformetrica:
    https://gitlab.icm-institute.org/aramislab/deformetrica/wikis/3_user_manual/3.2_model_xml_file

    :param source, key_word, list_of_elements: parameters passed to the superclass BuildXML
    :param prototype_id: ID of the model to be used as a template. Should be as close to mean and represent the topology
    of the family of models
    :param deformation_kernel_width, prototype_kernel_width, noise_std: Parameters to control the quality of atlas
    construction. Described in detail on deformetrica website:
    https://gitlab.icm-institute.org/aramislab/deformetrica/wikis/1_lddmm
    """
    def __init__(self, source='data', template_path=None, output=None, key_word=None, list_of_elements=None,
                 prototype_id=None, deformation_kernel_width=10, prototype_kernel_width=10, noise_std=0.1,
                 deformable_object='Polyline'):

        super().__init__(source, output, key_word, list_of_elements)

        self.template_path = source if template_path is None else os.path.join(self.__location__, template_path)
        self.deformation_kernel_width = str(deformation_kernel_width)
        self.prototype_kernel_width = str(prototype_kernel_width)
        self.deformable_object = deformable_object
        self.noise_std = str(noise_std)
        self.prototype_id = str(prototype_id).zfill(2)
        self.files = []
        if not list_of_elements:
            self.files = glob.glob(os.path.join(self.template_path, '*{}*.vtk'.format(self.prototype_id)))
            if not self.files:
                exit('Prototype file missing, check the source path and folder.')
        else:
            for elem in list_of_elements:
                self.elem = elem
                element_files = glob.glob(os.path.join(self.template_path,
                                                       '*{}*_{}.vtk'.format(self.prototype_id, self.elem)))
                self.files.extend(element_files)
        self.files.sort()
        self.number_of_cases = len(self.files)
        self.ids = self.find_ids()
        self.current_id = self.ids[0]

    def insert_default_branches(self, obj):
        dot = SubElement(obj, 'deformable-object-type')
        dot.text = self.deformable_object
        at = SubElement(obj, 'attachment-type')
        at.text = 'Varifold'
        kernel_width = SubElement(obj, 'kernel-width')
        kernel_width.text = self.prototype_kernel_width
        kernel_type = SubElement(obj, 'kernel-type')
        kernel_type.text = 'keops'
        noise_st = SubElement(obj, 'noise-std')
        noise_st.text = self.noise_std
    #  -------------------------------------------

    def build_with_lxml_tree(self):

        self.top = Element('model')
        model_type = SubElement(self.top, 'model-type')
        model_type.text = 'DeterministicAtlas'
        dimension = SubElement(self.top, 'dimension')
        dimension.text = '3'
        template = SubElement(self.top, 'template')

        self.element_files = self.find_elements()

        if self.list_of_elements:
            for element in self.list_of_elements:
                self.elem = '_' + element
                obj = SubElement(template, 'object', id=element)
                self.insert_default_branches(obj)
                filename = SubElement(obj, 'filename')
                filename.text = self.get_element_filename()
        else:
            obj = SubElement(template, 'object', id=self.key_word)
            self.insert_default_branches(obj)
            filename = SubElement(obj, 'filename')
            filename.text = self.files[0]

        deformation_parameters = SubElement(self.top, 'deformation-parameters')
        kernel_width = SubElement(deformation_parameters, 'kernel-width')
        kernel_width.text = self.deformation_kernel_width
        kernel_type = SubElement(deformation_parameters, 'kernel-type')
        kernel_type.text = 'keops'  # Allows for CUDA usage - expedites the process exponentially


class ModelShooting(BuildXML):
    """Class for building the model.xml file for shooting (transforming with warp fields).
        Saved as model.xml in the 'source' directory. Model.xml contains information about model to be deformed and
        corresponding Control_Points.txt and Momenta.txt files.
        Notes:
        Control_Points.txt must be provided from the atlas construction process and exist in 'source' directory.
        Momenta.txt must be in the same format as momenta generated by deformetrica and exist in 'source' directory.
        The default values for parameters not included in this class are described on the wiki page of deformetrica:
        https://gitlab.icm-institute.org/aramislab/deformetrica/wikis/3_user_manual/3.2_model_xml_file

        :param source, key_word, list_of_elements: parameters passed to the superclass BuildXML
        :param deformation_kernel_width: Parameters to control the quality of shooting. It should correspond to the
        width used for atlas construction of the model. Described in detail on deformetrica website:
        https://gitlab.icm-institute.org/aramislab/deformetrica/wikis/1_lddmm
        """
    def __init__(self, source='data', template_path=None, output_path=None, key_word=None, list_of_elements=None,
                 momenta_filename=None, deformation_kernel_width=10, deformable_object='Polyline'):

        super().__init__(source, output_path, key_word, list_of_elements)

        self.template_path = source if template_path is None else os.path.join(self.__location__, template_path)
        self.momenta_file = os.path.join(self.template_path, momenta_filename)
        self.deformation_kernel_width = str(deformation_kernel_width)
        self.deformable_object = deformable_object
        self.files = []
        if not list_of_elements:
            self.files = glob.glob(os.path.join(self.template_path, '*.vtk'))
        else:
            for elem in list_of_elements:
                self.elem = elem
                element_files = glob.glob(os.path.join(self.template_path, '*_{}.vtk'.format(self.elem)))
                self.files.extend(element_files)
        self.files.sort()
        self.number_of_cases = len(self.files)
        self.ids = self.find_ids()
        self.current_id = self.ids[0]

    def build_with_lxml_tree(self):

        self.top = Element('model')
        model_type = SubElement(self.top, 'model-type')
        model_type.text = 'Shooting'
        dimension = SubElement(self.top, 'dimension')
        dimension.text = '3'
        template = SubElement(self.top, 'template')

        self.element_files = self.find_elements()
        if self.list_of_elements:
            for element in self.list_of_elements:
                self.elem = '_' + element
                obj = SubElement(template, 'object', id=element)
                filename = SubElement(obj, 'filename')
                dot = SubElement(obj, 'deformable-object-type')
                dot.text = self.deformable_object
                filename.text = self.get_element_filename()
        else:
            obj = SubElement(template, 'object', id=self.key_word)
            filename = SubElement(obj, 'filename')
            dot = SubElement(obj, 'deformable-object-type')
            dot.text = self.deformable_object
            filename.text = self.files[0]

        initial_control_points = SubElement(self.top, 'initial-control-points')
        initial_control_points.text = os.path.join(self.template_path,
                                                   'DeterministicAtlas__EstimatedParameters__ControlPoints.txt')
        initial_momenta = SubElement(self.top, 'initial-momenta')
        initial_momenta.text = os.path.join(self.template_path, self.momenta_file)
        deformation_parameters = SubElement(self.top, 'deformation-parameters')
        kernel_width = SubElement(deformation_parameters, 'kernel-width')
        kernel_width.text = self.deformation_kernel_width
        kernel_type = SubElement(deformation_parameters, 'kernel-type')
        kernel_type.text = 'keops'  # Allows for CUDA usage - expedites the proces exponentially
    #  -------------------------------------------


def prepare_data_for_atlas_construction(source_path=os.path.join(str(Path.home()), 'Python', 'data', 'h_case'),
                                        output_path='/home/mat/Deformetrica/deterministic_atlas_ct',
                                        key_word='h_case'):
    ds = DataSet(source=os.path.join(source_path),
                 key_word=key_word)
    ds.build_with_lxml_tree()
    ds.write_xml(output_path)
    mdl = ModelAtlas(source=source_path,
                     key_word=key_word,
                     prototype_id=13,
                     deformation_kernel_width=10,  # lambda_V
                     prototype_kernel_width=10,
                     deformable_object='Polyline')  # lambda_W
    mdl.build_with_lxml_tree()
    mdl.write_xml(output_path)


def prep_modes_for_reconstruction(source_path=os.path.join(str(Path.home()), 'Deformetrica', 'deterministic_atlas_ct',
                                                           'output_separate_tmp10_def10_prttpe13_corrected',
                                                           'Decomposition'),
                                  output_path='/home/mat/Deformetrica/deterministic_atlas_ct'):

    # extreme modes, requires the $ in the wild card in search for IDs [method find_ids()]
    mom_mdl = ModelShooting(source=source_path,
                            key_word='Template',  # Template copied to Decomposition folder
                            momenta_filename='Extreme_Momenta.txt',
                            deformation_kernel_width=10,
                            deformable_object='Polyline')
    print(mom_mdl.source)
    mom_mdl.build_with_lxml_tree()
    mom_mdl.write_xml(output_path)
    os.rename(os.path.join(output_path, 'model.xml'), os.path.join(output_path, 'model_shooting.xml'))


def prep_random_cohort_for_reconstruction(source_path=os.path.join(str(Path.home()),
                                                                   'Deformetrica', 'deterministic_atlas_ct',
                                                                   'output_separate_tmp10_def10_prttpe13_corrected',
                                                                   'Decomposition', 'RandomMeshGeneration'),
                                          output_path='/home/mat/Deformetrica/deterministic_atlas_ct'):

    for i in range(1, 41):
        mom_mdl = ModelShooting(source=source_path,
                                key_word='Template',  # Template copied to Decomposition folder
                                momenta_filename='Randomly_weighted_modes_{}.txt'.format(i),
                                deformation_kernel_width=10)
        mom_mdl.build_with_lxml_tree()
        mom_mdl.write_xml(output_path)
        os.rename(os.path.join(output_path, 'model.xml'), os.path.join(output_path, 'model_shooting_{}.xml'.format(i)))


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


if __name__ == '__main__':

    prep_predefined_cohort_for_reconstruction()
