import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import os
import glob
from PIL import Image
# from MeshAlignment import calculate_rotation, calculate_plane_normal
# from MeshSlices import create_plax_slices

# 01. LV myocardium (endo + epi)
# 02. RV myocardium (endo + epi)
# 03. LA myocardium (endo + epi)
# 04. RA myocardium (endo + epi)
#
# 05. Aorta
# 06. Pulmonary artery
#
# 07. Mitral valve
# 08. Triscupid valve
#
# 09. Aortic valve
# 10. Pulmonary valve

# 11. Appendage
# 12. Left superior pulmonary vein
# 13. Left inferior pulmonary vein
# 14. Right inferior pulmonary vein
# 15. Right superior pulmonary vein
#
# 16. Superior vena cava
# 17. Inferior vena cava

# 18. Appendage border
# 19. Right inferior pulmonary vein border
# 20. Left inferior pulmonary vein border
# 21. Left superior pulmonary vein border
# 22. Right superior pulmonary vein border
# 23. Superior vena cava border
# 24. Inferior vena cava border


class Model:

    list_of_elements = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'MV', 'TV', 'AV', 'PV',
                        'APP', 'LSPV', 'LIPV', 'RIPV', 'RSPV', 'SVC', 'IVC',
                        'AB', 'RIPVB', 'LIPVB', 'LSPVB', 'RSPVB', 'SVCB', 'IVCB']

    # TODO: Add the dictionary with labels, to use in alignment

    def __init__(self, filename='h_case06.vtk', to_polydata=False):

        self.filename, self.input_type = filename.split('.')
        print(self.filename)

        # Initialize log of mesh manipulations
        w = vtk.vtkFileOutputWindow()
        w.SetFileName(self.filename.split('/')[0] + '/errors.txt')
        vtk.vtkOutputWindow.SetInstance(w)

        print('Reading the data from {}.{}...'.format(self.filename, self.input_type))
        if self.input_type == 'obj':
            self.mesh, self.scalar_range = self.read_obj()
        elif self.input_type == 'vtp':
            self.mesh, self.scalar_range = self.read_vtp()
        else:
            self.mesh, self.scalar_range = self.read_vtk(to_polydata)

        self.center_of_model = self.get_center(self.mesh)
        print('Model centered at: {}'.format(self.center_of_model))
        self.label = 0

    @staticmethod
    def get_center(_mesh):
        centerofmass = vtk.vtkCenterOfMass()
        centerofmass.SetInputData(_mesh.GetOutput())
        centerofmass.Update()
        return np.array(centerofmass.GetCenter())

    def visualize_mesh(self, display=True):
        # Create the mapper that corresponds the objects of the vtk file into graphics elements
        mapper = vtk.vtkDataSetMapper()
        try:
            mapper.SetInputData(self.mesh.GetOutput())
        except TypeError:
            print('Can\'t get output directly')
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(self.mesh.GetOutputPort())
        mapper.SetScalarRange(self.scalar_range)

        # Create the Actor
        camera = vtk.vtkCamera()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(vtk.util.colors.red)
        actor.GetProperty().SetOpacity(0.5)

        # Create the Renderer
        renderer = vtk.vtkRenderer()
        renderer.ResetCameraClippingRange()
        renderer.AddActor(actor)  # More actors can be added
        renderer.SetActiveCamera(camera)
        renderer.SetBackground(1, 1, 1)  # Set background to white

        # Create the RendererWindow
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(600, 600)
        render_window.Render()

        # Display the mesh
        # noinspection PyArgumentList
        if display:
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
            interactor.Initialize()
            interactor.Start()
        else:
            return render_window

    # -----3D rigid transformations---------------------------------------------------------------------------

    def rotate(self, alpha=0, beta=0, gamma=0, rotation_matrix=None):
        print('rotating')
        rotate = vtk.vtkTransform()
        if rotation_matrix is not None:
            translation_matrix = np.eye(4)
            translation_matrix[:-1, :-1] = rotation_matrix
            print('Translation matrix (rotation):\n', translation_matrix)
            rotate.SetMatrix(translation_matrix.ravel())
        else:
            rotate.Identity()
            rotate.RotateX(alpha)
            rotate.RotateY(beta)
            rotate.RotateZ(gamma)
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(rotate)
        transformer.Update()
        self.mesh = transformer
        self.center_of_model = self.get_center(self.mesh)

    def scale(self, factor=(0.001, 0.001, 0.001)):
        print('scaling')
        scale = vtk.vtkTransform()
        scale.Scale(factor[0], factor[1], factor[2])
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(scale)
        transformer.Update()
        self.mesh = transformer
        self.center_of_model = self.get_center(self.mesh)
        print(self.center_of_model)

    def translate(self, rotation_matrix, translation_vector):
        print('translating')
        translate = vtk.vtkTransform()
        translation_matrix = np.eye(4)
        translation_matrix[:-1, :-1] = rotation_matrix
        translation_matrix[:-1, -1] = translation_vector
        print('Translation matrix:\n', translation_matrix)
        translate.SetMatrix(translation_matrix.ravel())
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(translate)
        transformer.Update()
        self.mesh = transformer
        self.center_of_model = self.get_center(self.mesh)

    def translate_to_center(self, label=None):
        # vtkTransform.SetMatrix - enables for applying 4x4 transformation matrix to the meshes
        # if label is provided, translates to the center of the element with that label
        print('translating o center')
        translate = vtk.vtkTransform()
        if label is not None:
            central_element = self.threshold(label, label)
            center_of_element = self.get_center(central_element)
            translate.Translate(-center_of_element[0], -center_of_element[1], -center_of_element[2])
        else:
            translate.Translate(-self.center_of_model[0], -self.center_of_model[1], -self.center_of_model[2])
        translate.Update()
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(translate)
        transformer.Update()
        self.mesh = transformer
        self.center_of_model = self.get_center(self.mesh)
        print(self.center_of_model)

    # -----Mesh manipulation----------------------------------------------------------------------------------
    def apply_modes(self, modes_with_scales):
        print('applying modes')
        for mode, scale in modes_with_scales.items():
            print('Applying ' + mode + ' multiplied by ' + str(scale))
            self.mesh.GetOutput().GetPointData().SetActiveVectors(mode)
            warp_vector = vtk.vtkWarpVector()
            warp_vector.SetInputConnection(self.mesh.GetOutputPort())
            warp_vector.SetScaleFactor(scale)
            warp_vector.Update()
            self.mesh = warp_vector

    def build_tag(self, label):
        print('building tag')
        self.label = label
        tag = vtk.vtkIdFilter()
        tag.CellIdsOn()
        tag.PointIdsOff()
        tag.SetInputConnection(self.mesh.GetOutputPort())
        tag.SetIdsArrayName('elemTag')
        tag.Update()
        self.mesh = tag

    @staticmethod
    def calculate_bounding_box_diagonal(bounds):
        return np.sqrt(np.power(bounds[0] - bounds[1], 2) +
                       np.power(bounds[2] - bounds[3], 2) +
                       np.power(bounds[4] - bounds[5], 2))

    def calculate_maximum_distance(self, bounds, target_offset):
        d = self.calculate_bounding_box_diagonal(bounds)
        return target_offset / d

    def change_tag_label(self):
        print('changing tag label')
        size = self.mesh.GetOutput().GetAttributes(1).GetArray(0).GetSize()
        for id in range(size):
            self.mesh.GetOutput().GetAttributes(1).GetArray(0).SetTuple(id, (float(self.label),))

    def clean_polydata(self, tolerance=0.005, remove_lines=False):
        print('cleaning polydata')
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(self.mesh.GetOutputPort())
        cleaner.SetTolerance(tolerance)
        cleaner.ConvertLinesToPointsOn()
        cleaner.ConvertPolysToLinesOn()
        cleaner.ConvertStripsToPolysOn()
        cleaner.Update()
        self.mesh = cleaner
        if remove_lines:
            self.mesh.GetOutput().SetLines(vtk.vtkCellArray())

    def contouring(self):
        print('contouring')
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(self.mesh.GetOutputPort())
        contour.GenerateTrianglesOn()
        contour.SetValue(0, 10.0)
        contour.Update()
        self.mesh = contour

    def decimation(self, reduction=50):
        print('decimating')
        decimation = vtk.vtkQuadricDecimation()
        decimation.SetInputConnection(self.mesh.GetOutputPort())
        decimation.VolumePreservationOn()
        decimation.SetTargetReduction(reduction / 100)  # percent of kept triangles
        decimation.Update()
        self.mesh = decimation

    def delaunay2d(self):
        print('triangulating 2D')
        delaunay2d = vtk.vtkDelaunay2D()
        delaunay2d.SetInputConnection(self.mesh.GetOutputPort())
        delaunay2d.Update()
        self.mesh = delaunay2d

    def delaunay3d(self):
        print('triangulating 3D')
        delaunay3d = vtk.vtkDelaunay3D()
        delaunay3d.SetInputConnection(self.mesh.GetOutputPort())
        delaunay3d.Update()
        self.mesh = delaunay3d

    def extract_surface(self):
        print('extracting surface')
        # Get surface of the mesh
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(self.mesh.GetOutput())
        surface_filter.Update()
        self.mesh = surface_filter

    def fill_holes(self, hole_size=10.0):
        print('filling holes')
        filling_filter = vtk.vtkFillHolesFilter()
        filling_filter.SetInputConnection(self.mesh.GetOutputPort())
        filling_filter.SetHoleSize(hole_size)
        filling_filter.Update()
        self.mesh = filling_filter

    def get_external_surface(self):
        print('getting external surface')
        _center = np.zeros(3)
        _bounds = np.zeros(6)
        _ray_start = np.zeros(3)
        cell_id = vtk.mutable(-1)
        xyz = np.zeros(3)
        pcoords = np.zeros(3)
        t = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        _surf = 1.1

        self.mesh.GetOutput().GetCenter(_center)
        self.mesh.GetOutput().GetPoints().GetBounds(_bounds)
        for j in range(3):
            _ray_start[j] = _bounds[2 * j + 1] * _surf

        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(self.mesh.GetOutput())
        cell_locator.BuildLocator()
        cell_locator.IntersectWithLine(_ray_start, _center, 0.0001, t, xyz, pcoords, sub_id, cell_id)

        connectivity_filter = vtk.vtkConnectivityFilter()
        connectivity_filter.SetInputConnection(self.mesh.GetOutputPort())
        connectivity_filter.SetExtractionModeToCellSeededRegions()
        connectivity_filter.InitializeSeedList()
        connectivity_filter.AddSeed(cell_id)
        connectivity_filter.Update()
        self.mesh = connectivity_filter  # UnstructuredGrid

    def implicit_modeller(self, distance):
        print('implicit modelling')
        # Create implicit model with vtkImplicitModeller at the 'distance' (in mesh's units) from the provided geometry.
        bounds = np.array(self.mesh.GetOutput().GetPoints().GetBounds())
        max_dist = self.calculate_maximum_distance(bounds, distance)
        imp = vtk.vtkImplicitModeller()
        imp.SetInputConnection(self.mesh.GetOutputPort())
        imp.SetSampleDimensions(500, 500, 500)
        imp.SetMaximumDistance(max_dist)
        imp.ScaleToMaximumDistanceOn()
        imp.SetModelBounds(*(bounds * 1.5))
        imp.CappingOn()
        imp.SetCapValue(255)
        imp.Update()
        self.mesh = imp

    def measure_average_edge_length(self):
        print('Average edge length')
        size = vtk.vtkCellSizeFilter()
        size.SetInputConnection(self.mesh.GetOutputPort())
        size.Update()
        print(size)

    def normals(self):
        print('getting normals')
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(self.mesh.GetOutputPort())
        normals.FlipNormalsOn()
        normals.Update()
        self.mesh = normals

    def pass_array(self):
        print('passing arrays')
        passer = vtk.vtkPassArrays()
        passer.SetInputConnection(self.mesh.GetOutputPort())
        passer.AddCellDataArray('elemTag')
        passer.Update()
        self.mesh = passer

    def resample_to_image(self, label_name='elemTag'):
        print('resampling to image')
        resampler = vtk.vtkResampleToImage()
        resampler.SetInputConnection(self.mesh.GetOutputPort())
        resampler.UseInputBoundsOff()
        bounds = np.array(self.mesh.GetOutput().GetBounds())
        bounds[:4] = bounds[:4] + 0.1 * bounds[:4]
        assert np.sum(bounds[4:] < 0.001), 'The provided slice must be 2D and must be projected on the XY plane'

        resampler.SetSamplingBounds(*bounds[:5], 1.01)
        resampler.SetSamplingDimensions(1024, 1024, 1)
        resampler.Update()

        img_as_array = vtk_to_numpy(resampler.GetOutput().GetPointData().GetArray(label_name))
        img_as_array = img_as_array.reshape((int(np.sqrt(img_as_array.shape[0])), int(np.sqrt(img_as_array.shape[0]))))

        return img_as_array

    def slice_extraction(self, origin, normal):
        print('extracting slices')
        # create a plane to cut (xz normal=(1,0,0);XY =(0,0,1),YZ =(0,1,0)
        plane = vtk.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)

        # create cutter
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputConnection(self.mesh.GetOutputPort())
        cutter.Update()

        self.mesh = cutter

    def smooth_laplacian(self, number_of_iterations=50):
        print('laplacian smoothing')
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(self.mesh.GetOutputPort())
        smooth.SetNumberOfIterations(number_of_iterations)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOn()
        smooth.Update()
        self.mesh = smooth

    def smooth_window(self, number_of_iterations=30, pass_band=0.05):
        print('window smoothing')
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputConnection(self.mesh.GetOutputPort())
        smooth.SetNumberOfIterations(number_of_iterations)
        smooth.BoundarySmoothingOn()
        smooth.FeatureEdgeSmoothingOff()
        smooth.SetPassBand(pass_band)
        smooth.NonManifoldSmoothingOn()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        self.mesh = smooth

    def subdivision(self, number_of_subdivisions=3):
        print('subdividing')
        self.normals()
        subdivision = vtk.vtkLinearSubdivisionFilter()
        subdivision.SetNumberOfSubdivisions(number_of_subdivisions)
        subdivision.SetInputConnection(self.mesh.GetOutputPort())
        subdivision.Update()
        self.mesh = subdivision
        self.visualize_mesh(True)

    def tetrahedralize(self, leave_tetra_only=True):
        print('creating tetrahedrons')
        tetra = vtk.vtkDataSetTriangleFilter()
        if leave_tetra_only:
            tetra.TetrahedraOnlyOn()
        tetra.SetInputConnection(self.mesh.GetOutputPort())
        tetra.Update()
        self.mesh = tetra

    def threshold(self, low=0, high=100):
        print('thresholding')
        threshold = vtk.vtkThreshold()
        threshold.SetInputConnection(self.mesh.GetOutputPort())
        threshold.ThresholdBetween(low, high)
        threshold.Update()
        # choose scalars???
        return threshold

    def ug_geometry(self):
        print('setting unstructured grid geometry')
        geometry = vtk.vtkUnstructuredGridGeometryFilter()
        print(geometry.GetDuplicateGhostCellClipping())
        geometry.SetInputConnection(self.mesh.GetOutputPort())
        geometry.Update()
        self.mesh = geometry

    def unstructured_grid_to_poly_data(self):
        print('transforming UG into PD')
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputConnection(self.mesh.GetOutputPort())
        surface.Update()
        return surface

    # -----MeshInformation------------------------------------------------------------------------------------
    def get_volume(self):
        mass = vtk.vtkMassProperties()
        mass.SetInputConnection(self.mesh.GetOutputPort())
        return mass.GetVolume()

    def print_mesh_information(self):
        _mesh = self.mesh.GetOutput()
        print('Number of vertices: {}'.format(_mesh.GetNumberOfVerts()))
        print('Number of lines: {}'.format(_mesh.GetNumberOfLines()))
        print('Number of strips: {}'.format(_mesh.GetNumberOfStrips()))
        print('Number of polys: {}'.format(_mesh.GetNumberOfPolys()))
        print('Number of cells: {}'.format(_mesh.GetNumberOfCells()))
        print('Number of points: {}'.format(_mesh.GetNumberOfPoints()))

    # -----InputOutput----------------------------------------------------------------------------------------

    # -----Readers--------------------------------------------------------------------------------------------

    def read_vtk(self, to_polydata=False):
        # Read the source file.
        assert os.path.isfile('.' .join([self.filename, self.input_type])), \
            'File {} does not exist!'.format('.' .join([self.filename, self.input_type]))
        reader = vtk.vtkDataReader()
        reader.SetFileName('.' .join([self.filename, self.input_type]))
        reader.Update()
        print('Case ID : {}, input type: {}'.format(self.filename, self.input_type))
        if reader.IsFileUnstructuredGrid():
            print('Reading Unstructured Grid...')
            reader = vtk.vtkUnstructuredGridReader()
        elif reader.IsFilePolyData():
            print('Reading Polygonal Mesh...')
            reader = vtk.vtkPolyDataReader()
        elif reader.IsFileStructuredGrid():
            print('Reading Structured Grid...')
            reader = vtk.vtkStructuredGridReader()
        elif reader.IsFileStructuredPoints():
            print('Reading Structured Points...')
            reader = vtk.vtkStructuredPointsReader()
        elif reader.IsFileRectilinearGrid():
            print('Reading Rectilinear Grid...')
            reader = vtk.vtkRectilinearGridReader()
        else:
            print('Data format unknown...')
        reader.SetFileName(self.filename + '.' + self.input_type)
        reader.Update()  # Needed because of GetScalarRange
        scalar_range = reader.GetOutput().GetScalarRange()
        if to_polydata and not reader.IsFilePolyData():
            print('Transform to Polygonal Mesh')
            reader = self.unstructured_grid_to_poly_data(reader)
        print('Scalar range: \n{}'.format(scalar_range))
        return reader, scalar_range

    def read_vtp(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName('.' .join([self.filename, self.input_type]))
        reader.Update()
        scalar_range = reader.GetOutput().GetScalarRange()
        return reader, scalar_range

    def read_obj(self):
        reader = vtk.vtkOBJReader()
        reader.SetFileName('.' .join([self.filename, self.input_type]))
        reader.Update()
        scalar_range = reader.GetOutput().GetScalarRange()
        return reader, scalar_range
    # ---END-Readers--------------------------------------------------------------------------------------------

    # -----Writers----------------------------------------------------------------------------------------------
    def write_mha(self):

        output_filename = self.filename + '.mha'
        # output_filename_raw = self.filename + '.raw'
        print('writing mha')

        mha_writer = vtk.vtkMetaImageWriter()
        mha_writer.SetInputConnection(self.mesh.GetOutputPort())
        mha_writer.SetFileName(output_filename)
        # mha_writer.SetRAWFileName(output_filename_raw)
        mha_writer.Write()

    def write_stl(self):
        output_filename = self.filename + '.stl'

        # Get surface of the mesh
        print('Extracting surface to save as .STL file...')
        # self.extract_surface()

        # Write file to .stl format
        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetFileName(output_filename)
        stl_writer.SetInputConnection(self.mesh.GetOutputPort())
        stl_writer.Write()
        print('{} written succesfully'.format(output_filename))

    def write_obj(self, postscript=''):
        output_filename = self.filename
        render_window = self.visualize_mesh(False)

        print('Saving PolyData in the OBJ file...')
        obj_writer = vtk.vtkOBJExporter()
        obj_writer.SetRenderWindow(render_window)
        obj_writer.SetFilePrefix(output_filename + postscript)
        obj_writer.Write()
        print('{} written succesfully'.format(output_filename + postscript + '.obj'))

    def write_png(self, postscript=''):

        print('Saving slice in PNG file...')
        output_filename = self.filename + postscript + '.png'
        image = Image.fromarray(self.resample_to_image())
        image = image.convert('L')
        image.save(output_filename, 'PNG')
        print('{} written succesfully'.format(output_filename))

    def write_vtk(self, postscript='_new', type_='PolyData'):
        output_filename = self.filename + postscript + '.vtk'
        writer = None
        if type_ == 'PolyData':
            print('Saving PolyData...')
            self.extract_surface()
            writer = vtk.vtkPolyDataWriter()
        elif type_ == 'UG':
            print('Saving Unstructured Grid...')
            writer = vtk.vtkUnstructuredGridWriter()
        else:
            exit("Select \'Polydata\' or \'UG\' as type of the saved mesh")
        writer.SetInputConnection(self.mesh.GetOutputPort())
        writer.SetFileName(output_filename)
        writer.Update()
        writer.Write()
        print('{} written succesfully'.format(output_filename))

    def write_vtk_points(self, postscript='_points'):
        output_filename = self.filename + postscript + '.vtk'

        point_cloud = vtk.vtkPolyData()
        point_cloud.SetPoints(self.mesh.GetOutput().GetPoints())
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(point_cloud)
        writer.SetFileName(output_filename)
        writer.Update()
        writer.Write()
    # ---END-Writers------------------------------------------------------------------------------------------


def split_chambers(_model, return_as_surface=False, return_elements=True):
    # _model.translate_to_center()

    surfaces = []
    for i in range(1, int(_model.scalar_range[1]) + 1):

        x = _model.threshold(i, i)
        surfaces.append(x)

    full_model_appended = vtk.vtkAppendFilter()
    _model.filename = os.path.join(_model.filename)
    for surf, elem in zip(surfaces, _model.list_of_elements):
        print(elem)
        if return_elements:
            _model.mesh = surf
            _model.extract_surface()
            _model.write_vtk(postscript='_'+elem)

        full_model_appended.AddInputConnection(surf.GetOutputPort())

    full_model_appended.Update()
    _model.mesh = full_model_appended
    if return_as_surface:
        # _model.translate_to_center()
        _model.extract_surface()
        _model.write_vtk(postscript='surf')
    else:
        _model.write_vtk(postscript='tetra')
    return _model


def change_downloaded_files_names(path='h_case_', key='surfmesh', ext='vtk'):
    files = glob.glob(os.path.join(path, '*'+key+'*'+ext))
    for i, old_file in enumerate(files):
        new_file = old_file.split('.')[0]
        print(new_file)
        os.rename(old_file, new_file+'.'+ext)
    print(files)


def change_elem_tag(_mesh, label):
    size = _mesh.GetOutput().GetAttributes(1).GetArray(0).GetSize()
    for i in range(size):
        _mesh.GetOutput().GetAttributes(1).GetArray(0).SetTuple(i, (float(label),))
    return _mesh


def assign_tags(_mesh, label_and_range_tuple=({},)):
    _mesh.GetOutput().GetAttributes(1).GetArray(0).SetName('elemTag')
    _mesh.GetOutput().GetAttributes(0).RemoveArray('elemTag')  # remove point attribute
    for label_and_range in label_and_range_tuple:
        label = label_and_range['label']
        range_of_points = label_and_range['range']
        print('Assigning label {} to {} points'.format(label, range_of_points[1]))
        for id in range(*range_of_points):
            _mesh.GetOutput().GetAttributes(1).GetArray('elemTag').SetTuple(id, (float(label),))
    return _mesh


def merge_elements(elem1, elem2):
    """
    Appends elements and returns the single connected mesh. The points in the same position in 3D are merged into one.
    :param elem1: Single element. The order of the elements pays no role.
    :param elem2: Single element.
    :return: Merged element as filter.
    """
    merger = vtk.vtkAppendFilter()
    merger.MergePointsOn()
    merger.AddInputConnection(elem1.GetOutputPort())
    merger.AddInputConnection(elem2.GetOutputPort())
    merger.Update()
    return merger


# TODO: Make all of the functions and parameters below into a class!!!
# -----ApplyToCohort------------------------------------------------------------------------------------------
def apply_single_transformation_to_all(path, input_base, version, start=0, end=0, ext='_new', ext_type='PolyData',
                                       function_=None, args='()'):
    if function_ is not None:
        if start == end:
            cases = [os.path.join(path, f) for f in os.listdir(path) if f[-4:] == ".vtk"]
        else:
            cases = [path + '/' + input_base + str(case_no).zfill(2) + version + '.vtk' for case_no in
                     range(start, end + 1)]
        print('Cases: {}'.format(cases))
        for case in cases:
            single_model = Model(case)
            print('Executing single_model.' + function_ + args)
            exec('single_model.' + function_ + args)
            if ext is not None:
                single_model.write_vtk(postscript=ext, type_=ext_type)

# ------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    pass
