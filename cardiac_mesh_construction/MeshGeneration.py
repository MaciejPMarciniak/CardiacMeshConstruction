import os
import glob
from shutil import copyfile, move, rmtree
import subprocess
from cardiac_mesh_construction.Mesh import Model, merge_elements
from pathlib import Path


class MeshTetrahedralization:
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def __init__(
        self, main_path, models_path, geo_dir, temp_dir, output_path, k_model, template=False
    ):
        self.main_path = main_path
        self.models_path = models_path
        self.temp_path = os.path.join(main_path, temp_dir)
        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        self.geo_path = os.path.join(main_path, geo_dir)
        self.output_path = output_path
        self.k_model = k_model
        self.template = template

    def clean(self):
        [f.unlink() for f in Path(self.temp_path).glob("*") if f.is_file()]
        rmtree(self.temp_path)

    def run_tetrahedralization(self):
        subprocess.call(os.path.join(self.main_path, "./meshing.sh"))
        self.clean()

    def copy_surface_mesh_files(self):
        print(self.models_path)

        if not self.template:
            model_files = glob.glob(
                os.path.join(self.models_path, "Shooting_{}_*".format(self.k_model))
            )
        else:
            model_files = glob.glob(os.path.join(self.models_path, "*Template*"))
        print(model_files)
        model_files = [
            mf for mf in model_files if "ControlPoints" not in mf and "Momenta" not in mf
        ]

        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)

        for mf in model_files:
            copyfile(mf, os.path.join(self.temp_path, os.path.basename(mf)))

    def modify_geo_files(self):
        geo_files = glob.glob(os.path.join(self.geo_path, "*.geo"))
        geo_files.sort()
        model_files = glob.glob(os.path.join(self.__location__, self.temp_path, "*"))
        model_files.sort()

        for gf, mf in zip(geo_files, model_files):
            with open(gf, "r") as file:
                data = file.readlines()
            data[1] = 'Merge "' + mf + '";\n'
            with open(gf, "w") as file:
                file.writelines(data)

    # --- Tagging and merging-------------------------------------------------------------------------------------------
    def tag_and_merge_heart_elements(self):
        models = []
        tetra_files = [
            os.path.join(self.main_path, "tetra", el + "_tetra.vtk")
            for el in Model.list_of_elements
        ]

        for element in tetra_files:
            print(os.path.basename(element))
            element_name = os.path.basename(element).split("_")[0]
            print(element_name)
            model = Model(element)
            element_tag = [
                i + 1 for i, elem in enumerate(model.list_of_elements) if elem == element_name
            ][0]
            print("Element name: {}, element tag: {}".format(element_name, element_tag))
            model.build_tag(label=element_tag)
            model.change_tag_label()
            models.append(model)

        final_model = models.pop(0)

        for model_to_merge in models:
            final_model.mesh = merge_elements(final_model.mesh, model_to_merge.mesh)
        final_model.tetrahedralize()
        final_model.write_vtk(postscript="merged", type_="UG")

        if not self.template:
            os.rename(
                os.path.join(self.main_path, "tetra", "LV_tetramerged.vtk"),
                os.path.join(self.main_path, "tetra", "Full_Heart_{}.vtk".format(self.k_model)),
            )
            move(
                os.path.join(self.main_path, "tetra", "Full_Heart_{}.vtk".format(self.k_model)),
                os.path.join(self.output_path, "Full_Heart_{}.vtk".format(self.k_model)),
            )
        else:
            os.rename(
                os.path.join(self.main_path, "tetra", "LV_tetramerged.vtk"),
                os.path.join(self.main_path, "tetra", "Full_Template.vtk"),
            )
            move(
                os.path.join(self.main_path, "tetra", "Full_Template.vtk"),
                os.path.join(self.output_path, "Full_Template.vtk"),
            )

    def tag_and_merge_surf_elements(self):
        models = []
        surf_files = glob.glob(
            os.path.join(self.temp_path, "Shooting_" + str(self.k_model) + "**.vtk")
        )
        surf_files.sort()
        print("surf_files")
        print(surf_files)

        for i, element in enumerate(surf_files):
            file_ = os.path.basename(element)
            new_element_path = os.path.join(
                os.path.dirname(element), file_[27:32].strip("_t") + "_" + file_[9:11] + file_[-4:]
            )
            os.rename(element, new_element_path)
            element_name = os.path.basename(new_element_path).split("_")[0]
            model = Model(new_element_path)
            element_tag = [
                j + 1 for j, elem in enumerate(model.list_of_elements) if elem == element_name
            ][0]
            print(
                "Element file: {}, Element name: {}, element tag: {}".format(
                    os.path.basename(new_element_path), element_name, element_tag
                )
            )
            model.build_tag(label=element_tag)
            model.change_tag_label()
            models.append(model)

        final_model = models.pop(0)

        for model_to_merge in models:
            final_model.mesh = merge_elements(final_model.mesh, model_to_merge.mesh)

        final_model.write_vtk(postscript="not_tetra", type_="PolyData")

        os.rename(
            glob.glob(os.path.join(self.temp_path, "*not_tetra.vtk"))[0],
            os.path.join(self.temp_path, "Full_Heart_{}.vtk".format(self.k_model)),
        )
        move(
            os.path.join(self.temp_path, "Full_Heart_{}.vtk".format(self.k_model)),
            os.path.join(self.output_path, "Full_Heart_{}.vtk".format(self.k_model)),
        )

    # --- Pipelines-----------------------------------------------------------------------------------------------------
    def pipeline_surf_2_tetra_mesh(self):
        self.copy_surface_mesh_files()
        self.modify_geo_files()
        self.run_tetrahedralization()
        self.tag_and_merge_heart_elements()

    def pipeline_aggr_surf_mesh(self):
        self.copy_surface_mesh_files()
        self.tag_and_merge_surf_elements()


# --- END MeshTetrahedralization----------------------------------------------------------------------------------------


def random_dataset_generation(cohort_from, cohort_to, id_from, id_to):
    for i in range(cohort_from, cohort_to):
        # Path to models, surface or volumetric
        models_path = (
            "/media/mat/BEDC-845B/" "Surface_meshes/output_shooting_{}/final_steps"
        ).format(i)
        output_path = "/media/mat/BEDC-845B/Final_models_{}".format(str(i).zfill(2))  # Storing path

        merged_shapes_generation(id_from, id_to, models_path, output_path, merge_type="tetra")


if __name__ == "__main__":
    random_dataset_generation(1, 41, 0, 25)
