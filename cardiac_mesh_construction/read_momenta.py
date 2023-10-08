import os
from pathlib import Path
import numpy as np


class HandleMomenta:
    """
    Class for handling momenta file, which is a result of atlas computation of deformetrica deterministic atlas
    algorithm.
    """

    def __init__(
        self,
        momenta_file_path="",
        momenta_filename="",
        output_path="",
        output_filename="",
        momenta_matrix=None,
        configuration="deformetrica",
    ):
        """
        Read the basic information about the data set

        :param momenta_file_path: hard path to the generated momenta file
        """
        self.momenta_file_path = momenta_file_path
        self.output_path = momenta_file_path if output_path == "" else output_path
        self.momenta_file = os.path.join(self.momenta_file_path, momenta_filename)
        self.output_file = os.path.join(self.output_path, output_filename)
        self.configuration = configuration
        self.all_models = None

        if self.configuration == "deformetrica":
            with open(self.momenta_file, "r") as f:
                self.n_models, self.n_control_points, self.n_dimensions = [
                    int(val) for val in f.readline().split(sep=" ")
                ]
        elif self.configuration == "extreme":
            if momenta_matrix is not None:
                self.momenta_matrix = momenta_matrix
            else:
                self.momenta_matrix = np.genfromtxt(self.momenta_file, delimiter=",")
            self.n_models = self.momenta_matrix.shape[0]
            self.n_control_points = int(self.momenta_matrix.shape[1] / 3)
            self.n_dimensions = 3
            self.momenta_matrix = self.build_momenta_matrix()
        else:
            exit('Only "deformetrica" or "extreme" configurations are allowed')

    def save_result(self, output_filename=None, data_to_save=None):
        output_directory = self.output_path
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        np.savetxt(os.path.join(output_directory, output_filename), data_to_save, delimiter=",")

    def transform_momementa_into_table(self):
        """
        The method transforms original file to a table where number of rows is determined by the number of models used
        to calculate the atlas, and columns describe deformation of the template (average shape) in x, y and z
        direction, with regards to control points, effectively defining the warping field for each original model.

        :return: self.all_models - Momenta of all models. The order of values corresponds to dimension: in a row,
        firstly all the x values are given, then all the y, then z.
        """
        self.all_models = np.zeros((self.n_models, self.n_control_points * self.n_dimensions))
        with open(self.momenta_file, "r") as f:
            f.readline()
            f.readline()

            for model_i in range(self.n_models):
                current_model = np.zeros((self.n_control_points, self.n_dimensions))
                for control_point_i in range(self.n_control_points):
                    current_model[control_point_i] = [
                        val for val in f.readline().split(sep=" ")[:3]
                    ]
                self.all_models[model_i] = current_model.ravel(order=1)
                f.readline()

    def build_momenta_matrix(self):
        """
        Each of the row vectors in the momenta matrix must be reshaped individually, to avoid picking values one-by-one.
        The reshaped vectors of momenta are pasted into 'reshaped_momenta_matrix' according to the order of models in
        the original matrix of momenta.

        :return: The reshaped momenta matrix with size dependent on the number of control points and models for rows and
        dimensionality of the models for columns.
        """
        reshaped_momenta_matrix = np.zeros(
            (self.n_control_points * self.n_models, self.n_dimensions)
        )
        for model_number in range(self.n_models):
            reshaped_momenta_matrix[
                model_number * self.n_control_points : (model_number + 1) * self.n_control_points, :
            ] = self.momenta_matrix[model_number].reshape(
                (self.n_control_points, self.n_dimensions), order="F"
            )
        return reshaped_momenta_matrix

    def save_momenta_table(self):
        self.save_result("Momenta_Table.csv", self.all_models)

    def save_momenta_matrix_in_deformetrica_format(self):
        with open(self.output_file, "w") as f:
            f.write("{} {} {}\n\n".format(self.n_models, self.n_control_points, self.n_dimensions))
            for model_number in range(self.n_models):
                single_model_momenta = self.momenta_matrix[
                    model_number
                    * self.n_control_points : (model_number + 1)
                    * self.n_control_points
                ]
                with np.printoptions(threshold=np.inf):
                    f.writelines(str(single_model_momenta).translate({ord(c): None for c in "[]"}))
                f.write("\n\n")


def transform_momenta_before_pca(
    path_to_momenta=os.path.join(
        str(Path.home()),
        "Deformetrica",
        "deterministic_atlas_ct",
        "output_separate_tmp3_def10_prttpe13_corrected",
    ),
    path_to_output=os.path.join(
        str(Path.home()),
        "Deformetrica",
        "deterministic_atlas_ct",
        "output_separate_tmp3_def10_prttpe13_corrected",
        "Decomposition",
    ),
    momenta_file="DeterministicAtlas__EstimatedParameters__Momenta.txt",
    momenta_new="Momenta.txt",
):
    momenta = HandleMomenta(
        path_to_momenta,
        momenta_file,
        output_filename=momenta_new,
        output_path=path_to_output,
        configuration="deformetrica",
    )

    momenta.transform_momementa_into_table()
    momenta.save_momenta_table()


def transform_momenta_after_pca(
    path_to_momenta=os.path.join("DataOutput", "PCA", "Extremes"),
):
    momenta = HandleMomenta(
        momenta_file_path=path_to_momenta,
        momenta_filename="extreme_momenta.csv",
        output_filename="Extreme_Momenta.txt",
        configuration="extreme",
    )
    momenta.save_momenta_matrix_in_deformetrica_format()


def generate_randomly_weighted_modes(
    path_to_momenta=os.path.join("DataOutput", "PCA", "RandomMeshGeneration")
):
    for i in range(1, 41):
        momenta_vec = "Randomly_weighted_modes_{}.csv".format(i)
        momenta_new = "Randomly_weighted_modes_{}.txt".format(i)
        momenta = HandleMomenta(
            path_to_momenta, momenta_vec, output_filename=momenta_new, configuration="extreme"
        )
        momenta.save_momenta_matrix_in_deformetrica_format()


def generate_predefined_weighted_modes(
    path_to_momenta=os.path.join("DataOutput", "PCA", "WeightedModes"),
    output_path=os.path.join("DataInput", "Deformetrica", "Template"),
):
    momenta_vec = "Weighted_momenta.csv"
    momenta_new = "Weighted_momenta.txt"
    momenta = HandleMomenta(
        path_to_momenta,
        momenta_vec,
        output_path=output_path,
        output_filename=momenta_new,
        configuration="extreme",
    )
    momenta.save_momenta_matrix_in_deformetrica_format()


if __name__ == "__main__":
    generate_predefined_weighted_modes()
