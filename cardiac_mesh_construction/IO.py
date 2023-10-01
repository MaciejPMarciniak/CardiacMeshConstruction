import numpy as np
import os


class DataHandler:

    def __init__(self, covariates_data_path=None, covariates_data_filename=None, output_path=None,
                 reference_data_path=None, X=None, y=None):
        self.covariates_data_path = covariates_data_path
        self.output_path = output_path
        if X is not None:
            self.X = X
        else:
            self.X = np.genfromtxt(os.path.join(covariates_data_path, covariates_data_filename), delimiter=',', )

        if y is not None:
            self.y = y
            assert self.X.shape[0] == self.y.shape[0], ('The number of samples in covariate data and reference data is '
                                                        'not the same. Check the input data.')
        elif reference_data_path is not None:
            print(reference_data_path)
            # TODO: Allow for response data to be packed in the same file as the covariates
            self.y = np.genfromtxt(covariates_data_path, delimiter=',')
            assert self.X.shape[0] == self.y.shape[0], ('The number of samples in covariate data and reference data is '
                                                        'not the same. Check the input data.')

    def save_result(self, output_filename=None, data_to_save=None):
        output_directory = self.output_path
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        np.savetxt(os.path.join(output_directory, output_filename), data_to_save, delimiter=',')

    def save_dataframe(self, output_filename=None, dataframe=None):
        output_directory = self.output_path
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        dataframe.to_csv(os.path.join(output_directory, output_filename))
