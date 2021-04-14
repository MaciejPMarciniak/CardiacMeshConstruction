import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import matplotlib.pyplot as plt
from IO import DataHandler


class PcaWithScaling(DataHandler):

    def __init__(self, dataset_path=None, dataset_filename=None, output_path=None, number_of_components=None,
                 scale_with_std=False):
        super().__init__(dataset_path, dataset_filename, output_path)
        self.dataset_path = dataset_path
        self.dataset_filename = dataset_filename
        self.scale_with_std = scale_with_std
        self.samples = self.X.shape[0]
        self.number_of_components = number_of_components if number_of_components is not None else min(self.X.shape)
        self.transformed_X = None
        self.explained_variance = None
        self.normalized_explained_variance = None
        self.cumulative_variance = None
        self.components = None
        self.mean = None
        self.mode_number = 0
        self.number_of_std = 1
        self.extremes = np.zeros((self.number_of_components*2, self.X.shape[1]))

    def decompose_with_pca(self):
        """
        Principal component analysis run on centered data. Assumes that self.X is in a shape established for
        scikit-learn - a 2D data frame with samples as rows and features as columns.
        """
        scaler = StandardScaler()
        pca = PCA()

        scaled_pca = Pipeline([
            ('scale', scaler),
            ('pca', pca)])
        scaled_pca.set_params(pca__n_components=self.number_of_components,
                              scale__with_std=self.scale_with_std  # False for shape mode visualization, but should be
                              )                                      # true for discriminant or regression analysis.
        self.transformed_X = scaled_pca.fit_transform(self.X)
        self.explained_variance = scaled_pca.named_steps['pca'].explained_variance_
        self.number_of_components = self.explained_variance.shape[0]  # -1 because the last component is irrelevant
        self.normalized_explained_variance = scaled_pca.named_steps['pca'].explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.normalized_explained_variance)
        self.components = scaled_pca.named_steps['pca'].components_
        self.mean = scaled_pca.named_steps['pca'].mean_

    def get_weighted_mode(self, weight=np.array(1)):
        return weight * self.components[self.mode_number, :]

    def get_n_main_modes(self, weights=np.zeros(9)):
        assert 1 < len(weights) <= self.components.shape[0], \
            ('Too many weights provided\n'
             'len(weights): {}, n_components: {})').format(len(weights), self.components.shape[0])
        return np.sum(self.components[:len(weights), :].T * weights, axis=1)

    def get_sds(self):
        return np.sqrt(self.explained_variance)

    def get_random_weights(self, sds, n_sd, distribution='normal'):
        if distribution == 'normal':
            return np.random.normal(0, sds, self.number_of_components)  # Normal distribution
        elif distribution == 'uniform':
            return np.random.uniform(-n_sd * sds, n_sd * sds)  # Uniform distribution
        else:
            exit('Set "normal" or "uniform" distribution of the weights')

    def reconstruct_with_n_modes(self, n_modes=1):
        deformation_vectors = np.zeros((self.samples, self.components.shape[1]))
        print(deformation_vectors)
        for sample in range(self.samples):
            print(sample)
            print(n_modes)
            deformation_vectors[sample, :] = self.get_n_main_modes(self.transformed_X[sample, :n_modes])

        self.save_result('Reconstruction_w_{}_modes.csv'.format(n_modes), deformation_vectors)

    def get_random_combination_of_modes(self, n_samples=1000, n_sd=2):
        deformation_vectors = np.zeros((25, self.components.shape[1]))
        sds = self.get_sds()[:self.number_of_components]
        df_mode_weights = pd.DataFrame(columns=['Mode_{}'.format(i) for i in range(1, self.number_of_components + 1)])
        for sample in range(1, n_samples+1):
            random_weights = self.get_random_weights(sds, n_sd)
            df_mode_weights.loc[(sample - 1) % 25 + 1] = random_weights
            deformation_vectors[sample % 25, :] = self.get_n_main_modes(random_weights)
            if sample % 25 == 0 and sample > 0:
                print('Saving deformation file')
                self.save_result('Randomly_weighted_modes_{}.csv'.format(int(sample/25)), deformation_vectors)
                self.save_dataframe('Cases_weights_{}.csv'.format(int(sample/25)), df_mode_weights)
                df_mode_weights.drop(df_mode_weights.index, inplace=True)

    def get_extremes_of_mode(self, mode_number=0):
        """
        Calculates extreme loadings along a given mode. Useful for statistical shape analysis.

        :param mode_number: The number of the mode of interest.

        :return: 2D numpy array with positive [0, :] and negative [1, :] extreme calculated according to the number of
        standard deviations in the provided mode.
        """
        self.mode_number = mode_number
        sds = self.get_sds()
        weight = self.number_of_std * sds[self.mode_number]

        positive_extreme = self.get_weighted_mode(weight).reshape((1, -1))
        negative_extreme = -positive_extreme
        return np.concatenate((positive_extreme, negative_extreme), axis=0)

    def get_all_extremes(self, number_of_std=1):
        self.number_of_std = number_of_std
        for component in range(self.number_of_components):
            self.extremes[2*component:2*component+2, :] = self.get_extremes_of_mode(component)

    def save_transformed_data(self):
        self.save_result('modes.csv', self.transformed_X)

    def save_extremes(self):
        self.save_result('extreme_momenta.csv', self.extremes)
        # self.save_result(self.dataset_filename, self.extremes)

    def save_eigenvectors(self):
        self.save_result('eigenvectors.csv', self.components)

    def save_explained_variance(self):
        self.save_result('explained_variance.csv', self.explained_variance)
        self.save_result('normalized_explained_variance.csv', self.normalized_explained_variance)

    def save_all_decomposition_results(self):
        self.save_eigenvectors()
        self.save_explained_variance()
        self.save_extremes()
        self.save_transformed_data()


def pca_detailed_atlas_application(path_to_data=os.path.join('DataInput', 'PCA'),
                                   data_filename='Momenta_Table.csv',
                                   components=None,
                                   save_plots=False,
                                   save_pca_results=False):

    momenta_pca = PcaWithScaling(path_to_data, data_filename, number_of_components=components)
    momenta_pca.decompose_with_pca()

    print('Components shape: {}'.format(momenta_pca.components.shape))
    print('Mean Components : {}'.format(np.mean(momenta_pca.components[17, :])))
    print('Std components: {}'.format(np.std(momenta_pca.transformed_X, axis=1)))
    print('Cumulative variance: {}'.format(momenta_pca.cumulative_variance[:18]))
    print('Mean max and min: {}  {}'.format(max(momenta_pca.mean), min(momenta_pca.mean)))
    print('Explained variance: {}'.format(momenta_pca.explained_variance))
    print('Explained variance %: {}'.format(momenta_pca.normalized_explained_variance))
    print('Stds from variance: {}'.format(np.sqrt(momenta_pca.explained_variance)))

    if save_pca_results:
        momenta_pca.save_all_decomposition_results()

    if save_plots:
        plt.bar([*range(len(momenta_pca.normalized_explained_variance))], momenta_pca.normalized_explained_variance)
        plt.plot(momenta_pca.cumulative_variance, 'r.-')
        plt.axhline(0.9)
        plt.savefig(os.path.join('DataOutput', 'Figures', 'Explained_variance.png'))

        for i in range(12):
            plt.scatter(momenta_pca.transformed_X[:, i], momenta_pca.transformed_X[:, i + 1])
            plt.xlabel('Mode {}'.format(i))
            plt.ylabel('Mode {}'.format(i + 1))
            plt.savefig(os.path.join('DataOutput', 'Figures', 'modes_{}_{}.png'.format(i, i + 1)))
            plt.clf()


def pca_get_extremes(path_to_data=os.path.join('DataInput', 'PCA'),
                     data_filename='Momenta_Table.csv',
                     output_path=os.path.join('DataOutput', 'Extremes'),
                     n_standard_deviations=2):

    momenta_pca = PcaWithScaling(path_to_data, data_filename, output_path)
    momenta_pca.decompose_with_pca()
    momenta_pca.get_all_extremes(n_standard_deviations)
    momenta_pca.save_extremes()
    momenta_pca.save_all_decomposition_results()


def pca_get_random_combinations(path_to_data=os.path.join('DataInput', 'PCA'),
                                data_filename='Momenta_Table.csv',
                                output_path=os.path.join('DataOutput', 'RandomMeshGeneration'),
                                n_standard_deviations=2,
                                n_modes=18):

    momenta_pca = PcaWithScaling(path_to_data, data_filename, output_path, number_of_components=n_modes)
    momenta_pca.decompose_with_pca()
    momenta_pca.get_random_combination_of_modes(n_sd=n_standard_deviations)
    momenta_pca.save_all_decomposition_results()


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


if __name__ == "__main__":

    # pca_get_random_combinations()
    pca_get_predefined_combinations()
