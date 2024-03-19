#!/bin/python3
import os
from time import sleep

import numpy as np

from .resting_state import RestingState

##########


class DevelopmentalRS(RestingState):
    """
    About DevelopmentalRS

    This is a developmental class that shouldn't be
    used in production. It's fun to try new things!
    """

    def __init__(self, sub_id, task, bids_root, suppress, template_space):

        RestingState.__init__(
            self,
            task=task,
            bids_root=bids_root,
            suppress=suppress,
            template_space=template_space,
        )

    def _matrix_from_labels_masker(
        self,
        run="ALL",
        atlas_to_use=None,
        labels_to_use=None,
        standardize=True,
        save_matrix_output=False,
        save_plots=False,
        show_plots=True,
        return_array=True,
        verbose=True,
    ):
        """
        This function applies when data from non-overlapping volumes should be extracted

        Parameters
              run:  str or int | Functional run from BIDS project; defaults to ALL runs (concatenated matrix)
              atlas_to_use:  Relative path to NifTi mask (defaults to native MSDL atlas)
              labels_to_use:  List of labels to feed to the correlation matrix
              standardize:  Boolean | if True, the extracted signal is z-transformed
              save_matrix_output:  Boolean | if True, correlation matrix is saved to subject folder
              save_plots:  Boolean | if True, correlation matrix and connectome is saved locally
              show_plots:  Boolean | if True, plots are displayed to the console
              return_array:  Boolean | if True, array is returned and can be assigned to variable
              verbose:  Boolean | if True, method mechanics are printed to the console

        Returns
              If return_array == True, returns correlation matrix
        """

        from nilearn.connectome import ConnectivityMeasure
        from nilearn.maskers import NiftiLabelsMasker

        # ==== Validate method attributes ====
        # -- Check reference atlas validity
        if atlas_to_use is None:
            atlas_maps, atlas_labels = self.pull_msdl_atlas()

        else:
            if labels_to_use is None:
                raise ValueError(
                    "NOTE: If supplying your own atlas, you MUST provide list of labels"
                )

            atlas_maps = atlas_to_use
            atlas_labels = labels_to_use

        # -- Load subject BOLD and confound data
        if run == "ALL":
            bold_run = [x for x in self.bids_container["all_preprocessed_bold"]]

        else:
            bold_run = self.bids_container[f"run-{run}"]["preprocessed_bold"]
            confound = self.load_confounds(run=run).loc[
                :, self.confound_regressor_names
            ]

        method = f"""
Running sub-{self.sub_id}_task-{self.task}_run-{run}...\n\n
Reference atlas:\t{atlas_maps}\n
BOLD run:\t\t{bold_run}\n
Confounds:\t\t{confound}
            """

        # -- If verbose, we'll print to the console as the method runs
        if verbose:
            value = 5
            print(method)
            sleep(2)
        else:
            value = 0

        # ==== Run labels masker ====
        masker = NiftiLabelsMasker(
            labels_img=atlas_maps, standardize=standardize, verbose=value
        )

        time_series = masker.fit_transform(bold_run, confounds=confound)

        # ==== Run correlation transformer ====
        correlation_transformer = ConnectivityMeasure(kind="correlation")
        correlation_matrix = correlation_transformer.fit_transform([time_series])[0]

        # ==== Apply both plotting functions ====
        self.plot_correlation_matrix(
            correlation_matrix,
            labels=atlas_labels,
            run=run,
            save_local=save_plots,
            suppress_plot=show_plots,
        )

        self.plot_connectomes(
            matrix=correlation_matrix,
            run=run,
            atlas_map=atlas_maps,
            save_local=save_plots,
            show_plot=show_plots,
        )

        # Save matrix locally if user desires
        if save_matrix_output:

            filename = (
                f"sub-{self.sub_id}_task-{self.task}_run-{run}_labels-masker-matrix.npy"
            )

            correlation_matrix.tofile(
                os.path.join(self.first_level_output, "models", filename)
            )

        if return_array:
            return correlation_matrix

    def _matrix_from_maps_masker(
        self,
        run,
        atlas_to_use=None,
        labels_to_use=None,
        standardize=True,
        save_matrix_output=False,
        save_plots=False,
        show_plots=True,
        return_array=True,
        verbose=True,
    ):
        """
        This function applies when data from overlapping volumes should be extracted

        Parameters
              run:  str or int | Functional run from BIDS project
              atlas_to_use:  Relative path to NifTi mask (defaults to native MSDL atlas)
              labels_to_use:  List of labels to feed to the correlation matrix
              standardize:  Boolean | if True, the extracted signal is z-transformed
              save_matrix_output:  Boolean | if True, correlation matrix is saved to subject folder
              save_plots:  Boolean | if True, correlation matrix and connectome is saved locally
              show_plots:  Boolean | if True, plots are displayed to the console
              return_array:  Boolean | if True, array is returned and can be assigned to variable
              verbose:  Boolean | if True, method mechanics are printed to the console

        Returns
              If return_array == True, returns correlation matrix
        """

        from nilearn.connectome import ConnectivityMeasure
        from nilearn.maskers import NiftiMapsMasker

        # ==== Validate method attributes ====
        # -- Check reference atlas validity
        if atlas_to_use is None:
            print("NOTE: Defaulting to MSDL Atlas")
            atlas_maps, atlas_labels = self.pull_msdl_atlas()

        else:
            if labels_to_use is None:
                raise ValueError(
                    "NOTE: If supplying your own atlas, you MUST provide list of labels"
                )

            atlas_maps = atlas_to_use
            atlas_labels = labels_to_use

        # -- Load subject BOLD and confound data
        bold_run = self.bids_container[f"run-{run}"]["preprocessed_bold"]
        confound = self.load_confounds(run=run).loc[:, self.confound_regressor_names]

        method = f"""
Running sub-{self.sub_id}_task-{self.task}_run-{run}...\n\n
Reference atlas\n{atlas_maps}\n\n
BOLD run\n{bold_run}\n\n
Confounds\n{self.bids_container[f"run-{run}"]["confounds"]}\n\n
            """

        # -- If verbose, we'll print to the console as the method runs
        if verbose:
            value = 5
            print(method)
            sleep(2)

        else:
            value = 0

        # ==== Run maps masker ====
        masker = NiftiMapsMasker(
            maps_img=atlas_maps, standardize=standardize, verbose=value
        )

        time_series = masker.fit_transform(bold_run, confounds=confound)

        # ==== Run correlation transformer ====
        correlation_transformer = ConnectivityMeasure(kind="correlation")
        correlation_matrix = correlation_transformer.fit_transform([time_series])[0]

        # ==== Apply both plotting functions ====
        self.plot_correlation_matrix(
            correlation_matrix,
            labels=atlas_labels,
            run=run,
            save_local=save_plots,
            show_plot=show_plots,
        )

        self.plot_connectomes(
            correlation_matrix,
            run=run,
            atlas_map=atlas_maps,
            save_local=save_plots,
            show_plot=show_plots,
        )

        # Save matrix locally if user desires
        if save_matrix_output:

            filename = (
                f"sub-{self.sub_id}_task-{self.task}_run-{run}_maps-masker-matrix.npy"
            )

            correlation_matrix.tofile(
                os.path.join(self.first_level_output, "models", filename)
            )

        if return_array:
            return correlation_matrix

    # -- Time series extraction
    def _extract_all_time_series(
        self,
        method="maps",
        atlas_to_use=None,
        labels_to_use=None,
        standardize=True,
        show_plots=False,
        save_plots=True,
        save_each_array=True,
        verbose=False,
    ):
        """
        DEVELOPMENTAL FUNCTION

        This function loops through all resting state BOLD
        runs and creates a connectivity matrix for each run

        Parameters
              method: str | "maps" or "labels", determines masker type
              atlas_to_use: NifTi file or path to NifTi file, reference atlas
              labels_to_use: list | Functional areas corresponding to reference atlas
              standardize: Boolean | if True, time series are z-transformed
              show_plots: Boolean | if True, plots are printed to the console
              save_plots: Boolean | if True, run-wise plots are saved locally
              save_each_array: Boolean | if True, run-wise matricies are saved locally
              verbose: Boolean | if True, processes are printed to the console

        Returns
              List of fun-wise connectivity matrices
        """

        method = method.lower().strip()

        if method not in ["maps", "labels"]:
            raise ValueError(
                f"ERROR: {method} is invalid input ... valid: ['maps', 'labels']"
            )

        # ==== Run Masker helper method ====
        container = []

        for ix in range(len(self.preprocessed_runs)):

            run = ix + 1

            temp_matrix = self.matrix_from_masker(
                run=run,
                atlas_to_use=atlas_to_use,
                method=method,
                labels_to_use=labels_to_use,
                standardize=standardize,
                show_plots=show_plots,
                save_plots=save_plots,
                verbose=verbose,
                save_matrix_output=save_each_array,
            )

            container.append(temp_matrix)

        return container

    def _weigh_average_timeseries(
        self,
        extracted_timeseries=[],
        atlas_to_use=None,
        labels_to_use=None,
        show_plots=False,
        save_plots=True,
        save_matrix=True,
    ):
        """
        DEVELOPMENTAL FUNCTION

        This function aggregates all BOLD connectivity matrices into a single
        index of a subject's functional connectivity at reast

        Parameters
              extracted_timeseries: list | List of connectivity matricies from each BOLD run
              atlas_to_use: NifTi file or path to NifTi file to use as reference atlas
              labels_to_use: list | Functional areas corresponding to reference atlas
              show_plots: Boolean | if True, plots are displayed to the console
              save_plots: Boolean | if True, plots are saved to participant output directory
              save_matrix: Boolean | if True, each BOLD connectivity matrix is saved locally

        Returns
              Average functional connectivity matrix
        """

        # ==== Validate user input ====
        if atlas_to_use is None:
            print("Defaulting to MSDL atlas")
            atlas_to_use, labels_to_use = self.pull_msdl_atlas()

        # Weigh average functional connectivity matrix
        average_matrix = np.mean(np.array(extracted_timeseries), axis=0)

        # Run plotting functions
        self.plot_correlation_matrix(
            average_matrix,
            labels=labels_to_use,
            run="ALL",
            save_local=save_plots,
            show_plot=show_plots,
        )

        self.plot_connectomes(
            average_matrix,
            run="ALL",
            atlas_map=atlas_to_use,
            save_local=save_plots,
            show_plot=show_plots,
        )

        # Save matrix locally
        if save_matrix:
            filename = f"sub-{self.sub_id}_task-{self.task}_averaged-matrix.npy"

            average_matrix.tofile(
                os.path.join(self.first_level_output, "models", filename)
            )

        return average_matrix

    def _run_weighted_timeseries(
        self,
        method="maps",
        atlas_to_use=None,
        labels_to_use=None,
        standardize=True,
        show_plots=False,
        save_plots=True,
        save_each_array=True,
        verbose=False,
        return_matrix=True,
    ):
        """
        DEVELOPMENTAL FUNCTION

        This function combines the wrapping functions above. In practice, it extacts a
        time series for each BOLD run, creates a connectivity matrix for each run, and
        aggregates a weighted average as an index of the subject's overall functional
        connectivity at rest

        Parameters
              method: str | maps or labels, determines which masker will be used
              atlas_to_use: NifTi file or path to NifTi file to use as reference atlas
              labels_to_use: list | Functional areas corresponding to reference atlas
              standardize: Boolean | if True, time series is z-transformed
              show_plots: Boolean | if True, plots are displayed to the console
              save_plots: Boolean | if True, plots are saved to participant output directory
              save_each_array: Boolean | if True, a matrix is saved per run
              verbose: Boolean | if True, processes are printed to the console
              return_matrix: Boolean | if True, the weighted matrix returned

        Returns
              If return_matrix, the weighted matrix is returned
        """

        # ==== Validate user input ====
        if atlas_to_use is None:
            print("Defaulting to MSDL atlas...")
            atlas_to_use, labels_to_use = self.pull_msdl_atlas()

        if method not in ["maps", "labels"]:
            raise ValueError(f"{method} is invalid input ... valid: ['maps', 'labels']")

        # ==== Extract run-wise timeseries ====
        if verbose:
            print("Extracting run-wise timeseries...")
            sleep(1)

        aggregate_timeseries = self.extract_all_time_series(
            method=method,
            atlas_to_use=atlas_to_use,
            labels_to_use=labels_to_use,
            standardize=standardize,
            show_plots=show_plots,
            save_plots=save_plots,
            save_each_array=save_each_array,
            verbose=verbose,
        )

        # ==== Weigh individual timeseries ====
        if verbose:
            print("Aggregating connectivity matrices...")
            sleep(1)

        weighted_connectomes = self.weigh_average_timeseries(
            aggregate_timeseries,
            atlas_to_use=atlas_to_use,
            labels_to_use=labels_to_use,
            show_plots=show_plots,
            save_plots=save_plots,
            save_matrix=save_each_array,
        )

        if return_matrix:
            return weighted_connectomes
