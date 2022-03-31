# GLM-Express

<img src="./img/GLM_Express_cardinal.png" width=65%>

This is a package for modeling functional neuroimaging tasks. As the name implies, it's optimized to be simple and straightforward! The `task_information.json` file stores all of the regressors and modeling specifications for each task; modifying this file allows you to test a range of analytical outcomes.

## Currently

This package comes equipped with two modeling objects:

* `Subject` is a first-level modeler for subject-specific functional neuroimaging data
  
  * Click <a href="https://github.com/IanRFerguson/glm-express/blob/main/demo/Subject_Demo.pdf" target="_blank">**here**</a> for a demo of the `Subject` object

* `GroupLevel` is a second-level modeler that is optimized to aggregate contrast maps derived by the `Subject` object

  * Click **here** for a demo of the `GroupLevel` object

## In Development

A `RestingState` object is currently in the works, for analyses of subject-level resting state functional connectivity


## Assumptions

We assume the following about your data:

* Your data is in valid `BIDS` format     <br> 
* Your data has been preprocessed by `fmriprep` 
  * These preprocessed data are stored in a `derivatives` folder nested in your `BIDS` project <br>
* You have adequate events TSV files for all of your functional tasks <br>
* Any parametric modulators are stored within each event file <br>
  * Otherwise, you can build custom design matrices and feed them into the modeling function


## About `task_information.json`

Glossary of keys in the `task_information` file; manipulating these allow you to quickly and effectively customize your modeling parameters without editing any source code

* `tr`: Repetition time (defined here, but can be overriden for any one subject)
* `excludes`: Subjects in your project you need to exclude for a given task
* `condition_identifier`: Column in your events file that denotes trial type; **NOTE** this will be changed to `trial_type` in the script
* `confound_regressors`: Regressors to include from `fmriprep` output
* `modulators`: Parametric modulators to weight trial type (these should be in your events file)
* `block_identifier`: Column in your events file that denotes block type; defaults to `null`
* `design_contrasts`: Your defined contrasts! Include as few or as many as you see fit