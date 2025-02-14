# project-bisg
Evaluation of algorithmic fairness via proxy estimation.

## Preliminary Task

### Requirements

1. Download North Carolina voter registration database available here: https://www.ncsbe.gov/results-data/voter-registration-data
2. Using the BISG implementation available here: https://surgeo.readthedocs.io/en/dev/
3. And the “weighted estimator” as described in this paper: https://arxiv.org/pdf/1811.11154

### Task

Write code (in python preferably) to approximate the racial composition of each political party (DEM, REP, LIB, IND) using the weighted estimator and the BISG implementation as your proxy predictor. Do this for a county of your choosing. Also chose some appropriate visualization to show the error of your estimates and the true race proportions. Some things to keep in mind:

1. You will need to do a little bit of data processing of the North Carolina voter registration dataset. Make sure that the code you write to do this is well-documented and easy to follow
2. I would recommend wrapping the BISG library in a custom class since we will be implementing many other methods for prediction by proxy. Try writing a “ProxyPredictor” interface that contains an “inference” method
3. Your subclass’s implementation of the “inference” method should take as input a pandas data frame, and should output a pandas data frame with race predictions
4. Note: this method will not be complicated for this example, and should just interface the functionality of Surgeo (the BISG library) with the codebase that you are developing

## Questions to Confirm
1. How to preprocess ZTAC and Surname? Need to normalize names and ztacs as Surgeo base model provides?
2. Political party information only used in analysis stage when approximating the racial composition of each political party (DEM, REP, LIB, IND)?
3. Need to preprocess race to align with predicted results? Then do we need to normalize the predicted probabilities? How to handle races beyond the 6 Surgeo race categories?
4. Weighted estimator uses voter info?