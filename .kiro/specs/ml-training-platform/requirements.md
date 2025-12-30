## Overview
Machine Learning Model Training Platform - a web application developed to allow anyone to create machine learning models without needing to write code. The platform enables users to upload their own datasets and choose algorithms with specific hyperparameters, build and train those algorithms, and evaluate the performance of their models. The primary audience for the ML Model Training Platform is students, new ML users (beginners), and hackathon teams that are leveraging ML capabilities and providing the ability for users to reproduce the results generated from their models.

## Requirements

### Requirement 1: Dataset Management
**User Story:** As a user of this platform, I would like to be able to upload a CSV file containing my dataset so I may train a model using the datasets I create

#### Acceptance Criteria

1. Once users have uploaded their CSV file, validation of the format and structure of the file will occur.
2. If the CSV file passes validation, the platform will display the headers and a sample of the data contained within that file in preview mode.
3. If the uploaded CSV file fails validation, the platform will notify the user of the error and will not allow the user to continue with any processing on that file.
4. When a user has uploaded a data set, the platform will automatically determine the fundamental type of data in each column (e.g., numeric, categorical, text).

### Requirement 2: ML Task Configuration

**User Story:** As a user, I can classify and/ or regress and/or cluster using the appropriate ML algo/method by selecting from pre-defined ML Task Types and/or Algos.

#### Acceptance Criteria

1. The Platform will allow for selecting multiple different types of Classification, Regression, and Clustering Tasks.
2. When you select an ML Task Type, the Platform will display the ML Algos associated with that Task Type.
3. Classification Tasks: The Platform will provide the following Classification Algorithms Random Forest, Support Vector Machines, and Logistic Regression.
4. Regression Tasks: The Platform will provide the following Regression Algorithms: Linear Regression, Random Forest Regressor, and Support Vector Regressor.
5. Clustering Tasks: The Platform will provide the following Clustering Algorithms: K-Means, DBSCAN, and Hierarchical clustering.

### Requirement 3: Model Training

**User Story:** As a user, I would like to be able to train models with my own configured settings in order to be able to utilize them for previewing/model generation of my own data.

#### Acceptance Criteria
1. When a user starts training, The Platform will split the datasets between Training Set and Test set.
2. When training begins, The Platform will provide an indicator of the progress of the training along with an estimate of when the training will be finished.
3. The Platform will build the Model based on the algorithm and hyper parameters selected by the user.
4. When the training has finished, The Platform will generate and display appropriate Metrics.
5. The Platform will create a trained model and be made available for the user to download and to use for performance comparisons between other techniques/models.
6. If the training process fails, The Platform should display a complete error message that provides suggestions on how to resolve the issue.

### Requirement 4: Model Comparison

**User Story:** As the user, I would like to find out which trained models perform best compared to other trained models visually, allowing me to choose which model(s) I believe would best suit my needs.

#### Acceptance Criteria
The Platform must keep track of all trained models created during one session, and if there is more than one model available, the Platform must have a comparison table with all models listed.
Additionally, the Platform must generate visual charts to present each trained model's metrics/ performance comparisons against the other models.
Moreover, The Platform must allow users to select their models of interest for the purpose of generating a detailed comparison of all aspects related to the selected model(s).
Lastly, the Platform should provide a colour-coded method of highlighting which trained models performed the best according to primary metrics.

### Requirement 5: Model Export

**User Story:** As an end-user, I want to be able to download my trained model(s), so that I can utilize them in external software solutions.

#### Acceptance Criteria

1. AFTER training is complete, THE Platform MUST allow users to download a trained model.
2. TRAINED Models MUST be exported as a Pickle File (.pkl)
3. METADATA associated with the TRAINED Model (algorithm type(s), hyperparameter values and dates) MUST be included in the downloadable FILE.
4. DOWNLOADABLE MODELS MUST contain a unique filename.
5. PRIOR to downloading, THE Platform MUST verify the INTEGRITY of the model.

### Requirement 6: Reproducible Training

**User Story:** As a user, I would like to have training results that are reproducible. This enables me to recreate the identical model using the same configuration.

## Acceptance Criteria
1. THE Platform USES consistent random seeds across all training actions.
2. WHEN identical sets of data and configuration elements ARE applied, identical results WILL BE produced by THE Platform.
3. THE Platform SHOULD log ALL training parameters and configuration information for EVERY model that is trained.
4. THE Platform WILL HAVE the ability to EXPORT training configuration information for FUTURE use.
5. THE Platform WILL KEEP VERSION information for algorithms and dependencies.

### Requirement 7: User Interface Simplicity

**User Story:** As someone starting out with using ML tools, I want an easy-to-understand platform so I'll be able to easily use ML tools without extensive background knowledge.

### Acceptance Criteria
1. THE Platform provides a user interface with a wizard-style navigation to guide users through each stage of setting up their machine learning project.
2. THE Platform uses nontechnical, easy to read English words in all aspects of the user interface.
3. THE Platform provides tutorials in context and definitions of the ML terms along the way.
4. THE Platform reduces the amount of information the user must enter before running the ML solution.
5. THE Platform will automatically fill out the most reasonable values for users for each of the possible configuration options.
