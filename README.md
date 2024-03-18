# MDS_288R_capstone
Ojective: Create a high-quality commercial-size music dataset for data science analysis and ML. Apply predictive classifiers to find patterns in genre, sound features, and year-released to further our understanding of music. 

Description:  Human understanding and recognition of music is thought to be as inherent as language. The field of music theory has long tried to characterize these patterns, but statistics has rarely been used to show relationships, patterns, and evolution. Therefore, applying statistics and machine learning could provide novel insights that can be applied to furthering the study of music and building more sophisticated music recommendation systems through these patterns. However, studying music with machine learning requires large datasets, and direct audio files are often protected by copyright laws. Additionally, the computational resources necessary to use direct audio files are tremendous. Therefore, there is a need for music data with features of genre, artist, and year released to further the open-source study of music and utilization of its discovered patterns. To answer this need, our project aims to create a high-quality music dataset from the sources AcousticBrainz and MusicBrainz. Through extensive cleaning, our team was able to compile two datasets of nearly 300,000 rows each: one for song release year prediction and one for genre and scale classification. Proof of this quality was our ability to apply classifiers to genre and scale, achieving an 85.22% and 85.21% accuracy respectively using random forest (RF) models. Less successfully (Year stuff here) . Lastly, our extraction and preprocessing pipelines have the potential to be applied to over 550GBs of music data available through AcousticBrainz and MetaBrainz

File Descriptions
Project is split by utilizing two subdatasets: one for year prediction; and one for genre&scale prediction.

data: 
genre_and_scale_data:
    processed_genre_numeric_data: processed data for machine learning
    processed_nonnumeric_metadata.csv: metasata for "processed_genre_numeric_data.csv" inclusing information MusicBrainzID(ID), artist, track, and album
    test_indices_RS_42.csv: indices for test set with random_state set for 42. This is used test all scale and genre models
    train_indices_RS_42.csv: indices for train set with random_state 42. This is used to train all scale and genre models

year_prediction_data:

models:
    build: building models
        bu_genre_classifiers: py files to build genre classification models. Accuracy and model type is noted in model name.
        bu_scale_classifiers: py files to build scale classification model. Model type is noted in name.
    
    validate: validating models
        val_genre_classifiers: py files and joblib files to validate test accuracy for genre models
        val_scale_classifiers: py files and joblib files to validate test accuracy for scale models

Preprocessing_and_EDA:
        genre_and_scale_preprocessing: preprocessing piplines for genre and scale
        year_prediction: preprocessing pipelines for year