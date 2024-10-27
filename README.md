# Speech-Emotion-Recognition-using-Wave2vec-base-960-h


**ABSTRACT**
Speech Emotion Recognition (SER) has become a prominent field in Human-Computer Interaction (HCI) as it enables machines to recognize and interpret human emotions from audio signals. In this research, we propose an efficient SER system using Wav2Vec2, a pre-trained model developed by Facebook AI for self-supervised speech representation learning. This paper presents the implementation and evaluation of SER using Wav2Vec2 with the RAVDESS dataset, classifying six distinct emotions. By leveraging transfer learning, we fine-tuned the Wav2Vec2 model for emotion classification, achieving a test accuracy of 90.57%. This paper discusses the dataset, model architecture, training process, evaluation metrics, and results, along with comparisons to other state-of-the-art methods in speech emotion recognition.
**INTRODUCTION**
1.1 **BACKGROUND**
Speech Emotion Recognition (SER) is a vital area in the field of human-computer interaction, enabling machines to interpret and respond to human emotions conveyed through speech. It has applications in customer service, mental health monitoring, education, and more. Recognizing emotions from speech signals is challenging due to the variability in speech patterns, accents, and emotional expressions across different individuals.
1.2 **MOTIVATION**
Traditional SER systems often rely on handcrafted features and require extensive domain expertise for feature extraction. With the advent of deep learning, models can learn features directly from raw data. Recently, self-supervised learning models like Wav2Vec2 have shown promise in learning robust representations from unlabeled data, which can be fine-tuned for specific tasks with smaller labelled datasets. Leveraging Wav2Vec2 for SER can potentially improve accuracy and reduce the need for manual feature engineering.
1.3 **PROBLEM STATEMENT**
The primary challenge addressed in this research is developing an effective SER model that can accurately classify emotions from raw audio data with minimal preprocessing. The goal is to utilise the Wav2Vec2 model to classify six emotions from the RAVDESS dataset and achieve high accuracy, demonstrating the model's capability in SER tasks.
1.4 **Objectives**
To implement a Speech Emotion Recognition system using the pre-trained Wav2Vec2 model.
To fine-tune the model on the RAVDESS dataset for classifying six emotions.
To evaluate the model's performance using appropriate metrics and analyse the results.
To identify challenges faced during implementation and suggest future improvements.


**LITERATURE REVIEW**
2.1 **TRADITIONAL APPROACHES TO SPEECH EMOTION RECOGNITION** 
Traditional SER methods rely on extracting handcrafted features from audio signals, such as Mel-Frequency Cepstral Coefficients (MFCCs), pitch, energy, and formants. These features capture the spectral and prosodic characteristics of speech. Classical machine learning algorithms like Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), and Hidden Markov Models (HMM) are then used for classification. While these methods have been effective to some extent, they often require significant domain expertise for feature extraction and may not generalise well across different datasets.
2.2 **DEEP LEARNING MODELS FOR SER**
Deep learning has revolutionised SER by enabling models to learn hierarchical representations directly from data. Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) have been employed to capture spatial and temporal features in speech signals. These models reduce the need for manual feature extraction but typically require large labelled datasets for training.
2.3 Wav2Vec2 AND SELF SUPERVISED LEARNING IN SER
Wav2Vec2 is a self-supervised learning model developed by Facebook AI. It learns powerful representations from raw audio by pre-training on large amounts of unlabeled data and can be fine-tuned with a smaller labelled dataset for specific tasks like SER. This approach leverages the vast amount of available unlabeled speech data, reducing the dependency on large labelled datasets and manual feature engineering.
2.4 **GAPS IN EXISTING RESEARCH**
While Wav2Vec2 has shown excellent performance in Automatic Speech Recognition (ASR), its application to Speech Emotion Recognition is still an emerging area. There is a need to explore how well Wav2Vec2 can be fine-tuned for emotion recognition tasks and how it compares with traditional and other deep learning models in terms of accuracy and generalisation.


**METHODOLOGY**
3.1 **DATASET DESCRIPTION (RAVDESS)**
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is used for this research. It contains  recordings from 24 professional actors (12 male, 12 female), expressing two lexically-matched statements across eight emotions. For this study, six emotions are selected:
Neutral(144 files)
Happy(144 files)
Sad(144 files)
Angry(144 files)
Fear(144 files)
Disgust(144 files)
Each emotion has an equal number of samples, ensuring a balanced dataset.The dataset is structured in a systematic manner, with subdirectories allocated to each emotional category. This organisation allows for straightforward access to the audio files and facilitates the efficient processing required for training the model. The audio recordings are in WAV format, sampled at 48 kHz, which is conducive to capturing the nuances of human speech.
3.2 **Wav2Vec2 MODEL ARCHITECTURE**
The model utilised for the speech emotion recognition task is Wav2Vec2ForSequenceClassification from the Hugging Face Transformers library, specifically the pre-trained  model "facebook/wav2vec2-base-960h." This architecture is designed to effectively process and classify audio data based on emotion labels.
Key Components of the Architecture:
Feature Encoder:This component is based on a convolutional neural network (CNN) that processes raw audio waveforms into latent speech representations. In the implementation, audio files are loaded using the librosa library, resampled to 16kHz, and then transformed into input features suitable for the Wav2Vec2 model. This preprocessing step ensures that the input format is compatible with the model’s expected requirements.
Transformer Encoder:The transformer encoder captures contextual information across the sequence of audio features. This allows the model to leverage long-range dependencies within the audio input, which is crucial for distinguishing between different emotional states. The model is fine-tuned to classify six different emotions: neutral, happy, sad, angry, fear, and disgust, each represented by unique labels processed through a LabelEncoder.
Classification Head:The classification head consists of a fully connected layer added on top of the transformer encoder for the classification tasks. In this implementation, the classification layer is initialised with the number of labels corresponding to the unique emotions in the dataset. The model is trained using cross-entropy loss, optimising the parameters through the Adam optimizer with a learning rate of 1e-5.
3.3 **PREPROCESSING OF AUDIO DATA**
Effective preprocessing of audio data is crucial for the success of the speech emotion recognition model. The following steps outline the detailed preprocessing pipeline employed in this project:
Audio Loading:
The audio files are loaded using the librosa library, which is a powerful tool for audio and music analysis. When loading the audio files, the librosa.load() function is used to read each .wav file into a time-series representation. To ensure consistency and compatibility with the Wav2Vec2 model, all audio signals are resampled to a uniform sampling rate of 16,000 Hz. This resampling is essential as the model has been pre trained on audio data with this specific sampling rate. The resampling step not only preserves the quality of the audio but also facilitates effective feature extraction by the model's convolutional layers.
Label Encoding:
The emotions represented in the dataset are categorical variables, which need to be converted into a numerical format to be processed by the model. This conversion is achieved using the LabelEncoder from the scikit-learn library. Each unique emotion label (e.g., neutral, happy, sad, angry, fear, disgust) is transformed into a distinct integer label. This encoding process helps in organising the output layer of the model, where each integer corresponds to a specific emotion class.
Data Splitting:
The dataset is divided into training and testing subsets to evaluate the model's performance objectively. The train_test_split function from scikit-learn is employed for this purpose. A fixed random state is set (e.g., random_state=42) to ensure reproducibility across different runs of the model. In this implementation, 90% of the dataset is allocated for training, while 10% is reserved for testing. This stratified split ensures that the distribution of emotions is preserved in both the training and testing sets, which is vital for a balanced evaluation of the model's performance.
3.4 **MODEL FINE TUNING**
Fine-tuning the Wav2Vec2 model for speech emotion recognition involves careful design and implementation of the dataset class, collate function, and training parameters. The following details outline these components:
1.Dataset Class:
  A custom dataset class, SERDataset, is implemented by inheriting from torch.utils.data.Dataset. This class is responsible for handling data loading and processing, allowing for seamless integration with PyTorch's data pipeline. The __init__ method initialises the dataset with a DataFrame containing file paths and emotion labels, as well as the Wav2Vec2 processor, which will transform the raw audio into the input format required by the model..
 The __getitem__ method loads an audio file, processes it using the Wav2Vec2    processor, and returns the processed audio tensor along with its corresponding label.
 2. Collate Function:
A collate_fn function is defined to handle variable-length audio sequences within a batch. This function ensures that all audio inputs are padded to the same length, allowing for batch processing without dimension mismatch errors. The padding is done dynamically based on the maximum length of audio samples in the current batch.
The collate function first processes each audio input and then pads them to ensure consistent dimensions within the batch. It returns a tensor of padded audio inputs and their corresponding labels.
3.Training Parameters: The following key training parameters are defined for fine-tuning the model:
Optimizer: The Adam optimizer is utilised with a learning rate of 1e-5, which is known for its efficiency and effectiveness in training deep learning models. A lower learning rate is chosen to ensure stable convergence during fine-tuning.
Loss Function: The CrossEntropyLoss function is employed, which is suitable for multi-class classification tasks. This loss function computes the loss between the predicted probabilities and the true labels, guiding the model in optimising its parameters during training.
Batch Size: A batch size of 8 is selected due to computational constraints. This size strikes a balance between memory usage and model convergence speed, allowing for efficient processing without overwhelming the GPU memory.
Number of Epochs: The training process is configured to run for 100 epochs, with early stopping implemented to prevent overfitting. The early stopping mechanism monitors validation loss, and if it does not improve over a set number of epochs, the training will cease.
The careful design of the SERDataset class and the collate_fn function allows for effective handling of audio data within the model training pipeline. By setting appropriate training parameters such as the optimizer, loss function, batch size, and number of epochs, the model is fine-tuned for the task of speech emotion recognition. This structured approach ensures that the model can learn meaningful representations from the audio data while maintaining efficiency and stability throughout the training process.
3.5 **HYPERPARAMETER TUNING**
Hyperparameter tuning is essential for optimising the performance of the Wav2Vec2 model in speech emotion recognition. The following parameters and strategies were utilised in the current implementation:
Optimizer:
   The Adam optimizer is used for training the model. It is effective for handling sparse gradients and is often used in deep learning for its adaptive learning rate properties.
Learning Rate: Set to 1e-5, which provides a good balance between training speed and convergence
      2. Loss Function:
The CrossEntropyLoss function is employed, which is suitable for multi-class classification tasks, such as emotion classification in this case
     3.Early Stopping:
To combat overfitting, early stopping is implemented to monitor the validation loss during training. If there is no improvement in the validation loss over a specified number of epochs (patience), training will be halted.
I had set the patience to 5.
3.6 **EVALUATION METRICS**
To assess the performance of the Wav2Vec2 model for speech emotion recognition, several evaluation metrics were utilised:
Accuracy:
Definition: Accuracy is defined as the proportion of correctly classified samples out of the total number of samples. It provides a straightforward metric to gauge the model's performance in correctly identifying emotions.
Implementation: The accuracy is computed using the accuracy_score function from the sklearn.metrics module.
    2.Loss:
Definition: Loss, specifically the cross-entropy loss, quantifies the difference between the predicted probabilities and the actual labels. During training, a lower loss indicates that the model is improving in terms of making correct predictions.
Implementation: The loss is tracked during both the training and validation phases. The average training loss is calculated and printed after each epoch. Similarly, the validation loss is computed using the same CrossEntropyLoss criterion and is reported at the end of each validation pass:

3.Confusion Matrix:
Definition: A confusion matrix is a useful tool for visualising the performance of a classification model, showing the true positives, false positives, true negatives, and false negatives across different emotion classes. It helps in understanding which classes are being confused with one another.
Implementation: After evaluating the model on the test set, the confusion matrix is calculated using the confusion_matrix function from the sklearn.metrics module. The confusion matrix is then visualised using a heat map created with the seaborn library


**IMPLEMENTATION**
4.1 **DATASET SPLITTING AND LOADING**
The data is loaded using the load_data_from_directory function, which reads .wav files from the dataset directory and creates a DataFrame with file paths and emotion labels. The data is then split into training and testing sets.
4.2 **TRAINING PROCESS**
Model Initialization: The Wav2Vec2 model is initialised with the number of emotion labels.
Training Loop: The model is trained over multiple epochs, with each epoch consisting of a training  phase and a validation phase.
Forward Pass: Inputs are passed through the model to obtain outputs.
Loss Calculation: The loss between outputs and true labels is computed.
Backward Pass: Gradients are calculated, and the optimizer updates the model weights.
Validation: After each epoch, the model is evaluated on the validation set, and metrics are recorded.
4.3 **USE OF EARLY STOPPING**
Early stopping is implemented to halt training when the validation loss does not improve for 5 consecutive epochs. This prevents overfitting and saves computational resources.
4.4 **MODEL EVALUATION**
Loading Best Model: After training, the best model (with the lowest validation loss) is loaded for evaluation.
Testing: The model is evaluated on the test set, and the accuracy is calculated.
Visualisation:
Training and Validation Loss Plot: Shows the decrease in loss over epochs.
Validation Accuracy Plot: Illustrates the improvement in accuracy.
Confusion Matrix: Provides a detailed view of the model's performance across emotion classes.


**RESULTS AND DISCUSSION**
5.1 **PERFORMANCE ON RAVDESS DATASET**
Training Progress: The model was trained for 24 epochs before early stopping was triggered. The training loss decreased steadily, indicating effective learning.
Example Epochs:
Epoch 10/100, Train Loss: 0.7541, Val Loss: 0.5826, Val Accuracy: 0.8491
 Epoch 15/100, Train Loss: 0.3402, Val Loss: 0.3875, Val Accuracy: 0.9057
Test Accuracy: The model achieved a test accuracy of 90.57%, demonstrating strong performance on the SER task.

Cross-Entropy Loss
Formula:
Cross-Entropy Loss= −∑  i=1 to N (yi)( log⁡(pi))
Where:
yi​ is the true label.
pi​ is the predicted probability of the correct label.
N is the number of classes.
The cross-entropy loss was used as the loss function to optimise the model. Both training and validation loss steadily decreased over the training epochs, as shown in the loss curves, with only minor fluctuations in the validation loss toward the end of training, indicating the potential onset of overfitting.


Accuracy=(TP+TN)/(FP+FN +TP+TN)​
TP (True Positives): Correctly predicted emotions.
TN (True Negatives): Correctly predicted non-emotions.
FP (False Positives): Incorrectly predicted an emotion.
FN (False Negatives): Failed to predict an emotion.
         The accuracy of the model reached approximately 90% during validation, indicating that 90% of the validation data was correctly classified. This high accuracy demonstrates the effectiveness of the model in recognizing emotions across the dataset.

The confusion matrix provides detailed insights into the performance of the model across the six emotion classes. The matrix shows the distribution of predictions for each class, allowing us to assess where the model is performing well and where misclassifications occur.
Confusion Matrix:
Confusion Matrix=​[TP 0,0	FN 0,1		FN 0,2       …….     FP 0,n]
                               [FN 1,0	TP1,1		FP1,2        …….     FP1,n]
                               :
                               :
                               [FNn,0	FNn,1		FNn,2       ……..    TPn,n]
From the confusion matrix, we observe that most predictions fall on the diagonal, indicating that the model has a high rate of correct classifications for most emotions. However, certain emotions (particularly classes 2, 4, and 5) experienced more misclassifications, suggesting the need for further fine-tuning or more data to improve performance on these classes.
5.2 **ANALYSIS OF RESULTS**
Loss Curves: The training and validation loss curves show a consistent decrease, with the validation loss stabilising before early stopping.
Accuracy: Validation accuracy improved over epochs, reaching a peak before plateauing.
Confusion Matrix: The confusion matrix indicates that the model performs well across most emotions, with some confusion between similar emotions like 'sad' and 'neutral'.
5.3 **CHALLENGES FACED**
Computational Constraints: Training with a batch size larger than 8 was not feasible due to hardware limitations.
Data Imbalance: While the dataset is relatively balanced, some emotions may have slightly fewer samples, affecting model performance.
Overfitting: Early stopping and learning rate scheduling were necessary to prevent overfitting.


**CONCLUSION AND FUTURE WORK**
6.1 **TO SUMMARISE**
This research successfully implemented a Speech Emotion Recognition system using the Wav2Vec2 model fine-tuned on the RAVDESS dataset. The model achieved a high test accuracy of 90.57%, indicating that Wav2Vec2 effectively captures emotional cues from speech without extensive feature engineering.

6.2 **FUTURE SCOPE**
Data Augmentation: Employ techniques like noise injection, pitch shifting, or time stretching to increase dataset diversity.
Model Enhancements: Experiment with larger models like wav2vec2-large or incorporate attention mechanisms to improve performance.
Cross-Dataset Evaluation: Test the model on different datasets to evaluate generalisation.
Multimodal Approaches: Combine audio with visual data (facial expressions) for a more comprehensive emotion recognition system.


**REFERENCES**
Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33, 12449-12460.
Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PloS one, 13(5), e0196391.

