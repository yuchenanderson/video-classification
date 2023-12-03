import pandas as pd
import numpy as np
import cv2
import os
from tensorflow import keras
import clip

# %%
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15  # 20

MAX_SEQ_LENGTH = 20  # 10, 15
NUM_FEATURES = 2048
# %%
train_dataframe = pd.read_csv("train.csv")
test_dataframe = pd.read_csv("test.csv")


# %%
def extract_square_region(image_frame):
    """
    Extracts a centered square region from a given image frame.

    Args:
    image_frame (numpy array): The image frame from which the square region is to be extracted.

    Returns:
    numpy array: The extracted square region of the image.
    """
    # Retrieve the height and width of the image frame
    height, width = image_frame.shape[0:2]

    # Determine the smallest dimension (height or width) to maintain aspect ratio
    smallest_dimension = min(height, width)

    # Calculate the starting x-coordinate for cropping
    crop_start_x = (width // 2) - (smallest_dimension // 2)

    # Calculate the starting y-coordinate for cropping
    crop_start_y = (height // 2) - (smallest_dimension // 2)

    # Extract and return the centered square region of the image
    return image_frame[crop_start_y: crop_start_y + smallest_dimension, crop_start_x: crop_start_x + smallest_dimension]


def fetch_video_frames(video_path, frame_limit=0, new_size=(IMG_SIZE, IMG_SIZE)):
    """
    Fetches frames from a video file, processes each frame, and returns them as an array.

    Args:
    video_path (str): The file path of the video.
    frame_limit (int): The maximum number of frames to fetch. 0 for no limit.
    new_size (tuple): The new size (width, height) to resize each frame to.

    Returns:
    numpy array: An array of processed video frames.
    """
    # Initialize video capture from the given path
    video_capture = cv2.VideoCapture(video_path)

    # Initialize an empty list to store processed frames
    processed_frames = []

    try:
        while True:
            # Read a frame from the video
            frame_read, current_frame = video_capture.read()

            # Break the loop if no frame is read
            if not frame_read:
                break

            # Crop the current frame to a centered square
            square_frame = extract_square_region(current_frame)

            # Resize the cropped frame to the new size
            resized_frame = cv2.resize(square_frame, new_size)

            # Reorder the color channels from BGR to RGB
            rgb_frame = resized_frame[:, :, [2, 1, 0]]

            # Append the processed frame to the list
            processed_frames.append(rgb_frame)

            # Break the loop if the frame limit is reached
            if frame_limit != 0 and len(processed_frames) == frame_limit:
                break
    finally:
        # Release the video capture object
        video_capture.release()

    # Convert the list of frames to a numpy array and return
    return np.array(processed_frames)


def create_inception_feature_model():
    """
    Creates and returns a feature extraction model based on the InceptionV3 architecture.

    This model is designed to extract features from images using the InceptionV3 network pre-trained on ImageNet,
    without its top classification layers, and with average pooling. The model also includes an input preprocessing step.

    Returns:
    keras.Model: A Keras model configured for feature extraction.
    """
    # Define the preprocessing function for InceptionV3
    preprocess_input = keras.applications.inception_v3.preprocess_input

    # Define the InceptionV3 model for feature extraction
    inception_extractor = keras.applications.InceptionV3(
        weights="imagenet",  # Use weights pre-trained on ImageNet
        include_top=False,  # Exclude the top classification layers
        pooling="avg",  # Use average pooling
        input_shape=(IMG_SIZE, IMG_SIZE, 3)  # Define the input shape
    )

    # Define the model input
    model_input = keras.Input((IMG_SIZE, IMG_SIZE, 3))

    # Apply the preprocessing to the input
    preprocessed_input = preprocess_input(model_input)

    # Get the output from the InceptionV3 model
    model_output = inception_extractor(preprocessed_input)

    # Create and return the final model
    return keras.Model(model_input, model_output, name="inception_feature_extractor")


import torch
import clip


def extract_clip_features(image_tensor):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    # Move the image to the same device as the model
    image_tensor = image_tensor.to(device)

    # Forward pass through the CLIP model
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)

    # Normalize the features and return
    return image_features / image_features.norm(dim=-1, keepdim=True)


def video_processing(video_directory, dataset_dataframe, feature_ext, is_clip=False):
    """
    This function processes a dataset of videos for feature extraction and mask creation.
    It is designed to handle a DataFrame where each row represents a video with associated metadata.

    Args:
    dataframe: A pandas DataFrame containing video information.
               It is expected to have columns 'video_name' and 'tag', where 'video_name' is the name of the video file,
               and 'tag' is the label associated with the video.
    directory: The root directory path as a string where the video files are stored. This path is used to locate the video files.
    feature_ext: A callable (function or class instance) that is used to extract features from video frames.
                 This function should take a batch of frames and return the corresponding features.

    Returns:
    A tuple of two elements:
        - feature_matrix: A NumPy array containing the extracted features from each video.
        - mask_matrix: A NumPy array indicating which frames in the feature_matrix are valid (not masked).
        - video_labels: The labels for each video extracted from the dataframe, processed by the label_processor.
    """

    # Number of videos in the dataset
    total_videos = len(dataset_dataframe)

    # Extracting paths to individual video files
    video_file_paths = dataset_dataframe["video_name"].tolist()

    # Extracting labels for each video and processing them
    video_labels = dataset_dataframe["tag"].values
    video_labels = label_processor(video_labels[..., None]).numpy()

    # Initializing matrices to store masks and features for all videos
    mask_matrix = np.zeros((total_videos, MAX_SEQ_LENGTH), dtype=bool)
    feature_matrix = np.zeros((total_videos, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

    # Iterating over each video in the dataset
    for video_index, video_path in enumerate(video_file_paths):
        # Fetching frames from the video file and adding a batch dimension
        video_frames = fetch_video_frames(os.path.join(video_directory, video_path))
        video_frames = video_frames[None, ...]

        # Initializing temporary masks and features for the current video
        video_mask = np.zeros((1, MAX_SEQ_LENGTH,), dtype=bool)
        video_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')

        # Iterating over batches of frames in the current video
        for frame_index, frame_batch in enumerate(video_frames):
            # Determining the number of frames to consider (up to MAX_SEQ_LENGTH)
            frame_count = frame_batch.shape[0]
            frame_limit = min(MAX_SEQ_LENGTH, frame_count)
            if is_clip == False:
                # Extracting features for each frame within the limit
                for frame_num in range(frame_limit):
                    video_features[frame_index, frame_num, :] = feature_ext.predict(
                        frame_batch[None, frame_num, :]
                    ).cpu()
            else:
                # Extracting features for each frame within the limit
                for frame_num in range(frame_limit):
                    video_features[frame_index, frame_num, :] = extract_clip_features(
                        torch.from_numpy(frame_batch[None, frame_num, :]).permute(0, 3, 1, 2)
                    )

            # Updating the mask for the current video batch
            video_mask[frame_index, :frame_limit] = 1

        # Storing the features and mask of the current video in the respective matrices
        feature_matrix[video_index,] = video_features.squeeze()
        mask_matrix[video_index,] = video_mask.squeeze()

    # Returning the feature matrix, mask matrix, and video labels
    return (feature_matrix, mask_matrix), video_labels


# %%
def build_GRU_network():
    """
    Constructs a GRU-based neural network for sequence processing.

    Returns:
    A compiled Keras model designed for sequence classification tasks.

    The model comprises the following layers:
    - Input layers for frame features and a masking sequence.
    - Two GRU layers for sequential data processing.
    - A dropout layer for regularization.
    - A dense layer with ReLU activation for non-linear transformation.
    - An output dense layer with softmax activation for multi-class classification.
    """

    # Extracting the vocabulary size for the output layer
    num_class = len(label_processor.get_vocabulary())

    # Inputs for frame features and sequence mask
    input_features = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    input_mask = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # GRU layers for sequence processing, with specified units
    gru_layer1 = keras.layers.GRU(80, return_sequences=True)(input_features, mask=input_mask)
    gru_layer2 = keras.layers.GRU(40)(gru_layer1)

    # Regularization with dropout
    dropout_layer = keras.layers.Dropout(0.5)(gru_layer2)

    # Dense layer for further transformation
    dense_layer = keras.layers.Dense(40, activation="relu")(dropout_layer)

    # Final output layer for classification
    classification_layer = keras.layers.Dense(num_class, activation="softmax")(dense_layer)

    # Assembling the model
    gru_network = keras.Model(inputs=[input_features, input_mask], outputs=classification_layer)

    # Model compilation with appropriate loss and optimizer
    gru_network.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return gru_network


def build_LSTM_network():
    """
    Constructs and returns an LSTM-based neural network for processing sequential data.

    The network is designed for sequence classification tasks and includes the following components:
    - Input layers for receiving frame features and a corresponding mask.
    - Two LSTM layers for sequential data processing, with specified unit sizes.
    - A dropout layer for regularization to reduce overfitting.
    - A dense layer with ReLU activation for non-linear data processing.
    - An output dense layer with softmax activation, tailored for multi-class classification.

    The model is compiled with loss and metrics suitable for multi-class classification tasks.

    Returns:
    A compiled Keras LSTM model.
    """

    # Extracting class vocabulary size for the output layer
    num_classes = len(label_processor.get_vocabulary())

    # Defining input layers for frame features and masking
    feature_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    sequence_mask = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Building LSTM layers
    lstm_first_layer = keras.layers.LSTM(160, return_sequences=True)(feature_input, mask=sequence_mask)

    # Adding dropout for regularization
    dropout_first_layer = keras.layers.Dropout(0.5)(lstm_first_layer)

    # Building LSTM layers
    lstm_second_layer = keras.layers.LSTM(80)(dropout_first_layer)

    # Adding dropout for regularization
    dropout_second_layer = keras.layers.Dropout(0.5)(lstm_second_layer)

    # Dense layer for additional data processing
    dense = keras.layers.Dense(80, activation="relu")(dropout_second_layer)

    # Output layer for classification
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense)

    # Assembling the LSTM model
    lstm_network = keras.Model(inputs=[feature_input, sequence_mask], outputs=output_layer)

    # Compiling the model with suitable loss function and optimizer
    lstm_network.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return lstm_network


# %%
def execute_training_and_testing(model_generator, training_data, training_labels, evaluation_data, evaluation_labels,
                                 use_mask=True):
    """
    This function handles the training and evaluation of a sequential model.

    Args:
    model_generator: A function that returns a sequential model ready for training.
    training_data: Data used for training the model. It can be a tuple of (features, mask) or just features.
    training_labels: Labels corresponding to the training data.
    evaluation_data: Data used for evaluating the model. It can be a tuple of (features, mask) or just features.
    evaluation_labels: Labels corresponding to the evaluation data.
    use_mask: A boolean indicating whether a mask should be used in training and evaluation.

    Returns:
    A tuple containing the training history and the trained model.
    """

    # Initialize the sequential model
    sequential_model = model_generator()

    # Prepare training inputs based on whether a mask is used
    training_inputs = [training_data[0], training_data[1]] if use_mask else training_data[0]

    # Conduct the training process
    training_history = sequential_model.fit(
        training_inputs,
        training_labels,
        # validation_split=0.2,
        epochs=EPOCHS
    )

    # Prepare evaluation inputs in a similar way
    evaluation_inputs = [evaluation_data[0], evaluation_data[1]] if use_mask else evaluation_data[0]

    # Perform model evaluation
    _, test_accuracy = sequential_model.evaluate(
        evaluation_inputs,
        evaluation_labels
    )
    print(f"Evaluation accuracy: {round(test_accuracy * 100, 2)}%")

    return training_history, sequential_model


def execute_training_and_testing_transformer(model_generator, training_data, training_labels, evaluation_data,
                                             evaluation_labels, use_mask=False):
    """
    This function handles the training and evaluation of a sequential model.

    Args:
    model_generator: A function that returns a sequential model ready for training.
    training_data: Data used for training the model. It can be a tuple of (features, mask) or just features.
    training_labels: Labels corresponding to the training data.
    evaluation_data: Data used for evaluating the model. It can be a tuple of (features, mask) or just features.
    evaluation_labels: Labels corresponding to the evaluation data.
    use_mask: A boolean indicating whether a mask should be used in training and evaluation.

    Returns:
    A tuple containing the training history and the trained model.
    """

    # Initialize the sequential model
    sequential_model = model_generator()

    # Prepare training inputs based on whether a mask is used
    training_inputs = [training_data[0], training_data[1]] if use_mask else training_data[0]

    # Conduct the training process
    training_history = sequential_model.fit(
        training_inputs,
        training_labels,
        # validation_split=0.2,
        epochs=EPOCHS
    )

    # Prepare evaluation inputs in a similar way
    evaluation_inputs = [evaluation_data[0], evaluation_data[1]] if use_mask else evaluation_data[0]

    # Perform model evaluation
    _, test_accuracy = sequential_model.evaluate(
        evaluation_inputs,
        evaluation_labels
    )
    print(f"Evaluation accuracy: {round(test_accuracy * 100, 2)}%")

    return training_history, sequential_model


# %%

image_feature_extractor = create_inception_feature_model()
# image_feature_extractor = create_clip_feature_model()
# %%
label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_dataframe["tag"]))
print(label_processor.get_vocabulary())
# %%
# Use CNN feature extractor.
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
train_data, train_labels = video_processing("train", train_dataframe, image_feature_extractor)
test_data, test_labels = video_processing("test", test_dataframe, image_feature_extractor)

# %%
# Using CLIP feature extractor
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 768
train_data, train_labels = video_processing("train", train_dataframe, None, True)
test_data, test_labels = video_processing("test", test_dataframe, None, True)
print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")
# %%
np.save('train_data.npy', train_data[0])
np.save('train_data_mask.npy', train_data[1])
np.save('test_data.npy', test_data[0])
np.save('test_data_mask.npy', test_data[1])
np.save('train_labels.npy', train_labels)
np.save('test_labels', test_labels)
# %%
train_data = (np.load('train_data.npy'), np.load('train_data_mask.npy'))
test_data = (np.load('test_data.npy'), np.load('test_data_mask.npy'))
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')
print(train_data[0].shape)
print(test_data[0].shape)
# %%
# Transformor model for video classification.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Positional embedding layer for Transformers
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        embedded_positions = self.position_embeddings(positions)
        return x + embedded_positions


# Transformer block layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Define the Transformer model
def get_transformer_model():
    # Get the vocabulary of the classes
    class_vocab = label_processor.get_vocabulary()

    # Define the input layer for the frame features
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))

    # Positional Embedding
    x = PositionalEmbedding(MAX_SEQ_LENGTH, NUM_FEATURES)(frame_features_input)

    # Transformer blocks
    x = TransformerBlock(2048, 3, 32)(x)
    x = TransformerBlock(2048, 3, 32)(x)
    # x = TransformerBlock(2048, 2, 32)(x)
    # x = TransformerBlock(2048, 2, 32)(x)

    # Pooling and final dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(len(class_vocab), activation="softmax")(x)

    # Creating the model
    transformer_model = keras.Model(frame_features_input, output)

    # Compile the model
    transformer_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return transformer_model


# %%

# _, sequence_model = execute_training_and_testing(build_GRU_network, train_data, train_labels, test_data, test_labels)
# _, sequence_model = execute_training_and_testing(build_LSTM_network, train_data, train_labels, test_data, test_labels)
_, sequence_model = execute_training_and_testing_transformer(get_transformer_model, train_data, train_labels, test_data,
                                                             test_labels)