import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

from keras.models import Sequential
from keras.layers import LSTM, Dense , Dropout
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import save_model, load_model


from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,Callback

df = pd.read_csv('dataset/Tasks_DataSet.csv')

# Extract features and target
X = df.drop(columns=['DataCenterID'])
y = df['DataCenterID']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(df.columns)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Reshape data for LSTM input (assuming a time series sequence length of 1)
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_lstm = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

#this params values from GA algorithm
model = Sequential()
model.add(LSTM(units=int(54), input_shape=(1, X_train.shape[1]), return_sequences=True))
model.add(LSTM(units=int(62)))
model.add(Dropout(0.08203125))
model.add(Dense(units=len(df['DataCenterID'].unique()), activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.014921875000000001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
history=model.fit(X_train_lstm, y_train, epochs=94, batch_size=18, validation_data=(X_val_lstm, y_val), callbacks=[early_stopping])



# Assuming you have your X_test and y_test prepared similarly to X_train and y_train
X_test_lstm = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_lstm, y_val)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Make predictions on the test set
y_pred = model.predict(X_test_lstm)
# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_val, y_pred_classes))

# Create a confusion matrix
cm = confusion_matrix(y_val, y_pred_classes)

# Create a colorful confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# model.save('models/lstm89.keras')



def predict_datacenter_id(requested_array,model_name):
    data_dict = {
        "TaskID": requested_array[0],
        "StartTime": requested_array[1],
        "TaskFileSize": requested_array[2],
        "TaskOutputFileSize": requested_array[3],
        "TaskFileLength": requested_array[4],

    }
    input_data = np.array([[
        data_dict["TaskID"],
        data_dict["StartTime"],
        data_dict["TaskFileSize"],
        data_dict["TaskOutputFileSize"],
        data_dict["TaskFileLength"]]])
    # Load the saved model
    loaded_model = load_model(model_name)
    # Reshape the input data to match the LSTM input shape
    input_data_lstm = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
    # Make predictions using the loaded model
    predicted_probabilities = loaded_model.predict(input_data_lstm)
    predicted_class = np.argmax(predicted_probabilities, axis=1)
    predicted_DC=predicted_class[0]
    return predicted_DC
requested_array = [0.1, 55, 0, 0, 55]
predicted_datacenter_id = predict_datacenter_id(requested_array,"models/lstm89.keras")
print(f"Predicted DataCenterID: {predicted_datacenter_id}")

# Define the LSTM model
def create_and_train_model(units1, units2, dropout_rate, learning_rate, epochs, batch_size):
    print("\n\nNew generation training start. Parameters:")
    print(f"Units1: {units1}, Units2: {units2}, Dropout Rate: {dropout_rate}, Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}\n")


    model = Sequential()
    model.add(LSTM(units=int(units1), input_shape=(1, X_train.shape[1]), return_sequences=True))
    model.add(LSTM(units=int(units2)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=len(df['DataCenterID'].unique()), activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train_lstm, y_train, epochs=int(epochs), batch_size=int(batch_size), validation_data=(X_val_lstm, y_val), callbacks=[early_stopping])

    # Return the validation accuracy as the fitness
    return history.history['accuracy'][-1]

# Define individual
indv_template = BinaryIndividual(ranges=[(10, 100), (10, 100), (0.0, 1.0), (0.001, 0.1), (10, 100), (10, 100)])

# Create population
population = Population(indv_template=indv_template, size=50).init()

# Define Genetic Algorithm operators
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create Genetic Algorithm engine
engine = GAEngine(population=population, selection=selection, crossover=crossover, mutation=mutation)

# Define and register fitness function
@engine.fitness_register
def fitness(indv):
    # Decode GA individual to LSTM parameters
    units1, units2, dropout_rate, learning_rate, epochs, batch_size = indv.solution

    # Create and train LSTM model
    val_accuracy = create_and_train_model(units1, units2, dropout_rate, learning_rate, epochs, batch_size)
    
    return val_accuracy

# Run the Genetic Algorithm
engine.run(ng=50)
