Diabetes Prediction Console Application
This is a Python-based console application that utilizes machine learning to predict whether a patient has diabetes based on specific symptoms. Using prior data stored in a CSV file, the app performs data analysis and provides a prediction. The project includes visualizations using matplotlib and a simple GUI interface built with tkinter.

Features
Machine Learning: Implements a machine learning model to predict diabetes based on input symptoms.
Data Visualization: Utilizes matplotlib to present data in a visual format.
Simple GUI: Uses tkinter to create a user-friendly interface.
Data Analysis: Reads and processes data from a CSV file containing historical patient information.
Installation
To run this project, you'll need Python 3 installed. You can install the required dependencies using pip:

pip install -r requirements.txt

Note: Ensure requirements.txt includes necessary libraries such as matplotlib, tkinter, pandas, scikit-learn, and any others used in the project.

Usage
Run the main script from your console:


python diabetes_formatted.py
Follow the on-screen instructions to input patient symptoms.

View the prediction results displayed on the console and, if available, in the GUI.

Project Structure
diabetes_formatted.py: Main application script, including machine learning model, data processing, and GUI code.
data/diabetes_data.csv: Historical data used to train the model (ensure this is in the specified path).
Contributing
Contributions are welcome! Feel free to submit issues or make pull requests for any improvements.
