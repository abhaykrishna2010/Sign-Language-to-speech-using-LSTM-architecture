# Sign-Language-to-speech-using-LSTM-architecture
This project converts sign language gestures into spoken words using a deep learning model based on LSTM.

To run this project on your local system follow the following steps:

1. Open terminal and type - ```git clone https://github.com/abhaykrishna2010/Sign-Language-to-speech-using-LSTM-architecture```
2. Open that directory - ```cd <FOLDERNAME>```
3. You can create a virtual environment if needed, look [here](https://docs.python.org/3/library/venv.html) for instructions.
4. Install all the dependencies ```pip install -r requirements.txt```
5. To do a sample test to check mediapipe functionalities ```python CheckMediaPipe.py```
6. To run the file follow the below sequence:
  1. ```python Config.py```
  2. ```python CreateFolders.py```
  3. ```python Function.py```
  4. ```python BuildDataset.py```
  5. ```python PreprocessingTrainModel.py```
  6. ```python Main.py```
7. press CTRL + C in terminal to end process while the cv2 capture is running.
8. To add more actions to the sign language change the values in ```Config.py``` and follow the sequence again.
