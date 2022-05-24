### Paragraph-Summarization
# Requirements

This project was done on Google Colab and then run on a local machine. However this is not system system and can be run on any computer that can run python. For some reason, if your system is not compatible to run python, you can upload "main.py" on Google Colab and view the output.
Machine - Apple Mac M1 - Monterey OS

In order to run this, you will first need to ensure that your system has the required libraries. These are mentioned in the requirements.txt file.

# Steps to run

pip install -r requirements.txt

python3 main.py

# Prompts on the command line

1. For the first prompt, you have to choose an excel file from your system that you would like to test for.
2. Next,from the list of column, enter the column that you want to train.
3. Now pick a representative sentence that you want to compare the rest with.
4. Enter 'n', i.e. the number of top similar sentences that you want the code to return.

# Output

You should see the top 'n' sentences with their similarity scores, the word cloud representation of the sentences and the sentiments (positive/negative/neutral) of the top 'n' sentences.
