
******************************************* For text_transcript_identifier.py ********************************************
The script takes in a text and a transcript with synonymous topic. The output is the percentages of the key-words (phrases), extracted from the text, that have matched components in the colloquial transcript.

The problem I'd like to resolve here is accurately retrieve, at least proportionally, phrases under casual speech, i.e. transcript with no capitalization or punctuation. Under these circumstances, they are likely to be separated by extraneous expressions or their sequences might vary, due to users' conversational habits.

The script here is the baseline system implementing only NLTK and fundamental functions of set data structure.

To run the script, type:
python text_transcript_identifier_ziang.py text.txt transcript_text.txt (least # of char in each word) (largest # of words in a phrase) (# of phrases to show)