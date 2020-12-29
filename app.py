import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random


model = pickle.load(open('twweet_model.pkl','rb'))

def func(custom_tweet):
  custom_tokens = remove_noise(word_tokenize(custom_tweet))
  return classifier.classify(dict([token, True] for token in custom_tokens))

def main():
	page_bg_img = '''
	<style>
	body {
	background-image: url("https://i.pinimg.com/originals/a1/d0/bd/a1d0bdd2b7e908826e05b03ef3c8718a.jpg");
	background-size: cover;
	}
	</style>
	'''

	st.markdown(page_bg_img, unsafe_allow_html=True)
	st.title("Tweet Classifier")
	st.write("Enter tweet in text form:")
	input = st.text_input("Label")
	predict_button = st.button("Predict")
	if predict_button:
		result = func(input)
		st.success("Type: {}".format(result))

if __name__=="__main__":
	main()