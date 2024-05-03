import tensorflow as tf
import streamlit as st
import joblib
import sklearn
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer

# Check if wordnet is installed
try:                                                                         
    nltk.find("corpora/popular.zip")          
except LookupError:
    nltk.download('popular')

# read sw_new txt file
with open("sw_new.txt", "r") as f:
  sw_new = f.read()
sw_new = sw_new.split("\n")

css = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://cdn.discordapp.com/attachments/1075699203046641687/1165351110828101673/PhotoReal_From_a_distance_the_plane_appears_as_a_tiny_speck_ag_1.jpg?ex=654688cb&is=653413cb&hm=096dcc994304b93afd210555607d563f725d74c1dddfd392176f11e15076bcfa&");
background-size: 120%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stExpander"] {{
background: rgba(0,0,0,0.5);
border: 2px solid #000071;
border-radius: 10px;
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# define text cleaner function
def text_cleaner(text, sw = sw_new):

  # mobile_regex = "(\+*)((0[ -]*)*|((91 )*))((\d{12})+|(\d{10})+)|\d{5}([- ]*)\d{6}"
  url_regex = "((http|https|www)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
  space_regex = "\s\s+"
  # remove url
  text = re.sub(url_regex, "", text)
  # remove mobile
  # text = re.sub(mobile_regex, "", text)
  # lower casing
  text = text.lower()
  # remove emoji & punctuation & numbers
  text = "".join([i for i in text if (ord(i) in range(97,123)) | (i == " ")])
  # remove multiple spaces
  text = re.sub(space_regex, " ", text)

  # stopword removal
  text = [i for i in text.split() if i not in sw]
  # lemmatizing
  lemma = WordNetLemmatizer()
  text = " ".join([lemma.lemmatize(i) for i in text])

  return text

# load model
@st.cache_resource
def cache_model(model1_add, model2_add):
    model1 = tf.saved_model.load(model1_add)
    model2 = joblib.load(model2_add)
    return model1, model2

nlp_model, nb_model = cache_model("food_review_use_model",
                     "food_review_nb.joblib")

def prediction(text):
    text = text_cleaner(text)
    y_pred_nb = nb_model.predict([text])
    y_pred_nlp = nlp_model([text]).numpy()
    # st.write(y_pred_nlp)
    return y_pred_nlp, y_pred_nb

# UI
# title
st.title("Food Review Sentiment Analysis")
st.image("banner.png")
review = st.text_area(
    "Enter or paste a food review to analyze",
)

pred = st.button("Get_Prediction")

if pred:
    if review:
        y_pred_nlp, y_pred_nb = prediction(review)
        nlp_result = ("Negative Review" if y_pred_nlp[0,0] >= 0.5 else "Positive Review")
        nb_result = ("Negative Review" if y_pred_nb[0] == 1 else "Positive Review")
        if nlp_result == "Positive Review":
            st.write(
            f"NLP Model Output: {nlp_result} with {round((1 - y_pred_nlp[0,0])*100, 2)}% Probability"
            )
        else:
            st.write(
                f"NLP Model Output: {nlp_result} with {round(y_pred_nlp[0,0]*100, 2)}% Probability"
            )

        st.write(
            f"Naive Bayes Model Output: {nb_result}"
        )

    else:
        st.write("Please enter review")

else:
    pass