import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Create a tiny dummy dataset
data = [
    # --- Real Samples ---
    ("Scientists confirm that COVID-19 vaccines are safe and effective, according to data published by the CDC and WHO.", 1),
    ("The president signed a new trade bill yesterday, officials confirmed. The legislation is expected to affect tariffs on imported goods.", 1),
    ("Research shows that regular exercise improves heart health and longevity, according to a report says.", 1),
    ("NASA's Artemis mission aims to return humans to the lunar surface by 2025, according to official statements.", 1),
    ("The local city council approved a new budget for public schools today, sources say.", 1),
    ("According to a study published in Nature, global temperatures continue to rise at an alarming rate.", 1),
    ("The company reported quarterly earnings that exceeded analyst expectations, officials confirmed.", 1),
    ("International peace talks began this morning in Geneva, according to sources familiar with the matter.", 1),
    
    # --- Fake Samples ---
    ("SHOCKING: Government has been secretly putting mind-control chemicals in the water supply. Wake up sheeple! They don't want you to know!", 0),
    ("EXCLUSIVE: Aliens have landed in DC and are currently meeting with world leaders in a secret underground bunker exposed.", 0),
    ("Unbelievable scandal: Global elite plans to ban all food and replace it with synthetic pills by 2025. Deep state conspiracy hoax!", 0),
    ("WATCH: Celebrities are actually lizard people in disguise. This shocking video proof is being banned everywhere!", 0),
    ("BREAKING: Magic crystals found to cure all diseases instantly. Doctors hate this one weird trick! Scandal exposed!", 0),
    ("The moon is actually a hollow space station made by an ancient civilization, secret documents reveal.", 0),
    ("New law requires all citizens to wear tinfoil hats to block 5G mind rays, unbelievable scandal!", 0),
    ("Shocking report: Drinking salt water can make you fly. Scientists are hiding this secret discovery from the public!", 0)
]

df = pd.DataFrame(data, columns=['text', 'label'])

# 2. Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Train a simple model
model = LogisticRegression()
model.fit(X, y)

# 4. Save artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Mock model and vectorizer generated successfully!")
