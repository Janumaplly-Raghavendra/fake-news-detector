
import os
import sys
from model import predict_news

headline = "Iran-Israel war LIVE: Trump again warns Iran to open Strait of Hormuz"
result = predict_news(headline)

print(f"Headline: {headline}")
print(f"Classification: {result['classification']}")
print(f"Score: {result['score']}")
print(f"Explanation: {result['explanation']}")
print(f"Keywords: {result['keywords']}")
