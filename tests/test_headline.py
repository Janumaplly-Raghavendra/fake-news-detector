
import os
from model import predict_news

def check(h):
    r = predict_news(h)
    print(f"Headline: {h}")
    print(f"Classification: {r['classification']}")
    print(f"Score: {r['score']}")
    print(f"Keywords: {r['keywords']}")

if __name__ == "__main__":
    check("Iran-Israel war LIVE: Trump again warns Iran to open Strait of Hormuz")
