from model import predict_news

case_a = "Marine biologists conducting a multi-year ecological survey in the South Pacific have documented a remarkably resilient recovery among coral reef systems previously devastated by bleaching events."
case_b = "Secret alien bases discovered on the dark side of the moon by military whistleblowers confirming global conspiracy."

res_a = predict_news(case_a)
res_b = predict_news(case_b)

print("=" * 60)
print("CASE A (Genuine News without publisher tags):")
print("Text:", case_a)
print("Classification:", res_a['classification'])
print(f"Confidence:     {res_a['confidence']}%")
print(f"Real Prob:      {res_a['real_prob']}%")
print(f"Fake Prob:      {res_a['fake_prob']}%")
print("Explanation:   ", res_a['explanation'])
print("Indicators:")
for ind in res_a['indicators']:
    print(f"  [{ind['status'].upper()}] {ind['title']}: {ind['desc']}")

print("\n" + "=" * 60)
print("CASE B (Obvious Fake News):")
print("Text:", case_b)
print("Classification:", res_b['classification'])
print(f"Confidence:     {res_b['confidence']}%")
print(f"Real Prob:      {res_b['real_prob']}%")
print(f"Fake Prob:      {res_b['fake_prob']}%")
print("Explanation:   ", res_b['explanation'])
print("Indicators:")
for ind in res_b['indicators']:
    print(f"  [{ind['status'].upper()}] {ind['title']}: {ind['desc']}")

# Test with publisher watermark prepended to genuine text:
case_a_reuters = "WASHINGTON (Reuters) - " + case_a
res_a_reuters = predict_news(case_a_reuters)

print("\n" + "=" * 60)
print("CASE A with 'WASHINGTON (Reuters) - ' prepended:")
print("Classification:", res_a_reuters['classification'])
print(f"Confidence:     {res_a_reuters['confidence']}%")
print(f"Real Prob:      {res_a_reuters['real_prob']}%")

assert res_a['classification'] == 'Real', "Case A failed: Expected Real"
assert res_a['confidence'] >= 70, f"Case A failed: Confidence too low ({res_a['confidence']}%)"
assert res_b['classification'] == 'Fake', "Case B failed: Expected Fake"
assert res_b['confidence'] >= 70, f"Case B failed: Confidence too low ({res_b['confidence']}%)"
print("\n" + "*" * 60)
print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
print("*" * 60)
