from collections import Counter

bigrams = lambda x: list(zip(x[:-1], x[1:]))

x = "I have been thinking about my Dad’s advice"
y = "I have been turning my Father’s advice over in my mind"
z = "I turned Dad’s advice over in my head"

x = Counter(bigrams(x.split()))
y = Counter(bigrams(y.split()))
z = Counter(bigrams(z.split()))
ref = y + z
num, denom = 0, 0
for b, c in x.items():
    num += min(ref.get(b, 0), c)
    denom += c

print(num / denom)


