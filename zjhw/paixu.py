import string

input_str = '4CA32cd'

letter = []
let_index = []
digits = []
ldig_index = []
a = 0

for i in input_str:
    if i in string.ascii_letters:
        letter.append(i)
        let_index.append(a)
    else:
        digits.append(i)
        ldig_index.append(a)

    a = a + 1

letter.sort()
digits.sort()

print("letter:", letter)
print("let_index:", let_index)
print("digits:", digits)
print("ldig_index:", ldig_index)

new = []
b = 0
while b < len(ldig_index):
    if let_index[b] < ldig_index[b]:
        new.append(letter[0])
    else:
        new.append(digits[0])
    b = b + 1

print(new)
