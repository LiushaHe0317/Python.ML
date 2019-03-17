
import re

x = ["This is a wolf #scary",
     "Welcome to this jungle #missing",
     "11322 the number to know",
     "remember the name s - John",
     "I     love       you"]

y = []
for sentence in x:
    mod1 = re.sub(r"\W"," ",sentence)
    mod2 = re.sub(r"\s+"," ",mod1)
    mod3 = re.sub(r"\s+[a-z]\s"," ",mod2)
    mod4 = re.sub(r"\d","",mod3)
    mod5 = re.sub(r"^\s","",mod4)
    mod6 = re.sub(r"\s$","",mod5)

    y.append(mod6)

print(y)