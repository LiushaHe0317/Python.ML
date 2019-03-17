
import time

a = ['a']

start_time = time.time()
i = 0;
for i in range(10000):
    i += 1
t = time.time() - start_time

print(start_time)
print(time.time())
print(t)