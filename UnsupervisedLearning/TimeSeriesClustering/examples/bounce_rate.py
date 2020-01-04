transitions = {}
row_sums = {}
file_name = r"../data/site_data.csv"

# collect counts
for line in open(file_name):
    s, e = line.rstrip().split(',')
    transitions[(s, e)] = transitions.get((s, e), 0.) + 1
    row_sums[s] = row_sums.get(s, 0.) + 1

# normalize
for (s, e), v in transitions.items():
    transitions[(s, e)] = v / row_sums[s]

# initial state distribution
print("initial state distribution")
print("======================================")
for (s, e), v in transitions.items():
    if s == '-1':
        print(e, v)

# which page has the highest bounce?
print('\n')
print("bounce rate")
print("========================================")
for (s, e), v in transitions.items():
    if e == 'B':
        print(f"bounce rate for {s}: {v}")