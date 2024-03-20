import pickle
a = {}
for i in range(1,81):
    benchmark_file_name = "/path/to/benchmark_dictionary_"+str(i)+".pkl"
    print("Reading file = " + benchmark_file_name)
    with open(benchmark_file_name, 'rb') as f:
        b = pickle.load(f)
    for k in b.keys():
        if k in a:
            print("Duplicate found")
            print(k)
        else:
            a[k] = b[k]
print(len(a))
with open("benchmark_all_archs_final.pkl", 'wb') as f:
    pickle.dump(a, f)
