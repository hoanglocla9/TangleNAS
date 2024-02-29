import pickle
a = dict()
for i in range(1,41):
    benchmark_file_name = "/path/to/benchmark_dictionary_small_"+str(i)+".pkl"
    with open(benchmark_file_name, 'rb') as f:
        b = pickle.load(f)
    print(len(b))
    a.update(b)
print(len(a))
with open("/path/to/entangled_cell_ss_benchmark.pkl", 'wb') as f:
    pickle.dump(a, f)