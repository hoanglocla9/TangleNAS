import pickle
a = dict()
for i in range(1,41):
    benchmark_file_name = "/work/dlclarge2/sukthank-tanglenas/small_search_spaces/TangleNAS-dev/search_spaces/toy_entangled_cell_ss/benchmark_dictionary_small_"+str(i)+".pkl"
    with open(benchmark_file_name, 'rb') as f:
        b = pickle.load(f)
    print(len(b))
    a.update(b)
print(len(a))
with open("/work/dlclarge2/sukthank-tanglenas/small_search_spaces/TangleNAS-dev/search_spaces/toy_entangled_cell_ss/entangled_cell_ss_benchmark.pkl", 'wb') as f:
    pickle.dump(a, f)