
import gzip
import pickle
import sys

def datatocnf(filename, output):
    num_vars, num_clauses, labels, edges = pickle.load(gzip.open(filename, 'rb'))
    foutput = open(output, 'wt')
    foutput.write(f'p cnf {num_vars} {num_clauses}\n')
    clauses = [list() for _ in range(num_clauses)]
    for c, l in edges:
        l = l+1 if l < num_vars else -(l-num_vars+1)
        clauses[c].append(l)
    for c in clauses:
        for l in c:
            foutput.write(f'{l} ')
        foutput.write('0\n')

if __name__ == "__main__":
    datatocnf(sys.argv[1], sys.argv[2])
