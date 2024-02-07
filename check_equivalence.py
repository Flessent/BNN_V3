from pysat.solvers import Glucose3
from pysat.formula import CNF
import random
def are_cnf_files_equivalent(file1, file2):
    cnf1 = CNF(from_file=file1)
    cnf2 = CNF(from_file=file2)

    solver = Glucose3()

    for clause in cnf1.clauses:
        solver.add_clause(clause)
    for clause in cnf2.clauses:
        solver.add_clause(clause)

    result = solver.solve()

    if result == True:
        print("The CNF formulas are equivalent.")
    elif result == False:
        print("The CNF formulas are not equivalent.")
    else:
        print("Solver could not determine equivalence.")
def read_dimacs_file(file_path):
    
    cnf = CNF()
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c') or line.startswith('p'):
                continue
            clause = [int(literal) for literal in line.split() if literal != '0']
            cnf.append(clause)
    return cnf

def write_dimacs_file(cnf, file_path):
    with open(file_path, 'w') as file:
        for clause in cnf.clauses:
            file.write(' '.join(map(str, clause)) + ' 0\n')

def cnf_conjunction(cnf1, cnf2):
    cnf_result = CNF()
    cnf_result.extend(cnf1.clauses)
    cnf_result.extend(cnf2.clauses)
    return cnf_result
def check_satisfiability(cnf_file):
    solver =Glucose3()
    cnf = CNF(from_file='output_conjunction.cnf')
    for clause in cnf.clauses:
        solver.add_clause(clause)
    result = solver.solve()
    if result == True:
        print("The CNF formulas are equivalent.")
    elif result == False:
        print("The CNF formulas are not equivalent.")
    else:
        print("Solver could not determine equivalence.")

def generate_binary_sequence(length):
    return ''.join(random.choice('01') for _ in range(length))

def generate_dataset(num_samples, sequence_length):
    dataset = []
    for _ in range(num_samples):
        binary_sequence = generate_binary_sequence(sequence_length)
        label = format(random.randint(0, 15), '04b')
        dataset.append(f"{binary_sequence} {label}")
    return dataset

if __name__ == "__main__":
    #file1 = 'test.cnf'
    #file2 = 'bnntocnf.cnf'
    #are_cnf_files_equivalent(file1, file2)
    cnf_bdd = read_dimacs_file('test.cnf')
    cnf_bnn = read_dimacs_file('bnntocnf.cnf')
    neg_cnf_bdd=cnf_bdd.negate()
    neg_cnf_bdd.to_file(fname='neg_cnf_bdd.cnf')

    cnf_result = cnf_conjunction(neg_cnf_bdd, cnf_bnn)

    write_dimacs_file(cnf_result, 'output_conjunction.cnf')
    check_satisfiability('output_conjunction.cnf')
    """num_samples = 1000
    sequence_length = 18
    generated_dataset = generate_dataset(num_samples, sequence_length)

    with open('generated_dataset.txt', 'w') as file:
        for sample in generated_dataset:
            file.write(sample + '\n')

    print("Dataset generated and saved to 'generated_dataset.txt'")"""


