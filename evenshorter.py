import numpy as np
import scipy.sparse as sparse
import warnings
import time
import sys
from numpy.linalg import eigvals
from scipy.sparse.linalg import expm, eigs
from scipy.optimize import minimize
from qiskit.opflow import PrimitiveOp, PauliTrotterEvolution
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit import Parameter
from qiskit import Aer, execute, QuantumCircuit, transpile

# ignore sparse efficiency warnings
warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)

# All matrix functions operate with Scipy spare csc as input/return type, unless specified differently

def char_to_pauli(char):
    """Returns the Pauli matrix for a given character in I, X, Y, Z"""
    char = char.upper() # convert lower to upper case
    if char == 'I':
        return np.matrix([[1,0],[0,1]])
    if char == 'X':
        return np.matrix([[0,1],[1,0]])
    if char == 'Y':
        return np.matrix([[0,-1j],[1j,0]])
    if char == 'Z':
        return np.matrix([[1,0],[0,-1]])
    raise Exception("Character '" + char + "' does not correspond to Pauli matrix")

def string_to_operator(string):
    """Calculates the operator (Kronecker product) from a Pauli string"""
    op = sparse.csc_matrix(1, dtype=np.complex128)
    for c in string:
        op = sparse.kron(op, char_to_pauli(c))
    return op

def pauli_unitary(string, coefficient):
    """Calculates the unitary matrix exp(-icH) for a given Pauli string H with coefficient c"""
    return expm(-1j * coefficient * string_to_operator(string))

def pauli_circuit(string, coefficient):
    """Quantum circuit for the given Pauli string with coefficient"""
    op = PrimitiveOp(SparsePauliOp(string, coefficient))
    unitary = PauliTrotterEvolution().convert(op.exp_i())
    return unitary.to_circuit().decompose()

def parameterized_pauli_circuit(string, parameter):
    """Quantum circuit for the given Pauli string with coefficient"""
    op = PrimitiveOp(SparsePauliOp(string, 1))
    unitary = PauliTrotterEvolution().convert((parameter*op).exp_i())
    return unitary.to_circuit().decompose()

def circuit_unitary(circuit, simulate = True):
    """Return unitary calculated from Quantum circuit
       if parameter simulate is True: simulate with Aer unitary backend
       default: obtain unitary from qiskit.Operator class
    """
    if simulate:
        backend = Aer.get_backend('unitary_simulator')
        job = execute(circuit, backend)
        result = job.result()
        u = result.get_unitary(circuit)
    else:
        u = Operator(circuit).data
    return sparse.csc_matrix(u, dtype=np.complex128)

def error(a, b, sparse = False):
    """Error between two equal sized unitaries a, b (sparse csc) defined
    as largest magnitude eigenvalue of a-b
    If sparse = True uses sparse matrix approximation (Lanczos) to find largest eigenvalue
    default is to convert to dense array first, which seems marginally faster for 10 qubits.
    """
    if sparse:
        ev = eigs(a-b, 1, which='LM', return_eigenvectors=False)
        return np.abs(ev[0])
    else:
        return np.max(np.abs(eigvals(a.toarray()-b.toarray())))
    
def permutations(iterable):
    """Returns permutations of an iterable, permuting the beginning of the iterable first.
       Does not include the original permutation.
       Based on itertools.permutations()
    """
    pool = tuple(reversed(iterable))
    n = len(pool)
    indices = list(range(n))
    cycles = list(range(n, 0, -1))
    while n:
        for i in reversed(range(n)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in reversed(indices))
                break
        else:
            return

class PauliHamiltonian:
    """Class for Hamiltonian built from Pauli strings
    Attributes:
        operators: list of Pauli strings, in order of coefficient size
        coefficients: dictionary {'Pauli String': coefficient}
        qubits: number of qbits 
    """
    
    def __init__(self, operators = [], coefficients = {}):
        nq = 0
        if len(operators):
            nq = len(operators[0])
            for o in operators:
                if len(o) != nq:
                    raise Exception("Operator length mismatch")
        self.operators = list(operators)
        self.coefficients = {}
        for o in self.operators:
            if o in coefficients.keys():
                self.coefficients[o] = coefficients[o]
            else:
                self.coefficients[o] = 0
        self.qubits = nq
        self.__matrix = None
        self.__unitary = None

    def order(self):
        """Return new PauliHamiltonian with operators in order of coefficients"""
        c = dict(self.coefficients) # copy of dictionary
        ops_ordered = []
        while len(c) > 0:
            omin = min(c, key=c.get)
            ops_ordered.append(omin)
            del c[omin]
        return PauliHamiltonian(ops_ordered, self.coefficients)
    
    def matrix(self):
        """Hamiltonian in matrix form"""
        if self.__matrix is None: # matrix form has not been calculated yet
            n = self.dim()
            self.__matrix = sparse.csc_matrix((n, n), dtype=np.complex128)
            for o in self.operators:
                self.__matrix += self.coefficients[o] * string_to_operator(o)
        return self.__matrix
    
    def unitary(self):
        """exp(-iH) in matrix form"""
        if self.__unitary is None: # unitary has not been calculated yet
            self.__unitary = expm(-1j * self.matrix())
        return self.__unitary
    
    def dim(self):
        """Matrix dimension 2^n with n qubits"""
        return 2**self.qubits
    
    def equals(self, hamiltonian):
        """Compares two objects, returns True if their coefficients and operators are equal (including order)"""
        if self.qubits == hamiltonian.qubits:
            if self.operators == hamiltonian.operators:
                for o in self.operators:
                    if self.coefficients[o] != hamiltonian.coefficients[o]:
                        return False
                return True
        return False
    
    def approx_from_unitaries(self, ops = None):
        """Unitary that approximates the full Hamiltonian evolution exp(-iH)
           calculated as matrix product of the single Pauli unitaries for each string:
           U = U1 * U2 ... * Un where Ui are in order as given by ops
           ops: list of Pauli Strings, subset of self.operators
           if ops is None: ops = self.operators
        """
        if ops is None:
            ops = self.operators
        u = sparse.eye(self.dim(), format='csc') # initialize with identity matrix
        for o in ops:
            u = u.dot(pauli_unitary(o, self.coefficients[o]))
        return u
            
    def approx_from_circuit(self, ops = None, **options):
        """Unitary that approximates the full Hamiltonian evolution exp(-iH)
           calculated as a circuit combination of the single Pauli circuits for each string.
           ops: list of Pauli Strings, subset of self.operators
           if ops is None: ops = self.operators
           simulate = True can be given as additional option to simulate with Aer
           (default is to retrieve unitary from Operator.data)
        """
        simulate = False
        if 'simulate' in options:
            simulate = options['simulate']
        if ops is None:
            ops = self.operators    
        if len(ops) < 1:
            return 0
        qc = QuantumCircuit(self.qubits)
        global_phase = 0
        # go through ops in reverse order to built circuit with rightmost unitary first
        for i in range(len(ops)-1,-1,-1):
            if ops[i] == 'I'*self.qubits:
                global_phase += self.coefficients[ops[i]]
            else:
                qc.append(pauli_circuit(ops[i], self.coefficients[ops[i]]),range(self.qubits))
        return np.exp(-1j*global_phase) * circuit_unitary(qc, simulate)
    
    def global_phase(self):
        """Prefactor exp(-ic) of global phase, c = coefficient of III...I"""
        if self.qubits:
            if 'I'*self.qubits in self.coefficients.keys():
                return np.exp(-1j*self.coefficients['I'*self.qubits])
        return 1
    
    def trotter_step(self, n):
        """Single trotter step, essentially circuit() but with 1/n coefficients
        """
        ops = self.operators    
        qc = QuantumCircuit(self.qubits)
        # go through ops in reverse order to built circuit with rightmost unitary first
        for i in range(len(ops)-1,-1,-1):
            if ops[i] != 'I'*self.qubits: # ignore identity (yields global phase)
                qc.append(pauli_circuit(ops[i], self.coefficients[ops[i]]/n),range(self.qubits))
        return transpile(qc, basis_gates=['u','cx'])
    
    def trotter_circuit(self, n = 2):
        """Repeat circuit n times to trotterize"""
        trotter = self.trotter_step(n)
        qc = QuantumCircuit(self.qubits)
        for i in range(n):
            qc.append(trotter,range(self.qubits))
        return transpile(qc, basis_gates=['u','cx'])
    
    def circuit(self, n):
        """Transpiled quantum circuit that approximates the full Hamiltonian evolution exp(-iH)
           calculated as a circuit combination of the single Pauli circuits for each string.
           ops: list of Pauli Strings, subset of self.operators
           if ops is None: ops = self.operators
        """
        qc = QuantumCircuit(self.qubits)
        # go through ops in reverse order to built circuit with rightmost unitary first
        for i in range(n):
            qc.append(self.trotter_step(n),range(self.qubits))
        return transpile(qc, basis_gates=['u','cx'])
    
    def circuit_deviation(self, unitary, n = 2):
        """Error of this Hamiltonians circuit() with respect to a different unitary
           If argument ops is provided: circuit shortened to lo that set of operators
        """
        u = self.global_phase() * circuit_unitary(self.circuit(n))
        return error(u, unitary)
    
    def save_circuit(self, filename, ops = None):
        qc = self.circuit(ops)
        try:
            file = open(filename, 'x') # read file
        except:
            print("An exception occurred trying to create the file: " + filename)
            sys.stdout.flush()
        file.write(qc.qasm())
        file.close()
        
    def save_to_file(self, filename):
        """Save to file in format 'coefficient Pauli-string' per line"""
        try:
            file = open(filename, 'x') # read file
        except:
            print("An exception occurred trying to create the file: " + filename) 
            sys.stdout.flush()
        for o in self.operators:
            file.write(str(self.coefficients[o])+" "+o+"\n")
        file.close()
        
    def crop(self, n = 1):
        """Returns a new PauliHamiltonian with the last n (default: 1) Pauli strings omitted"""
        return PauliHamiltonian(self.operators[:-n], self.coefficients)
    
    def shorten_circuit(self, unitary, n = 2, limit = 0.1):
        """Try if any of the operators can be omitted, keeping the error below the limit
           Returns the Hamiltonian shortened by the one operator that results in the lowest error
           (return same Hamiltonian if no shorter list of operators can be found)
        """
        initial_error = self.circuit_deviation(unitary, n)
        err = limit
        remove_op = None
        for o in self.operators:
            if o != 'I'*self.qubits: # ignore identity
                ops = list(self.operators)
                ops.remove(o)
                hnew = PauliHamiltonian(ops, self.coefficients) # list of operators without o
                # calculate weighted error
                e = hnew.circuit_deviation(unitary, n)
                ew = (e - initial_error) / pauli_circuit(o,1).depth()
                if ew < err and e < limit:
                    err = ew
                    remove_op = o
                if ew < 0:
                    break # if removing an operator reduces the error, remove immediately
        if remove_op is None:
            return self
        ops = list(self.operators)
        ops.remove(remove_op)
        return PauliHamiltonian(ops, self.coefficients)
    
    def optimize_coefficients(self, unitary, n = 2, verbose = True):
        """Attempts to optimize the circuit error (w.r.t. unitary) by minimizing over all coefficients"""
        copt = dict(self.coefficients) # takes optimized coefficients
        phase = self.global_phase()
        nq = self.qubits
        ops = list(self.operators)
        u = phase * circuit_unitary(self.circuit(n))
        err = error(u, unitary) # initial error
        for o in ops:
            if o != 'I'*nq: # ignore identity
                qc = QuantumCircuit(nq)
                trot = QuantumCircuit(nq)
                for i in range(len(ops)-1,-1,-1):
                    if ops[i] == o: # parametrize opertor o
                        param = Parameter('param')
                        trot.append(parameterized_pauli_circuit(ops[i], param),range(nq))
                    elif ops[i] != 'I'*nq: # ignore identity (yields global phase)
                        trot.append(pauli_circuit(ops[i], copt[ops[i]]),range(nq))
                for i in range(n):
                    qc.append(trot,range(nq))
                def f(p):
                    u = phase * circuit_unitary(qc.bind_parameters({param: float(p)}))
                    return error(u, unitary)
                if verbose:
                    print("    Minimizing coefficient for operator ",o)
                    sys.stdout.flush()
                m = minimize(f, copt[o], method='COBYLA', tol=0.001)
                if m.fun < err:
                    copt[o] = m.x
                    err = m.fun
                    if verbose:
                        print("      New minimum found: ",err)
                        print("      New coefficient:   ",copt[o])
                        sys.stdout.flush()
        return PauliHamiltonian(ops, copt)
    
    def shortest_circuit(self, unitary, n = 2, dump_prefix = 'h-dump', verbose = True):
        """Finds the shortest circuit by iteratively minimizing coefficients and cropping operators"""
        h = self
        count = 0
        need_optimization = True # set to true if optimization already done on current Hamiltonian
        while True:
            count += 1
            if verbose:
                print("Entering optimization cycle ",count)
                print("  Shortening circuit:")
                sys.stdout.flush()
            shorter_found = False
            further_seach = True
            while further_seach:
                hnew = h.shorten_circuit(unitary,n)
                if h.equals(hnew):
                    further_seach = False
                else:
                    if verbose:
                        print("    Found shorter circuit with depth ",hnew.circuit(n).depth(),
                              " and error ",hnew.circuit_deviation(unitary,n))
                        sys.stdout.flush()
                    h = hnew
                    h.save_to_file(dump_prefix+"-short-"+time.strftime("%Y%m%d-%H%M%S"+".txt"))
                    need_optimization = True # new circuit needs optimization again
            if need_optimization:
                if verbose:
                    print("  Optimizing coefficients:")
                    sys.stdout.flush()
                hnew = h.optimize_coefficients(unitary, n, verbose)
                if h.equals(hnew): # no optimization possible
                    need_optimization = False 
                else:
                    if verbose:
                        print("    Found optimized circuit with error ",hnew.circuit_deviation(unitary, n))
                        sys.stdout.flush()
                    h = hnew
                    h.save_to_file(dump_prefix+"-opt-"+time.strftime("%Y%m%d-%H%M%S"+".txt"))
            else: # No further need for optimization, and shortest circut found, return
                if verbose:
                    print("No further optimization found. Quitting.")
                    sys.stdout.flush()
                return h           

    
            
def hamiltonian_from_file(filename):
    """Read Hamiltonian from file and return PauliHamiltonian object.
    File format: 'coefficient Pauli-string' per line, separated by space"""
    try:
        file = open(filename) # read file
    except:
        print("An exception occurred trying to read the file: " + filename) 
    lines = file.read().split('\n')
    file.close()
    h = PauliHamiltonian() # create object to return
    for l in lines:
        if len(l): # exclude empty lines
            [c,o] = l.split() # split in coefficient c and operator string o
            if h.qubits:
                if len(o) != h.qubits: # Make sure all operators have the same dimension
                    raise Exception("Operator string '" + o + "' does not match dimension (" + h.qubits + ")")
            else:
                h.qubits = len(o) # set no. qubits to length of operator string
            h.operators.append(o) # add operator to list of operators of object h
            h.coefficients[o] = float(c) # add to the dictionary of coefficients for h
    return h

if __name__== "__main__":
    n = 2
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    # order Hamiltonian (already done in hamiltionian_ordered.txt):
    # h_unordered = hamiltonian_from_file("hamiltonian.txt")
    # h = h_unordered.order()

    print("Loading Hamiltonian")
    sys.stdout.flush()
    h = hamiltonian_from_file("hamiltonian_ordered.txt") # initial Hamiltonian, already ordered by coefficient magnitude
    u = h.unitary() # store the unitary operator used for comparison

    print("Starting optimization for trotter length ",n)
    sys.stdout.flush()
    h0 = h.crop(52) # try a reasonable guess as compromise between leaving some Pauli strings with small coefficients out but leaving enough room for optimization

    h1 = h0.shortest_circuit(u,n,'ht'+str(n)+'-dump')
    h1.save_to_file("h1.txt")
    h1.save_circuit("c1.txt")
