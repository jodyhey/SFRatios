from Bio import SeqIO
import random
from collections import Counter

rate_matrix = [[0.0,0.066,0.135,0.129],
               [0.121,0.0,0.071,0.478],
               [0.478,0.071,0.0,0.121],
               [0.129,0.135,0.066,0.0]]

# Build list of possible changes and cumulative rates
bases = ['A', 'C', 'G', 'T']
total = 0
changes = [] # List of tuples (from_base, to_base, cumulative_rate)
for i in range(4):
    for j in range(4):
        if i != j: # Skip diagonal
            rate = rate_matrix[i][j]
            total += rate
            changes.append((bases[i], bases[j], total))


AminoAcids3to1 = {
    "CYS": "C",
    "ASP": "D", 
    "SER": "S", 
    "GLN": "Q",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "LYS": "K",
    "THR": "T", 
    "PHE": "F", 
    "ALA": "A", 
    "GLY": "G", 
    "ILE": "I", 
    "LEU": "L", 
    "HIS": "H", 
    "ARG": "R",
    "TRP": "W",
    "VAL": "V", 
    "GLU": "E", 
    "TYR": "Y",
    "STOP": "*"
    }

AA3toSynonymousCodons = {
    "CYS": ["TGT", "TGC"],
    "ASP": ["GAT", "GAC"],
    "SER": ["TCT", "TCG", "TCA", "TCC", "AGC", "AGT"],
    "GLN": ["CAA", "CAG"],
    "MET": ["ATG"],
    "ASN": ["AAC", "AAT"],
    "PRO": ["CCT", "CCG", "CCA", "CCC"],
    "LYS": ["AAG", "AAA"],
    "THR": ["ACC", "ACA", "ACG", "ACT"],
    "PHE": ["TTT", "TTC"],
    "ALA": ["GCA", "GCC", "GCG", "GCT"],
    "GLY": ["GGT", "GGG", "GGA", "GGC"],
    "ILE": ["ATC", "ATA", "ATT"],
    "LEU": ["TTA", "TTG", "CTC", "CTT", "CTG", "CTA"],
    "HIS": ["CAT", "CAC"],
    "ARG": ["CGA", "CGC", "CGG", "CGT", "AGG", "AGA"],
    "TRP": ["TGG"],
    "VAL": ["GTA", "GTC", "GTG", "GTT"],
    "GLU": ["GAG", "GAA"],
    "TYR": ["TAT", "TAC"],
    "STOP": ["TAG", "TGA", "TAA"]}

AminoAcids1to3 = {value.lower(): key for key, value in AminoAcids3to1.items()}
SynonymousCodonstoAA3 = {item: key for key, value_list in AA3toSynonymousCodons.items() for item in value_list}
SynonymousCodonstoAA1 = {key: AminoAcids3to1[value].lower() for key, value in SynonymousCodonstoAA3.items()}

def simmut():
   """
   Pick a random mutation based on relative rates in rate_matrix.
   rate_matrix: 4x4 matrix of relative rates [A,C,G,T]
   Returns: (from_base, to_base) tuple
   """
   # Pick random number between 0 and total rate
   r = random.random() * total
   
   # Find first change where cumulative rate exceeds r
   for from_base, to_base, cum_rate in changes:
       if r <= cum_rate:
           return from_base, to_base
           
   # Should never reach here if matrix properly normalized
   return changes[-1][0], changes[-1][1]

def check_mutation_type(original_codon, mutated_codon):
    """Determine if mutation is synonymous, nonsynonymous, or nonsense"""
    # Genetic code dictionary
    if SynonymousCodonstoAA1[original_codon] != "*":
        if SynonymousCodonstoAA1[mutated_codon] == "*":
            return "nonsense"
        elif SynonymousCodonstoAA1[original_codon] == SynonymousCodonstoAA1[mutated_codon]:
            return "synonymous"
        else:
            return "nonsynonymous"
    else:
        return None

def check_sequences(sequences):
   """
   Check sequences for valid coding sequences:
   - Length must be multiple of 3
   - No internal stop codons (only at end)
   Returns list of valid sequences
   """
   valid_sequences = []
   
   for seq in sequences:
       sequence = str(seq.seq)
       # Check length
       if len(sequence) % 3 != 0:
           continue
           
       # Check for internal stops
       valid = True
       for i in range(0, len(sequence)-3, 3):
           codon = sequence[i:i+3]
           if codon not in SynonymousCodonstoAA1:
               valid = False
               break
           if SynonymousCodonstoAA1[codon] == '*':
               valid = False
               break
               
       # Check final codon
       final_codon = sequence[-3:]
       if SynonymousCodonstoAA1[final_codon] != '*':
           valid = False
           
       if valid:
           valid_sequences.append(seq)
           
   return valid_sequences
def main(k):
    # Read all sequences
    sequences = list(SeqIO.parse("/mnt/d/genemod/better_dNdS_models/popgen/prfratio/drosophila/drosophila_dm6_Autosomes_coding_sequences.txt", "fasta"))
    sequences = check_sequences(sequences)

    # Counter for mutation types
    mutation_counts = Counter()
    
    for _ in range(k):
        # Select random sequence
        seq = random.choice(sequences)
        sequence = str(seq.seq)
        
        # select random mutation
        orig_base, mut_base = simmut()
        # Select random position,  keep picking until the base matches orig_base
        while True:
            pos = random.randint(0, len(sequence) - 1)
            if sequence[pos] == orig_base:
                break
        
        # Get codon position and full codon
        codon_pos = pos % 3
        codon_start = pos - codon_pos
        original_codon = sequence[codon_start:codon_start + 3]
        
       
        # Create mutated codon
        mutated_codon = (original_codon[:codon_pos] + 
                        mut_base + 
                        original_codon[codon_pos + 1:])
        
        # Check mutation type
        mutation_type = check_mutation_type(original_codon, mutated_codon)
        if mutation_type != None:
            mutation_counts[mutation_type] += 1
    
    # Calculate proportions
    total = sum(mutation_counts.values())
    print("\nAfter", k, "mutations:")
    for mut_type, count in mutation_counts.items():
        proportion = count / total
        print(f"{mut_type}: {proportion:.3f} ({count} mutations)")


if __name__ == "__main__":
    k = 100000  # Number of mutations to simulate
    main(k)