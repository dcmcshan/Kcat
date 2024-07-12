import math
from typing import List
from collections import Counter
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO, SeqIO

def perform_multiple_sequence_alignment(sequences: List[str]) -> MultipleSeqAlignment:
    """
    Perform multiple sequence alignment on a list of sequences.

    Args:
        sequences (List[str]): List of protein sequences.

    Returns:
        MultipleSeqAlignment: Aligned sequences.
    """
    # Create SeqRecord objects for each sequence
    seq_records = [SeqRecord(Seq(seq), id=str(i)) for i, seq in enumerate(sequences)]
    
    # Write sequences to a temporary fasta file
    with open("temp_sequences.fasta", "w") as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")
    
    # Run Clustal Omega to perform the multiple sequence alignment
    clustalomega_cline = ClustalOmegaCommandline(infile="temp_sequences.fasta", outfile="aligned_sequences.fasta", verbose=True, auto=True, force=True)
    clustalomega_cline()
    
    # Read the aligned sequences
    alignment = AlignIO.read("aligned_sequences.fasta", "fasta")
    
    return alignment

def calculate_conservation_scores(alignment: MultipleSeqAlignment) -> List[float]:
    """
    Calculate conservation scores for each position in the alignment.

    Args:
        alignment (MultipleSeqAlignment): Aligned sequences.

    Returns:
        List[float]: Conservation scores for each position.
    """
    num_sequences = len(alignment)
    alignment_length = alignment.get_alignment_length()
    conservation_scores = []

    for i in range(alignment_length):
        column = alignment[:, i]
        counts = Counter(column)
        entropy = sum(-(count/num_sequences) * math.log2(count/num_sequences) for count in counts.values() if count > 0)
        conservation = 1 - (entropy / math.log2(min(20, num_sequences)))  # 20 is the number of amino acid types
        conservation_scores.append(conservation)

    return conservation_scores

# Example usage
sequences = [
    "MAGWAGGS",
    "MAGWVGGS",
    "MAGWGGGS"
]

alignment = perform_multiple_sequence_alignment(sequences)
conservation_scores = calculate_conservation_scores(alignment)

print("Alignment:")
for record in alignment:
    print(record.seq)

print("Conservation Scores:")
print(conservation_scores)
