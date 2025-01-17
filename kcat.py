import logging
import math
import os
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO, Align
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from pypdb import get_pdb_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Extract relevant data from the CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing PDB IDs and kcat_mut values.
    """
    logger.info(f"Extracting data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Extracted data for {len(df)} structures")
    return df

def download_pdb_files(pdb_ids: List[str], output_dir: str = "pdb_files") -> Dict[str, str]:
    """
    Download PDB files for given PDB IDs.

    Args:
        pdb_ids (List[str]): List of PDB IDs to download.
        output_dir (str): Directory to save PDB files.

    Returns:
        Dict[str, str]: Dictionary mapping PDB IDs to their file paths.
    """
    logger.info(f"Downloading PDB files for {len(pdb_ids)} structures")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdb_file_paths = {}
    for pdb_id in pdb_ids:
        file_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        if os.path.exists(file_path):
            logger.info(f"PDB file for {pdb_id} already exists, skipping download")
            pdb_file_paths[pdb_id] = file_path
            continue
        
        try:
            logger.info(f"Downloading PDB file: {pdb_id}")
            pdb_file = get_pdb_file(pdb_id, filetype="pdb", compression=False)
            if pdb_file:
                with open(file_path, "w") as f:
                    f.write(pdb_file)
                logger.info(f"Successfully downloaded {pdb_id}")
                pdb_file_paths[pdb_id] = file_path
            else:
                logger.warning(f"Failed to download {pdb_id}")
        except Exception as e:
            logger.error(f"Error downloading {pdb_id}: {e}")
    
    logger.info(f"Downloaded {len(pdb_file_paths)} PDB files")
    return pdb_file_paths

def get_sequence_from_pdb(pdb_file: str) -> str:
    """
    Extract protein sequence from a PDB file.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        str: Protein sequence.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # Check if it's a standard amino acid
                    sequence += Seq.IUPAC.protein_letters_3to1[residue.get_resname()]
    return sequence

def perform_multiple_sequence_alignment(sequences: List[str]) -> MultipleSeqAlignment:
    """
    Perform multiple sequence alignment on a list of sequences.

    Args:
        sequences (List[str]): List of protein sequences.

    Returns:
        MultipleSeqAlignment: Aligned sequences.
    """
    seq_records = [SeqRecord(Seq(seq), id=str(i)) for i, seq in enumerate(sequences)]
    aligner = Align.MultipleSeqAlignment()
    aligned = aligner.align(seq_records)
    return aligned

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

def predict_active_site_residues(pdb_file: str, conservation_scores: List[float], threshold: float = 0.8) -> List[str]:
    """
    Predict active site residues based on conservation scores and structural properties.

    Args:
        pdb_file (str): Path to the PDB file.
        conservation_scores (List[float]): Conservation scores for each residue.
        threshold (float): Conservation score threshold for considering a residue as potentially active.

    Returns:
        List[str]: Predicted active site residues in the format "chain:residue_number".
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    dssp = DSSP(structure[0], pdb_file, dssp='mkdssp')

    active_site_candidates = []
    for residue in structure.get_residues():
        if residue.get_id()[0] == " ":  # Check if it's a standard amino acid
            chain = residue.get_parent().id
            res_id = residue.get_id()[1]
            res_key = (chain, (' ', res_id, ' '))

            if res_key in dssp.keys():
                conservation = conservation_scores[res_id - 1]  # Assuming 1-based residue numbering
                accessibility = dssp[res_key][3]
                if conservation >= threshold and accessibility > 0.2:  # Consider exposed and conserved residues
                    active_site_candidates.append(f"{chain}:{res_id}")

    return active_site_candidates

def calculate_dihedrals_for_residues(structure, residues):
    """
    Calculate dihedral angles for specified residues in a structure.

    Args:
        structure (Bio.PDB.Structure.Structure): The protein structure.
        residues (List[str]): List of residue identifiers (e.g., "A:123").

    Returns:
        Dict[str, Dict]: Dictionary mapping residue identifiers to their dihedral angles.
    """
    logger.info(f"Calculating dihedrals for {len(residues)} residues")
    dihedral_data = {}
    
    for res_id in residues:
        chain_id, res_num = res_id.split(':')
        res_num = int(res_num)
        
        if chain_id in structure[0] and res_num in structure[0][chain_id]:
            residue = structure[0][chain_id][res_num]
            phi, psi = calculate_residue_dihedrals(residue, structure[0][chain_id])
            
            if phi is not None and psi is not None:
                dihedral_data[res_id] = {
                    'Phi': math.degrees(phi),
                    'Psi': math.degrees(psi),
                    'cos_phi': math.cos(phi),
                    'cos_psi': math.cos(psi),
                    'sin_phi': math.sin(phi),
                    'sin_psi': math.sin(psi)
                }
            else:
                logger.warning(f"Could not calculate dihedrals for residue {res_id}")
        else:
            logger.warning(f"Residue {res_id} not found in structure")
    
    return dihedral_data

def calculate_residue_dihedrals(residue, chain):
    """
    Calculate phi and psi angles for a single residue.

    Args:
        residue (Bio.PDB.Residue): The residue to calculate angles for.
        chain (Bio.PDB.Chain): The chain containing the residue.

    Returns:
        Tuple[float, float]: Phi and Psi angles in radians.
    """
    phi, psi = None, None
    
    res_id = residue.get_id()
    prev_res = chain.child_list[chain.child_list.index(residue) - 1]
    next_res = chain.child_list[chain.child_list.index(residue) + 1]
    
    if prev_res.get_id()[0] == ' ':  # Check if it's a standard amino acid
        phi = calc_dihedral(prev_res['C'].get_vector(),
                            residue['N'].get_vector(),
                            residue['CA'].get_vector(),
                            residue['C'].get_vector())
    
    if next_res.get_id()[0] == ' ':  # Check if it's a standard amino acid
        psi = calc_dihedral(residue['N'].get_vector(),
                            residue['CA'].get_vector(),
                            residue['C'].get_vector(),
                            next_res['N'].get_vector())
    
    return phi, psi

def analyze_structure_kcat(pdb_id: str, pdb_file_path: str, kcat_mut: float, conservation_scores: List[float]):
    """
    Analyze a single PDB structure, its kcat_mut value, and predict active site residues.

    Args:
        pdb_id (str): PDB ID of the structure.
        pdb_file_path (str): Path to the PDB file.
        kcat_mut (float): kcat_mut value for the enzyme.
        conservation_scores (List[float]): Conservation scores for each residue.

    Returns:
        Dict: Analysis results including PDB ID, kcat_mut, predicted active site residues, and dihedral angles.
    """
    logger.info(f"Analyzing structure {pdb_id}")
    parser = PDBParser()
    structure = parser.get_structure('Enzyme', pdb_file_path)
    
    predicted_active_site = predict_active_site_residues(pdb_file_path, conservation_scores)
    dihedral_data = calculate_dihedrals_for_residues(structure, predicted_active_site)
    
    result = {
        'PDB_ID': pdb_id,
        'kcat_mut': kcat_mut,
        'predicted_active_site': predicted_active_site,
        'dihedral_data': dihedral_data
    }
    
    return result

def prepare_ml_data(results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning.

    Args:
        results (List[Dict]): List of analysis results for each structure.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features) and y (target) arrays for ML.
    """
    X = []
    y = []
    for result in results:
        features = []
        for res_data in result['dihedral_data'].values():
            features.extend([res_data['cos_phi'], res_data['cos_psi'], res_data['sin_phi'], res_data['sin_psi']])
        if features:  # Only include if we have features (i.e., dihedral angles were calculated)
            X.append(features)
            y.append(result['kcat_mut'])
    
    return np.array(X), np.array(y)

def train_and_evaluate_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf'):
    """
    Train and evaluate a machine learning model.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Target array.
        model_type (str): Type of model to use ('rf' for Random Forest, 'gb' for Gradient Boosting).

    Returns:
        Tuple[Pipeline, Dict]: Trained model pipeline and evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'rf' or 'gb'.")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }

    return pipeline, metrics

def main():
    csv_file_path = 'kcat_km_str_clean_filtered.csv'
    output_dir = "pdb_files"

    # Extract data from CSV
    df = extract_data_from_csv(csv_file_path)

    # Download PDB files
    pdb_file_paths = download_pdb_files(df['PDBID'].tolist(), output_dir)

    # Extract sequences and perform multiple sequence alignment
    sequences = [get_sequence_from_pdb(pdb_file) for pdb_file in pdb_file_paths.values()]
    alignment = perform_multiple_sequence_alignment(sequences)

    # Calculate conservation scores
    conservation_scores = calculate_conservation_scores(alignment)

    # Analyze structures and kcat_mut values
    results = []
    for _, row in df.iterrows():
        pdb_id = row['PDBID']
        if pdb_id in pdb_file_paths:
            result = analyze_structure_kcat(pdb_id, pdb_file_paths[pdb_id], row['kcat_mut'], conservation_scores)
            if result:
                results.append(result)
        else:
            logger.warning(f"PDB file for {pdb_id} not found, skipping analysis")

    # Prepare data for machine learning
    X, y = prepare_ml_data(results)

    # Train and evaluate Random Forest model
    rf_model, rf_metrics = train_and_evaluate_model(X, y,