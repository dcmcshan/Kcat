import logging
import math
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder, calc_dihedral
from pypdb import get_pdb_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_pdb_file(pdb_file_path: str) -> None:
    """
    Parse a PDB file and print sequence and residue information.

    Args:
        pdb_file_path (str): Path to the PDB file.
    """
    logger.info(f"Parsing PDB file: {pdb_file_path}")
    p = PDBParser()
    structure = p.get_structure("structure", pdb_file_path)
    ppb = PPBuilder()

    for pp in ppb.build_peptides(structure):
        logger.info(f"Sequence: {pp.get_sequence()}")
        for residue in pp:
            resname = residue.get_resname()
            resnum = residue.get_id()[1]
            logger.debug(f"Residue: {resname} {resnum}")

        seq = pp.get_sequence().__str__()
        logger.info(f"Last sequence as string: {seq}")

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

def calculate_dihedrals(file_path: str, target_residue_name: str, target_residue_number: int) -> List[List[float]]:
    """
    Calculate dihedral angles for a specific residue in a PDB structure.

    Args:
        file_path (str): Path to the PDB file.
        target_residue_name (str): Name of the target residue.
        target_residue_number (int): Number of the target residue.

    Returns:
        List[List[float]]: List of dihedral angle features [cos(phi), cos(psi), sin(phi), sin(psi)].
    """
    logger.info(f"Calculating dihedrals for {target_residue_name}{target_residue_number} in {file_path}")
    parser = PDBParser()
    structure = parser.get_structure('Enzyme', file_path)
    model = structure[0]
    dihedral_data = []

    for chain in model:
        ppb = PPBuilder()
        for pp in ppb.build_peptides(chain):
            for i, res in enumerate(pp):
                if (res.get_resname() == target_residue_name and
                    res.id[1] == target_residue_number):
                    if i > 0 and i < len(pp) - 1:
                        phi = calc_dihedral(pp[i-1]['C'].get_vector(),
                                            pp[i]['N'].get_vector(),
                                            pp[i]['CA'].get_vector(),
                                            pp[i]['C'].get_vector())
                        psi = calc_dihedral(pp[i]['N'].get_vector(),
                                            pp[i]['CA'].get_vector(),
                                            pp[i]['C'].get_vector(),
                                            pp[i+1]['N'].get_vector())
                        if phi is not None and psi is not None:
                            dihedral_data.append([np.cos(phi), np.cos(psi), np.sin(phi), np.sin(psi)])
                            logger.debug(f"Calculated dihedrals: phi={phi}, psi={psi}")

    return dihedral_data

def process_dihedral_data(dihedral_features: List[List[float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process dihedral angle data and convert to DataFrames.

    Args:
        dihedral_features (List[List[float]]): List of dihedral angle features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for trigonometric features and angle data.
    """
    logger.info("Processing dihedral data")
    dihedral_features_df = pd.DataFrame(dihedral_features, columns=['cos_phi', 'cos_psi', 'sin_phi', 'sin_psi'])
    
    angle_data = []
    for cos_phi, cos_psi, sin_phi, sin_psi in dihedral_features:
        phi = math.degrees(math.atan2(sin_phi, cos_phi))
        psi = math.degrees(math.atan2(sin_psi, cos_psi))
        angle_data.append([phi, psi])
    
    angle_df = pd.DataFrame(angle_data, columns=['Phi', 'Psi'])
    
    return dihedral_features_df, angle_df

def extract_kcat_values(file_path: str) -> Dict[str, float]:
    """
    Extract kcat values from the CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Dict[str, float]: Dictionary mapping PDB IDs to kcat values.
    """
    logger.info(f"Extracting kcat values from {file_path}")
    df = pd.read_csv(file_path)
    kcat_dict = dict(zip(df['PDBID'], df['kcat']))
    logger.info(f"Extracted {len(kcat_dict)} kcat values")
    return kcat_dict

def analyze_structure_kcat(pdb_id: str, pdb_file_path: str, kcat: float, target_residue_name: str, target_residue_number: int) -> Dict:
    """
    Analyze a single PDB structure and its kcat value.

    Args:
        pdb_id (str): PDB ID of the structure.
        pdb_file_path (str): Path to the PDB file.
        kcat (float): kcat value for the enzyme.
        target_residue_name (str): Name of the target residue.
        target_residue_number (int): Number of the target residue.

    Returns:
        Dict: Analysis results including PDB ID, kcat, and dihedral angles.
    """
    dihedral_features = calculate_dihedrals(pdb_file_path, target_residue_name, target_residue_number)
    
    if not dihedral_features:
        logger.warning(f"No dihedral features found for {pdb_id}")
        return None

    dihedral_features_df, angle_df = process_dihedral_data(dihedral_features)
    
    result = {
        'PDB_ID': pdb_id,
        'kcat': kcat,
        'Phi': angle_df['Phi'].iloc[0],
        'Psi': angle_df['Psi'].iloc[0],
        'cos_phi': dihedral_features_df['cos_phi'].iloc[0],
        'cos_psi': dihedral_features_df['cos_psi'].iloc[0],
        'sin_phi': dihedral_features_df['sin_phi'].iloc[0],
        'sin_psi': dihedral_features_df['sin_psi'].iloc[0]
    }
    
    return result

def main():
    csv_file_path = 'kcat_km_str_clean_filtered.csv'
    target_residue_name = "LYS"
    target_residue_number = 13
    output_dir = "pdb_files"

    # Extract kcat values
    kcat_dict = extract_kcat_values(csv_file_path)

    # Download PDB files
    pdb_file_paths = download_pdb_files(list(kcat_dict.keys()), output_dir)

    # Analyze structures and kcat values
    results = []
    for pdb_id, kcat in kcat_dict.items():
        if pdb_id in pdb_file_paths:
            result = analyze_structure_kcat(pdb_id, pdb_file_paths[pdb_id], kcat, target_residue_name, target_residue_number)
            if result:
                results.append(result)
        else:
            logger.warning(f"PDB file for {pdb_id} not found, skipping analysis")

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    logger.info("Analysis results:")
    logger.info(results_df.head())

    # Save results to CSV
    output_file = 'structure_kcat_analysis.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()