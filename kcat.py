import logging
import math
from typing import List, Tuple

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

def download_pdb_files(file_path: str) -> None:
    """
    Download PDB files based on PDB IDs from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing PDB IDs.
    """
    logger.info(f"Reading CSV file: {file_path}")
    df = pd.read_csv(file_path)
    pdb_ids = df['PDBID'].dropna().unique().tolist()

    for pdb_id in pdb_ids:
        try:
            logger.info(f"Downloading PDB file: {pdb_id}")
            pdb_file = get_pdb_file(pdb_id, filetype="pdb", compression=False)
            if pdb_file:
                with open(f"{pdb_id}.pdb", "w") as f:
                    f.write(pdb_file)
                logger.info(f"Successfully downloaded {pdb_id}")
            else:
                logger.warning(f"Failed to download {pdb_id}")
        except Exception as e:
            logger.error(f"Error downloading {pdb_id}: {e}")

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

def main():
    # Example usage
    pdb_file_path = '2jk2.pdb'
    target_residue_name = "LYS"
    target_residue_number = 13

    # Parse PDB file
    parse_pdb_file(pdb_file_path)

    # Calculate dihedral angles
    dihedral_features = calculate_dihedrals(pdb_file_path, target_residue_name, target_residue_number)

    # Process dihedral data
    dihedral_features_df, angle_df = process_dihedral_data(dihedral_features)

    logger.info("Dihedral features:")
    logger.info(dihedral_features_df.head())
    logger.info("Angle data:")
    logger.info(angle_df.head())

if __name__ == "__main__":
    main()