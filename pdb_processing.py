import os
import logging
import math
from typing import List, Dict, Tuple
from Bio.PDB import PDBParser, DSSP, calc_dihedral
from Bio.PDB.Polypeptide import protein_letters_3to1
from pypdb.clients.pdb import pdb_client

logger = logging.getLogger(__name__)

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
    failed_downloads = []
    
    for pdb_id in pdb_ids:
        # Remove any whitespace and convert to uppercase
        pdb_id = pdb_id.strip().upper()
        
        # Check if the PDB ID is valid (4 characters)
        if len(pdb_id) != 4:
            logger.warning(f"Invalid PDB ID: {pdb_id}. Skipping.")
            failed_downloads.append(pdb_id)
            continue
        
        file_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        if os.path.exists(file_path):
            logger.info(f"PDB file for {pdb_id} already exists, skipping download")
            pdb_file_paths[pdb_id] = file_path
            continue
        
        try:
            logger.info(f"Downloading PDB file: {pdb_id}")
            pdb_file = pdb_client.get_pdb_file(pdb_id, filetype="pdb", compression=False)
            if pdb_file:
                with open(file_path, "w") as f:
                    f.write(pdb_file)
                logger.info(f"Successfully downloaded {pdb_id}")
                pdb_file_paths[pdb_id] = file_path
            else:
                logger.warning(f"Failed to download {pdb_id}. PDB file not found.")
                failed_downloads.append(pdb_id)
        except Exception as e:
            logger.error(f"Error downloading {pdb_id}: {str(e)}")
            failed_downloads.append(pdb_id)
    
    logger.info(f"Downloaded {len(pdb_file_paths)} PDB files")
    if failed_downloads:
        logger.warning(f"Failed to download {len(failed_downloads)} PDB files: {', '.join(failed_downloads)}")
    
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
                    try:
                        sequence += protein_letters_3to1[residue.get_resname()]
                    except KeyError:
                        # If the residue is not a standard amino acid, skip it
                        logger.warning(f"Non-standard residue {residue.get_resname()} found in {pdb_file}. Skipping.")
                        continue
    return sequence

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

def calculate_residue_dihedrals(residue, chain) -> Tuple[float, float]:
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    pdb_ids = ["1A2B", "3C4D"]  # Replace with actual PDB IDs
    conservation_scores = [0.85, 0.9, 0.7, 0.95]  # Example conservation scores
    kcat_mut = 1.5