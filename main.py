import logging
from typing import List
import pandas as pd
from data_handling import extract_data_from_csv
from pdb_processing import download_pdb_files, get_sequence_from_pdb, analyze_structure_kcat
from sequence_analysis import perform_multiple_sequence_alignment, calculate_conservation_scores
from ml_models import prepare_ml_data, train_and_evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    csv_file_path = 'kcat_km_str_clean_filtered.csv'
    output_dir = "pdb_files"

    # Extract data from CSV
    df = extract_data_from_csv(csv_file_path)
    total_pdb_count = len(df)

    # Download PDB files
    pdb_file_paths = download_pdb_files(df['PDBID'].tolist(), output_dir)
    downloaded_pdb_count = len(pdb_file_paths)

    # Filter out entries for which PDB files couldn't be downloaded
    df_filtered = df[df['PDBID'].isin(pdb_file_paths.keys())]
    logger.info(f"Proceeding with analysis for {len(df_filtered)} out of {total_pdb_count} original entries")

    # Extract sequences and perform multiple sequence alignment
    sequences = [get_sequence_from_pdb(pdb_file) for pdb_file in pdb_file_paths.values()]
    alignment = perform_multiple_sequence_alignment(sequences)

    # Calculate conservation scores
    conservation_scores = calculate_conservation_scores(alignment)

    # Analyze structures and kcat_mut values
    results = []
    total_active_sites = 0
    total_angles_computed = 0
    for _, row in df_filtered.iterrows():
        pdb_id = row['PDBID']
        result = analyze_structure_kcat(pdb_id, pdb_file_paths[pdb_id], row['kcat_mut'], conservation_scores)
        if result:
            results.append(result)
            total_active_sites += len(result['predicted_active_site'])
            total_angles_computed += len(result['dihedral_data'])

    # Prepare data for machine learning
    X, y = prepare_ml_data(results)

    # Train and evaluate Random Forest model
    rf_model, rf_metrics = train_and_evaluate_model(X, y, model_type='rf')
    logger.info("Random Forest Model Metrics:")
    logger.info(f"RMSE: {rf_metrics['rmse']}")
    logger.info(f"R2 Score: {rf_metrics['r2']}")

    # Train and evaluate Gradient Boosting model
    gb_model, gb_metrics = train_and_evaluate_model(X, y, model_type='gb')
    logger.info("Gradient Boosting Model Metrics:")
    logger.info(f"RMSE: {gb_metrics['rmse']}")
    logger.info(f"R2 Score: {gb_metrics['r2']}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Total PDB entries in CSV: {total_pdb_count}")
    print(f"PDB files successfully downloaded: {downloaded_pdb_count}")
    print(f"Total active sites found: {total_active_sites}")
    print(f"Total dihedral angles computed: {total_angles_computed}")
    print("\nMachine Learning Results:")
    print("Random Forest:")
    print(f"  RMSE: {rf_metrics['rmse']:.4f}")
    print(f"  R2 Score: {rf_metrics['r2']:.4f}")
    print("Gradient Boosting:")
    print(f"  RMSE: {gb_metrics['rmse']:.4f}")
    print(f"  R2 Score: {gb_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()