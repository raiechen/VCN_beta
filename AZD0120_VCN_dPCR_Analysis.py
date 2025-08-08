# Import the necessary libraries
import streamlit as st
import pandas as pd
import io # Use io to handle the uploaded file bytes
from io import StringIO
import numpy as np
import datetime # For timestamping the export file
import re # For parsing Input ID

# Define global constants for analysis thresholds and parameters
MIN_VALID_PARTITION = 7000
NTC_POS_PART_CUTOFF = 5
MAX_CV_PERCENTAGE = 20.0  # CV threshold as percentage (21%)
LLOQ_WPRE_COPIES_UL = 45.49
LLOQ_RPP30_COPIES_UL = 4.99
# CAR_PC constants will be set by user input
# Default values for session state initialization
DEFAULT_CAR_PC_MIN_COPY_NUMBER_CELL = 3.35
DEFAULT_CAR_PC_MAX_COPY_NUMBER_CELL = 3.72

def add_designation_column(summary_df, original_df, min_valid_partition, ntc_pos_part_cutoff, max_cv_percentage, lloq_wpre_copies_ul, lloq_rpp30_copies_ul):
    """
    Adds a 'Designation' column to the summary_df based on several criteria.

    Args:
        summary_df (pd.DataFrame): The summary table.
        original_df (pd.DataFrame): The original uploaded data.
        min_valid_partition (int): Threshold for MIN_VALID_PARTITION.
        ntc_pos_part_cutoff (int): Threshold for NTC_POS_PART_CUTOFF.
        max_cv_percentage (float): Threshold for MAX_CV_PERCENTAGE (as percentage, e.g., 21.0 for 21%).
        lloq_wpre_copies_ul (float): Threshold for LLOQ_WPRE_COPIES_UL.
        lloq_rpp30_copies_ul (float): Threshold for LLOQ_RPP30_COPIES_UL.

    Returns:
        pd.DataFrame: The summary_df with an added 'Designation' column and intermediate helper columns.
    """
    # Ensure 'Sample Group' is in original_df if not already present
    # This assumes 'Sample/NTC/Control' is the column for individual sample IDs in original_df
    if 'Sample Group' not in original_df.columns and 'Sample/NTC/Control' in original_df.columns:
        original_df['Sample Group'] = original_df['Sample/NTC/Control'].astype(str).str.split('-').str[0]
    elif 'Sample Group' not in original_df.columns: # Fallback if 'Sample/NTC/Control' is missing
        st.warning("Cannot derive 'Sample Group' in original_df for Designation logic.")
        original_df['Sample Group'] = "Unknown_Group_In_Function" # Placeholder

    # Dynamically detect 'Partitions (valid)' column (case-insensitive)
    possible_valid_part_cols = ["Partitions (valid)", "Partitions (Valid)"]
    valid_part_col = next((col for col in possible_valid_part_cols if col in original_df.columns), None)
    if valid_part_col is not None and 'Sample/NTC/Control' in original_df.columns:
        partitions_data = original_df[["Sample/NTC/Control", valid_part_col]].drop_duplicates(subset=["Sample/NTC/Control"])
        if "Sample ID" in summary_df.columns:
            summary_df = pd.merge(summary_df, partitions_data,
                                  left_on="Sample ID", right_on="Sample/NTC/Control",
                                  how="left")
            # Clean up merged column if necessary
            if "Sample/NTC/Control_y" in summary_df.columns: # Suffix if "Sample/NTC/Control" was already there
                 summary_df.rename(columns={"Sample/NTC/Control_x": "Sample ID Key"}, inplace=True) # Temp rename
                 summary_df.drop(columns=["Sample/NTC/Control_y"], inplace=True, errors='ignore')
            elif "Sample/NTC/Control" in summary_df.columns and "Sample/NTC/Control" != "Sample ID":
                 summary_df.drop(columns=["Sample/NTC/Control"], inplace=True, errors='ignore')
            # Rename to standard name for downstream logic
            summary_df.rename(columns={valid_part_col: "Partitions (valid)"}, inplace=True)
        else:
            summary_df["Partitions (valid)"] = np.nan # Ensure column exists if 'Sample ID' missing in summary_df
    else:
        st.warning("'Partitions (valid)' or 'Partitions (Valid)' or 'Sample/NTC/Control' not found in original CSV for Designation logic.")
        summary_df["Partitions (valid)"] = np.nan # Ensure column exists

    # 2. LLOQ checks are now per-replicate directly on summary_df.
    # The old group-wise WPRE LLOQ check (all_wpre_replicates_meet_lloq) is removed.

    # 3. Ensure necessary columns are numeric in summary_df for checks
    cols_to_numerify_in_summary = ["WPRE Concentration %CV", "RPP30 Concentration %CV",
                                   "RPP30 Concentration", "Partitions (valid)", "WPRE Concentration"]
    for col in cols_to_numerify_in_summary:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        else:
            summary_df[col] = np.nan # Add as NaN if missing to prevent key errors later

    # 4. Boolean helper columns - ensure we handle NaN values properly
    summary_df['is_wpre_cv_ok'] = (summary_df["WPRE Concentration %CV"] <= max_cv_percentage).fillna(False)
    summary_df['is_rpp30_cv_ok'] = (summary_df["RPP30 Concentration %CV"] <= max_cv_percentage).fillna(False)
    summary_df['is_partitions_valid_ok'] = (summary_df["Partitions (valid)"] >= min_valid_partition).fillna(False)
    summary_df['is_wpre_lloq_ok'] = (summary_df["WPRE Concentration"] >= lloq_wpre_copies_ul).fillna(False)
    summary_df['is_rpp30_lloq_ok'] = (summary_df["RPP30 Concentration"] >= lloq_rpp30_copies_ul).fillna(False)

    # 5. Ensure all boolean columns exist (already handled in step 4)
    helper_bool_cols = ['is_wpre_cv_ok', 'is_rpp30_cv_ok', 'is_partitions_valid_ok', 'is_wpre_lloq_ok', 'is_rpp30_lloq_ok']
    for col in helper_bool_cols:
        if col not in summary_df.columns:
            summary_df[col] = False

    # 6. Determine Designation and Designation Summary
    def get_designation_summary(row):
        reasons = []
        if not row['is_partitions_valid_ok']:
            reasons.append(f"Partitions < {min_valid_partition}")
        if not row['is_wpre_cv_ok']:
            reasons.append(f"WPRE %CV > {max_cv_percentage}%")
        if not row['is_rpp30_cv_ok']:
            reasons.append(f"RPP30 %CV > {max_cv_percentage}%")
        if not row['is_wpre_lloq_ok']:
            reasons.append(f"WPRE < {lloq_wpre_copies_ul}")
        if not row['is_rpp30_lloq_ok']:
            reasons.append(f"RPP30 < {lloq_rpp30_copies_ul}")
        
        if not reasons:
            return "All Pass"
        else:
            return "; ".join(reasons)

    summary_df["Designation Summary"] = summary_df.apply(get_designation_summary, axis=1)
    summary_df["Designation"] = np.where(summary_df["Designation Summary"] == "All Pass", "Pass", "Fail")
    
    return summary_df

# Helper function to convert multiple DataFrames to a single Excel bytes object, each DF on a new sheet
def dfs_to_excel_bytes(dfs_map):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs_map.items():
            if df is not None and not df.empty: # Only write if DataFrame exists and is not empty
                df.to_excel(writer, index=False, sheet_name=sheet_name)
    processed_data = output.getvalue()
    return processed_data

# Set the title of the Streamlit app
st.markdown(
    "<h2 style='text-align: left; color: black;'>AZD0120 VCN dPCR Analysis App beta v0.1</h2>",
    unsafe_allow_html=True
)

# Add a file uploader widget for multiple files
uploaded_files = st.file_uploader("Choose CSV files (.csv)", type=['csv'], accept_multiple_files=True, key=f"file_uploader_{st.session_state.get('uploader_key', 0)}")

# User input for CAR_PC constants
st.subheader("CAR PC Range Configuration")
st.write("Please set the CAR PC copy number/cell range for analysis:")

# Initialize session state for CAR_PC values if not exists
if 'car_pc_min' not in st.session_state:
    st.session_state.car_pc_min = DEFAULT_CAR_PC_MIN_COPY_NUMBER_CELL
if 'car_pc_max' not in st.session_state:
    st.session_state.car_pc_max = DEFAULT_CAR_PC_MAX_COPY_NUMBER_CELL

col1, col2 = st.columns(2)
with col1:
    car_pc_min = st.number_input(
        "CAR PC Min Copy Number/Cell:",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.car_pc_min,
        step=0.01,
        format="%.2f",
        key="car_pc_min_input"
    )
    st.session_state.car_pc_min = car_pc_min

with col2:
    car_pc_max = st.number_input(
        "CAR PC Max Copy Number/Cell:",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.car_pc_max,
        step=0.01,
        format="%.2f",
        key="car_pc_max_input"
    )
    st.session_state.car_pc_max = car_pc_max

# Set the user-defined values as the constants for this session
CAR_PC_MIN_COPY_NUMBER_CELL = st.session_state.car_pc_min
CAR_PC_MAX_COPY_NUMBER_CELL = st.session_state.car_pc_max

st.markdown("---")

# Check if files have been uploaded
if uploaded_files is not None and len(uploaded_files) > 0:
    
    # Initialize session state for storing all results
    if 'all_files_results' not in st.session_state:
        st.session_state.all_files_results = {}
    
    # Initialize session state for CD19 user inputs
    if 'user_cd19_inputs' not in st.session_state:
        st.session_state.user_cd19_inputs = {}
    
    # Process each uploaded file
    for file_index, uploaded_file in enumerate(uploaded_files):
        
        st.markdown(f"## Processing File {file_index + 1}: {uploaded_file.name}")
        st.markdown("---")
        
        # Store file identifier to detect new uploads
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"    

        # Robust CSV reading to handle optional 'sep=' line and different encodings
        skiprows_val = 0
        actual_sep = ','  # Default separator
        first_line_str = None
        df = None # Initialize df to None

        encodings_to_try = ['utf-8', 'latin-1', 'utf-16']

        # Try to read and decode the first line to check for 'sep='
        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0)
                first_line_bytes = uploaded_file.readline()
                if not first_line_bytes: # Handle empty file case
                    st.error(f"Uploaded file {uploaded_file.name} is empty.")
                    break # Exit to next file
                first_line_str = first_line_bytes.decode(encoding).strip()
                break  # If decoding worked, break from loop
            except UnicodeDecodeError:
                continue # Try next encoding
            except Exception as e:
                st.error(f"Error reading the first line of file {uploaded_file.name} with encoding {encoding}: {e}")
                first_line_str = None # Ensure it's None if this attempt fails
                # Continue to allow other encodings to be tried for the full read, or fail there.
        
        if first_line_str is None:
            continue # Skip to next file if we couldn't read the first line
            
        uploaded_file.seek(0) # Reset for the main read

        if first_line_str and first_line_str.lower().startswith("sep="):
            skiprows_val = 1
            # Determine the actual separator from the 'sep=' line
            parts = first_line_str.split('=', 1) # Split only on the first '='
            if len(parts) > 1 and parts[1]:
                actual_sep = parts[1]
                # Basic handling for common named separators or if it's just the character
                if actual_sep.lower() == 'comma': 
                    actual_sep = ','
                elif actual_sep.lower() == 'semicolon': 
                    actual_sep = ';'
                elif actual_sep.lower() == 'tab': 
                    actual_sep = '\t'
                # If actual_sep is already a single character like ',' or ';', it's fine.
            else: # Default if 'sep=' is present but no char follows (e.g., "sep=\n")
                actual_sep = ',' # Or consider error/warning
        else: # No 'sep=' line found or error reading first line, assume header is on first line
            skiprows_val = 0
            actual_sep = ',' # Default to comma, or use None for auto-detection by pandas

        # Now, use these dynamic values in pd.read_csv
        read_successful = False
        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0) # Reset for each read attempt
                
                # Read the entire file as bytes and decode to text
                file_content = uploaded_file.read().decode(encoding)
                
                # Split into lines
                all_lines = file_content.split('\n')
                
                # Check if first line contains 'sep=' and skip it if so
                # Remove BOM and other invisible characters before checking
                first_line_clean = all_lines[0].lstrip('\ufeff\ufffe').strip().lower() if all_lines else ''
                if first_line_clean.startswith('sep='):
                    csv_lines = all_lines[1:]  # Skip the sep= line
                else:
                    csv_lines = all_lines  # Use all lines
                
                # Remove empty lines from the end
                while csv_lines and not csv_lines[-1].strip():
                    csv_lines.pop()
                
                # Join lines back together
                cleaned_csv_content = '\n'.join(csv_lines)
                
                # Create StringIO and read with pandas
                csv_stringio = StringIO(cleaned_csv_content)
                df = pd.read_csv(csv_stringio, sep=actual_sep)
                        
                read_successful = True
                break # Exit loop if read is successful
            except UnicodeDecodeError:
                continue # Try next encoding
            except pd.errors.EmptyDataError:
                st.error(f"The file {uploaded_file.name} appears to be empty or became empty after processing with encoding {encoding}.")
                df = pd.DataFrame() # Create an empty DataFrame
                read_successful = True # Treat as "handled"
                break
            except Exception as e:
                # Log error for this encoding, but allow trying others
                st.warning(f"Could not parse CSV {uploaded_file.name} with encoding {encoding} and separator '{actual_sep}': {e}")
                continue
                
        if not read_successful:
            st.error(f"Failed to read or parse the CSV file {uploaded_file.name} with all attempted encodings and separator '{actual_sep}'. Please check the file format.")
            continue # Skip to next file
        
        if df.empty: # Double check if df is empty due to read failure
            st.error(f"Failed to load data into DataFrame for {uploaded_file.name}. The DataFrame is empty.")
            continue # Skip to next file

        # Rename the first column (which is unnamed) to "Well Position"
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "Well Position"}, inplace=True)

        # Strip whitespace from all column names (helps with hidden/extra spaces)
        df.columns = df.columns.str.strip()

        # Dynamically detect concentration column name with robust matching
        possible_conc_columns = ["Conc. [copies/µL]", "Conc. [cp/ÂµL] (dPCR reaction)", "Conc. [cp/µL] (dPCR reaction)"]
        
        # First try exact matching
        conc_col = next((col for col in possible_conc_columns if col in df.columns), None)
        
        # If exact matching fails, try with stripped whitespace
        if conc_col is None:
            df_columns_stripped = {col.strip(): col for col in df.columns}
            possible_conc_columns_stripped = [col.strip() for col in possible_conc_columns]
            conc_col_stripped = next((col for col in possible_conc_columns_stripped if col in df_columns_stripped), None)
            if conc_col_stripped:
                conc_col = df_columns_stripped[conc_col_stripped]
        
        if conc_col is None:
            # Debug: show actual column names found
            st.error(f"No recognized concentration column found in CSV {uploaded_file.name}.")
            st.error(f"Expected one of: {possible_conc_columns}")
            st.error(f"Found columns: {list(df.columns)}")
            continue # Skip to next file

        # Dynamically detect target column name
        possible_target_columns = ["Target", "Target (Name)"]
        target_col = next((col for col in possible_target_columns if col in df.columns), None)
        if target_col is None:
            st.error(f"No recognized target column found in CSV {uploaded_file.name}. Expected one of: 'Target' or 'Target (Name)'.")
            continue # Skip to next file

        # Dynamically detect 'Partitions (Valid)' column
        possible_valid_part_cols = ["Partitions (Valid)", "Partitions (valid)"]
        valid_part_col = next((col for col in possible_valid_part_cols if col in df.columns), None)
        if valid_part_col is None:
            st.error(f"No recognized 'Partitions (Valid)' column found in CSV {uploaded_file.name}. Expected one of: 'Partitions (Valid)' or 'Partitions (valid)'.")
            continue # Skip to next file

        # Dynamically detect 'Partitions (Positive)' column
        possible_positive_part_cols = ["Partitions (Positive)", "Partitions (positive)"]
        positive_part_col = next((col for col in possible_positive_part_cols if col in df.columns), None)
        if positive_part_col is None:
            st.error(f"No recognized 'Partitions (Positive)' column found in CSV {uploaded_file.name}. Expected one of: 'Partitions (Positive)' or 'Partitions (positive)'.")
            continue # Skip to next file
        
        # Assay Status Check based on NTC performance
        assay_status_message = "Assay Status: Pending"  # Start with pending until all checks complete
        status_color = "orange"
        assay_failure_reasons = []
        assay_pending_reasons = []

        # --- Comprehensive Assay Status Checks ---

        # 1. NTC Positive Partition Check
        ntc_df = df[df["Sample/NTC/Control"].astype(str).str.startswith("NTC-")]
        if ntc_df.empty:
            assay_failure_reasons.append("NTC Check: No NTC samples found.")
        else:
            ntc_positive_parts = pd.to_numeric(ntc_df[positive_part_col], errors='coerce')
            failing_ntcs = ntc_df[ntc_positive_parts > NTC_POS_PART_CUTOFF]
            if not failing_ntcs.empty:
                for _, row in failing_ntcs.iterrows():
                    reason = f"NTC Check: Sample '{row['Sample/NTC/Control']}' failed with {row[positive_part_col]} positive partitions (threshold: <= {NTC_POS_PART_CUTOFF})."
                    assay_failure_reasons.append(reason)

        # 2. Valid Partitions Check for NTC & PC
        pc_df = df[df["Sample/NTC/Control"].astype(str).str.startswith("PC-")]
        control_samples_df = pd.concat([ntc_df, pc_df])
        if control_samples_df.empty:
            assay_failure_reasons.append("Partition Check: No NTC or PC samples found to check.")
        else:
            valid_partitions = pd.to_numeric(control_samples_df[valid_part_col], errors='coerce')
            failing_partitions = control_samples_df[valid_partitions < MIN_VALID_PARTITION]
            if not failing_partitions.empty:
                for _, row in failing_partitions.iterrows():
                    reason = f"Partition Check: Sample '{row['Sample/NTC/Control']}' failed with {row[valid_part_col]} valid partitions (threshold: >= {MIN_VALID_PARTITION})."
                    assay_failure_reasons.append(reason)

        # 3. PC %CV Checks
        if pc_df.empty:
            assay_failure_reasons.append("PC CV Check: No PC samples found.")
        else:
            # Create a pivot table for PC samples to calculate CV
            pc_pivot = pc_df.pivot_table(index="Sample/NTC/Control", columns=target_col, values=conc_col).reset_index()
            pc_pivot.rename(columns={"WPRE": "WPRE Concentration", "RPP30": "RPP30 Concentration"}, inplace=True)
            
            # Add a single 'Sample Group' for CV calculation across all PC replicates
            pc_pivot['Sample Group'] = 'PC'

            # WPRE %CV for PC
            if "WPRE Concentration" in pc_pivot.columns:
                wpre_conc_pc = pd.to_numeric(pc_pivot["WPRE Concentration"], errors='coerce')
                if len(wpre_conc_pc.dropna()) > 1: # Need at least 2 values for std dev
                    wpre_mean_pc = wpre_conc_pc.mean()
                    wpre_std_pc = wpre_conc_pc.std()
                    wpre_cv_pc = (wpre_std_pc / wpre_mean_pc) * 100 if wpre_mean_pc != 0 else 0
                    if wpre_cv_pc > MAX_CV_PERCENTAGE:
                        assay_failure_reasons.append(f"PC CV Check: WPRE Concentration %CV is {wpre_cv_pc:.2f}% (threshold: <= {MAX_CV_PERCENTAGE}%).")
                else:
                     assay_failure_reasons.append("PC CV Check: Not enough PC WPRE data points to calculate %CV.")
            
            # RPP30 %CV for PC
            if "RPP30 Concentration" in pc_pivot.columns:
                rpp30_conc_pc = pd.to_numeric(pc_pivot["RPP30 Concentration"], errors='coerce')
                if len(rpp30_conc_pc.dropna()) > 1:
                    rpp30_mean_pc = rpp30_conc_pc.mean()
                    rpp30_std_pc = rpp30_conc_pc.std()
                    rpp30_cv_pc = (rpp30_std_pc / rpp30_mean_pc) * 100 if rpp30_mean_pc != 0 else 0
                    if rpp30_cv_pc > MAX_CV_PERCENTAGE:
                        assay_failure_reasons.append(f"PC CV Check: RPP30 Concentration %CV is {rpp30_cv_pc:.2f}% (threshold: <= {MAX_CV_PERCENTAGE}%).")
                else:
                    assay_failure_reasons.append("PC CV Check: Not enough PC RPP30 data points to calculate %CV.")

        # Finalize Status - will be updated after PC range check
        if assay_failure_reasons:
            assay_status_message = "Assay Status: Fail"
            status_color = "red"

        # Display will be updated after PC range check
        assay_status_placeholder = st.empty()
        assay_criteria_placeholder = st.empty()
                    
        # Display threshold values used in analysis
        st.write('**Thresholds Used in Analysis:**')
        st.write(f"MIN_VALID_PARTITION: {MIN_VALID_PARTITION}")
        st.write(f"NTC_POS_PART_CUTOFF: {NTC_POS_PART_CUTOFF}")
        st.write(f"MAX_CV_PERCENTAGE: {MAX_CV_PERCENTAGE}%")
        st.write(f"LLOQ_WPRE_COPIES_UL: {LLOQ_WPRE_COPIES_UL}")
        st.write(f"LLOQ_RPP30_COPIES_UL: {LLOQ_RPP30_COPIES_UL}")
        st.write(f"CAR_PC_MIN_COPY_NUMBER_CELL: {CAR_PC_MIN_COPY_NUMBER_CELL}")
        st.write(f"CAR_PC_MAX_COPY_NUMBER_CELL: {CAR_PC_MAX_COPY_NUMBER_CELL}")
        st.markdown("---") # Add a horizontal line for separation
        with st.expander("CSV Uploaded Table", expanded=False):
            st.dataframe(df)

        st.subheader("Summary Table")
        # Create the summary DataFrame using pivot_table
        # Ensure the column names match exactly what's in your df after previous processing
        # Based on your CSV output: "Sample/NTC/Control", "Target", "Conc. [copies/µL]"
        
        # Check if necessary columns exist before pivoting
        required_cols_for_pivot = ["Sample/NTC/Control", target_col, conc_col]
        if all(col in df.columns for col in required_cols_for_pivot):
            try:
                summary_df = df.pivot_table(
                    index="Sample/NTC/Control",
                    columns=target_col,
                    values=conc_col
                ).reset_index() # Convert index "Sample/NTC/Control" back to a column

                # Rename columns for the final summary table
                summary_df.rename(columns={
                    "Sample/NTC/Control": "Sample ID",
                    "WPRE": "WPRE Concentration",  # Assuming "WPRE" is a value in "Target" column
                    "RPP30": "RPP30 Concentration" # Assuming "RPP30" is a value in "Target" column
                }, inplace=True)
                
                # Remove only NTC samples from the summary table, keep PC and all other samples
                summary_df = summary_df[~summary_df["Sample ID"].astype(str).str.startswith("NTC")]

                # Calculate "Copy number/cell"
                # Ensure both concentration columns exist before attempting calculation
                if "WPRE Concentration" in summary_df.columns and "RPP30 Concentration" in summary_df.columns:
                    # Convert columns to numeric, coercing errors to NaN. This helps if they are not already numeric.
                    wpre_conc = pd.to_numeric(summary_df["WPRE Concentration"], errors='coerce')
                    rpp30_conc = pd.to_numeric(summary_df["RPP30 Concentration"], errors='coerce')
                    
                    # Perform calculation, using np.isclose for floating-point comparison
                    summary_df["Copy number/cell"] = np.where(
                        (np.isclose(rpp30_conc, 0, atol=1e-10)) | (rpp30_conc.isna()),
                        np.nan,  # Result is NaN if RPP30 is 0 or NaN
                        (wpre_conc / rpp30_conc) * 2
                    )
                else:
                    # If one of the concentration columns is missing, create the "Copy number/cell" column with NaNs
                    summary_df["Copy number/cell"] = np.nan

                # Calculate %CV for WPRE and RPP30 Concentrations
                if "Sample ID" in summary_df.columns:
                    # Extract Sample Group (part before '-')
                    summary_df['Sample Group'] = summary_df['Sample ID'].astype(str).str.split('-').str[0]
                    
                    # WPRE Concentration %CV
                    if "WPRE Concentration" in summary_df.columns:
                        wpre_conc_numeric = pd.to_numeric(summary_df["WPRE Concentration"], errors='coerce')
                        grouped_wpre_mean = wpre_conc_numeric.groupby(summary_df['Sample Group']).transform('mean')
                        grouped_wpre_std = wpre_conc_numeric.groupby(summary_df['Sample Group']).transform('std')
                        
                        summary_df["WPRE Concentration %CV"] = np.where(
                            (np.isclose(grouped_wpre_mean, 0, atol=1e-10)) | grouped_wpre_mean.isna() | grouped_wpre_std.isna(),
                            np.nan, # %CV is NaN if mean is 0, or std is NaN (e.g. single sample group)
                            (grouped_wpre_std / grouped_wpre_mean) * 100
                        ).round(2)
                    else:
                        summary_df["WPRE Concentration %CV"] = np.nan

                    # RPP30 Concentration %CV
                    if "RPP30 Concentration" in summary_df.columns:
                        rpp30_conc_numeric = pd.to_numeric(summary_df["RPP30 Concentration"], errors='coerce')
                        grouped_rpp30_mean = rpp30_conc_numeric.groupby(summary_df['Sample Group']).transform('mean')
                        grouped_rpp30_std = rpp30_conc_numeric.groupby(summary_df['Sample Group']).transform('std')

                        summary_df["RPP30 Concentration %CV"] = np.where(
                            (np.isclose(grouped_rpp30_mean, 0, atol=1e-10)) | grouped_rpp30_mean.isna() | grouped_rpp30_std.isna(),
                            np.nan, # %CV is NaN if mean is 0, or std is NaN
                            (grouped_rpp30_std / grouped_rpp30_mean) * 100
                        ).round(2)
                    else:
                        summary_df["RPP30 Concentration %CV"] = np.nan
                else:
                    summary_df["WPRE Concentration %CV"] = np.nan
                    summary_df["RPP30 Concentration %CV"] = np.nan

                # User input for %CD19 values
                st.subheader("User Input for %CD19 Values")
                st.write("Please enter the %CD19 values for each sample group:")
                
                # Get unique sample groups (excluding only NTC, but including PC)
                sample_groups = summary_df['Sample Group'].unique()
                control_groups = ['NTC']  # Only exclude NTC, include PC for CD19 input
                sample_groups_filtered = [group for group in sample_groups if group not in control_groups]
                
                # Initialize user_cd19_inputs for this file if not exists
                file_key = f"{uploaded_file.name}_cd19"
                if file_key not in st.session_state.user_cd19_inputs:
                    st.session_state.user_cd19_inputs[file_key] = {}
                
                # Create input fields for each sample group
                for group in sample_groups_filtered:
                    default_value = st.session_state.user_cd19_inputs[file_key].get(group, 0.0)
                    user_input = st.number_input(
                        f"%CD19 for {group}:",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_value,
                        step=0.1,
                        format="%.1f",
                        key=f"{file_key}_{group}_cd19"
                    )
                    st.session_state.user_cd19_inputs[file_key][group] = user_input
                
                # Apply user inputs to summary_df
                summary_df["User input %CD19"] = summary_df['Sample Group'].map(
                    st.session_state.user_cd19_inputs[file_key]
                ).fillna(0.0)
                
                # Calculate "Average copy number/transduced cell"
                # First calculate the average Copy number/cell for each sample group, then divide by CD19
                if "Copy number/cell" in summary_df.columns and "Sample Group" in summary_df.columns:
                    # Calculate average Copy number/cell per sample group
                    copy_number_numeric = pd.to_numeric(summary_df["Copy number/cell"], errors='coerce')
                    avg_copy_per_group = copy_number_numeric.groupby(summary_df['Sample Group']).transform('mean')
                    
                    # Calculate Average copy number/transduced cell using group average
                    def calculate_avg_copy_per_transduced_cell(avg_copy_value, cd19_value):
                        if pd.isna(avg_copy_value) or cd19_value == 0:
                            return "N/A"
                        else:
                            return (avg_copy_value / cd19_value) * 100
                    
                    # Apply calculation using vectorized operations
                    summary_df["Average copy number/transduced cell"] = np.where(
                        (pd.isna(avg_copy_per_group)) | (summary_df["User input %CD19"] == 0),
                        "N/A",
                        (avg_copy_per_group / summary_df["User input %CD19"]) * 100
                    )
                else:
                    summary_df["Average copy number/transduced cell"] = "N/A"
                
                # Calculate Designation
                if not summary_df.empty and 'Sample Group' in summary_df.columns and df is not None and not df.empty:
                    # Ensure original_df (df) has 'Sample Group' if it's derived from 'Sample/NTC/Control'
                    if 'Sample Group' not in df.columns and 'Sample/NTC/Control' in df.columns:
                        df_copy = df.copy() # Work on a copy to avoid modifying original df in this scope
                        df_copy['Sample Group'] = df_copy['Sample/NTC/Control'].astype(str).str.split('-').str[0]
                    elif 'Sample Group' in df.columns:
                        df_copy = df.copy()
                    else: # If 'Sample Group' cannot be derived or found in df
                        st.warning("Could not ensure 'Sample Group' in original data for Designation. Designation might be inaccurate.")
                        df_copy = df.copy() # Proceed with a copy, function will handle missing 'Sample Group'
                    
                    summary_df = add_designation_column(
                        summary_df, df_copy,
                        MIN_VALID_PARTITION, NTC_POS_PART_CUTOFF, MAX_CV_PERCENTAGE,
                        LLOQ_WPRE_COPIES_UL, LLOQ_RPP30_COPIES_UL
                    )
                else:
                    # If summary_df or original df is empty, or other issues, set defaults
                    summary_df["Designation"] = "Fail" # Default to Fail
                    summary_df["Designation Summary"] = "Default Fail"
                
                # Define helper columns to drop after calculations are done
                helper_columns_to_drop = [
                    'is_wpre_cv_ok', 'is_rpp30_cv_ok', 'is_partitions_valid_ok',
                    'is_wpre_lloq_ok', 'is_rpp30_lloq_ok',
                ]
                # Drop helper columns if they exist, ignore errors if they don't
                summary_df.drop(columns=[col for col in helper_columns_to_drop if col in summary_df.columns], inplace=True, errors='ignore')

                # Select and reorder columns for the final display
                final_columns = ["Sample ID"]
                required_columns = {
                    "WPRE Concentration": "WPRE Concentration",
                    "RPP30 Concentration": "RPP30 Concentration", 
                    "Copy number/cell": "Copy number/cell",
                    "User input %CD19": "User input %CD19",
                    "Average copy number/transduced cell": "Average copy number/transduced cell",
                    "WPRE Concentration %CV": "WPRE Concentration %CV",
                    "RPP30 Concentration %CV": "RPP30 Concentration %CV",
                    "Designation": "Designation",
                    "Designation Summary": "Designation Summary"
                }
                
                # Add columns that exist in the dataframe
                for col_key, col_name in required_columns.items():
                    if col_key in summary_df.columns:
                        final_columns.append(col_name)
                
                # Ensure only existing columns are selected and dataframe is not empty
                existing_columns = [col for col in final_columns if col in summary_df.columns]
                if existing_columns:
                    summary_df = summary_df[existing_columns]

                # Convert 'Average copy number/transduced cell' to string to handle "N/A" for display
                if "Average copy number/transduced cell" in summary_df.columns:
                    summary_df["Average copy number/transduced cell"] = summary_df["Average copy number/transduced cell"].astype(str)
                st.dataframe(summary_df)

                # --- PC Range Check for Assay Status ---
                # Check if PC samples have CD19 input and are within range
                pc_samples = summary_df[summary_df["Sample ID"].astype(str).str.startswith("PC-")]
                
                if not pc_samples.empty:
                    # Check if CD19 values are provided for PC samples
                    pc_cd19_missing = pc_samples[pc_samples["User input %CD19"] == 0.0]
                    
                    if not pc_cd19_missing.empty:
                        # CD19 values missing for PC samples
                        assay_pending_reasons.append("PC Range Check: CD19 values not provided for PC samples. Please enter CD19 values to complete assay status evaluation.")
                    else:
                        # CD19 values provided, check PC range
                        pc_avg_copy_issues = []
                        for _, row in pc_samples.iterrows():
                            avg_copy_str = row.get("Average copy number/transduced cell", "N/A")
                            if avg_copy_str != "N/A" and avg_copy_str != "" and pd.notna(avg_copy_str):
                                try:
                                    avg_copy_value = float(avg_copy_str)
                                    if avg_copy_value < CAR_PC_MIN_COPY_NUMBER_CELL or avg_copy_value > CAR_PC_MAX_COPY_NUMBER_CELL:
                                        pc_avg_copy_issues.append(f"PC Range Check: Sample '{row['Sample ID']}' average copy number/transduced cell ({avg_copy_value:.2f}) is outside acceptable range ({CAR_PC_MIN_COPY_NUMBER_CELL:.2f} - {CAR_PC_MAX_COPY_NUMBER_CELL:.2f}).")
                                except (ValueError, TypeError):
                                    pc_avg_copy_issues.append(f"PC Range Check: Sample '{row['Sample ID']}' has invalid average copy number/transduced cell value.")
                        
                        if pc_avg_copy_issues:
                            assay_failure_reasons.extend(pc_avg_copy_issues)

                # Final Status Determination
                if assay_pending_reasons and not assay_failure_reasons:
                    # Still pending - waiting for CD19 input
                    final_assay_status_message = "Assay Status: Pending"
                    final_status_color = "orange"
                elif assay_failure_reasons:
                    # Has failures
                    final_assay_status_message = "Assay Status: Fail"
                    final_status_color = "red"
                else:
                    # All checks passed
                    final_assay_status_message = "Assay Status: Pass"
                    final_status_color = "green"

                # Update the placeholder with final status
                with assay_status_placeholder:
                    st.markdown(f"<h4 style='color: {final_status_color};'>{final_assay_status_message}</h4>", unsafe_allow_html=True)

                # Update the criteria display
                with assay_criteria_placeholder:
                    with st.expander("View Assay Status Criteria"):
                        if final_assay_status_message == "Assay Status: Pass":
                            st.success("All assay status checks passed (NTC Partitions, Control Valid Partitions, PC %CV, PC Range).")
                        else:
                            if assay_failure_reasons:
                                for reason in assay_failure_reasons:
                                    st.error(reason)
                            if assay_pending_reasons:
                                for reason in assay_pending_reasons:
                                    st.warning(reason)

                # Update variables for storage
                assay_status_message = final_assay_status_message
                status_color = final_status_color

                # Store results for this file
                st.session_state.all_files_results[uploaded_file.name] = {
                    'summary_df': summary_df.copy(),
                    'original_df': df.copy(),
                    'assay_status_message': assay_status_message,
                    'assay_failure_reasons': assay_failure_reasons.copy(),
                    'assay_pending_reasons': assay_pending_reasons.copy()
                }
            except Exception as e:
                st.error(f"Error creating summary table for {uploaded_file.name}: {e}")
                st.write("Please ensure the CSV contains 'Sample/NTC/Control', a target column ('Target' or 'Target (Name)'), and a concentration column ('Conc. [copies/µL]' or 'Conc. [cp/ÂµL] (dPCR reaction)') and that the target column contains 'WPRE' and 'RPP30' values.")
        else:
            st.warning(f"Could not create summary table for {uploaded_file.name}. Required columns ('Sample/NTC/Control', a target column ['Target' or 'Target (Name)'], and a concentration column ['Conc. [copies/µL]' or 'Conc. [cp/ÂµL] (dPCR reaction)']) not found in the uploaded CSV.")

    # --- Combined Export Results Section for All Files ---
    if st.session_state.get('all_files_results', {}):
        st.markdown("---")
        st.header("📊 Combined Results from All Files")
        
        # Display summary of all processed files
        st.subheader("Files Processed:")
        for filename, results in st.session_state.all_files_results.items():
            if "Pass" in results['assay_status_message']:
                status_color = "green"
            elif "Pending" in results['assay_status_message']:
                status_color = "orange"
            else:
                status_color = "red"
            st.markdown(f"- **{filename}**: <span style='color: {status_color};'>{results['assay_status_message']}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("Export Combined Results")

        # Prepare combined data for multi-sheet Excel
        combined_data_to_export = {}
        
        # Combine Summary Results from all files
        all_summary_data = []
        all_raw_data = []
        all_assay_status_data = []
        
        for filename, results in st.session_state.all_files_results.items():
            # Add filename column to distinguish data from different files
            summary_with_file = results['summary_df'].copy()
            summary_with_file['Source_File'] = filename
            all_summary_data.append(summary_with_file)
            
            raw_with_file = results['original_df'].copy()
            raw_with_file['Source_File'] = filename
            all_raw_data.append(raw_with_file)
            
            # Create assay status data for this file
            assay_status_row = {
                'Source_File': filename,
                'Assay_Status': results['assay_status_message'].replace("Assay Status: ", ""),
                'Failure_Reasons': "; ".join(results['assay_failure_reasons']) if results['assay_failure_reasons'] else "None",
                'Pending_Reasons': "; ".join(results.get('assay_pending_reasons', [])) if results.get('assay_pending_reasons') else "None",
                'Analysis_Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'MIN_VALID_PARTITION': MIN_VALID_PARTITION,
                'NTC_POS_PART_CUTOFF': NTC_POS_PART_CUTOFF,
                'MAX_CV_PERCENTAGE': MAX_CV_PERCENTAGE,
                'LLOQ_WPRE_COPIES_UL': LLOQ_WPRE_COPIES_UL,
                'LLOQ_RPP30_COPIES_UL': LLOQ_RPP30_COPIES_UL,
                'CAR_PC_MIN_COPY_NUMBER_CELL': CAR_PC_MIN_COPY_NUMBER_CELL,
                'CAR_PC_MAX_COPY_NUMBER_CELL': CAR_PC_MAX_COPY_NUMBER_CELL
            }
            all_assay_status_data.append(assay_status_row)
        
        # Create combined DataFrames
        if all_summary_data:
            combined_summary_df = pd.concat(all_summary_data, ignore_index=True)
            combined_data_to_export['Combined_Summary'] = combined_summary_df
        
        if all_raw_data:
            combined_raw_df = pd.concat(all_raw_data, ignore_index=True)
            combined_data_to_export['Combined_Raw_Data'] = combined_raw_df
            
        if all_assay_status_data:
            combined_assay_status_df = pd.DataFrame(all_assay_status_data)
            combined_data_to_export['Assay_Status'] = combined_assay_status_df

        if combined_data_to_export:
            # Export combined Excel file
            excel_bytes = dfs_to_excel_bytes(combined_data_to_export)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_filename = f"Combined_AZD0120_Analysis_{timestamp}.xlsx"
            
            st.download_button(
                label="📥 Download Combined Excel Report",
                data=excel_bytes,
                file_name=combined_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No combined data available for export.")
    # --- End of Combined Export Results Section ---
else:
    st.info("Please upload one or more CSV files to begin analysis.")
    # Clear results when no files are uploaded
    if 'all_files_results' in st.session_state:
        del st.session_state.all_files_results
