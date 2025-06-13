# Import the necessary libraries
import streamlit as st
import pandas as pd
import io # Use io to handle the uploaded file bytes
import numpy as np
import datetime # For timestamping the export file
import re # For parsing Input ID

# Define global constants for analysis thresholds and parameters
MIN_VALID_PARTITION = 7000
NTC_POS_PART_CUTOFF = 5
MAX_CV_PERCENTAGE = 21.0  # Representing 0.21 as 21% for direct comparison if CVs are in %
LLOQ_WPRE_COPIES_UL = 37
LLOQ_RPP30_COPIES_UL = 146
CAR_PC_MIN_COPY_NUMBER_CELL = "TBD" # Assuming this relates to copy number/cell
CAR_PC_MAX_COPY_NUMBER_CELL = "TBD" # Assuming this relates to copy number/cell

# Note: MAX_CV_PERCENTAGE is defined as 21.0. If your CV calculations result in values like 0.21,
# you'll need to either multiply them by 100 before comparing with MAX_CV_PERCENTAGE,
# or define MAX_CV as 0.21 and compare directly.
# The user specified max_cv = 0.21, so let's stick to that for direct use.
MAX_CV = 0.21
def add_designation_column(summary_df, original_df, min_valid_partition, ntc_pos_part_cutoff, max_cv_percentage, lloq_wpre_copies_ul, lloq_rpp30_copies_ul):
    """
    Adds a 'Designation' column to the summary_df based on several criteria.

    Args:
        summary_df (pd.DataFrame): The summary table.
        original_df (pd.DataFrame): The original uploaded data.
        min_valid_partition (int): Threshold for MIN_VALID_PARTITION.
        ntc_pos_part_cutoff (int): Threshold for NTC_POS_PART_CUTOFF.
        max_cv_percentage (float): Threshold for MAX_CV_PERCENTAGE.
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

    # 4. Boolean helper columns
    summary_df['is_wpre_cv_ok'] = summary_df["WPRE Concentration %CV"] <= max_cv_percentage
    summary_df['is_rpp30_cv_ok'] = summary_df["RPP30 Concentration %CV"] <= max_cv_percentage
    summary_df['is_partitions_valid_ok'] = summary_df["Partitions (valid)"] >= min_valid_partition
    summary_df['is_wpre_lloq_ok'] = summary_df["WPRE Concentration"] >= lloq_wpre_copies_ul # New per-replicate WPRE LLOQ
    summary_df['is_rpp30_lloq_ok'] = summary_df["RPP30 Concentration"] >= lloq_rpp30_copies_ul # This was already per-replicate

    # 5. Fill NaNs in boolean checks with False (conservative: if info missing, condition fails)
    helper_bool_cols = ['is_wpre_cv_ok', 'is_rpp30_cv_ok', 'is_partitions_valid_ok', 'is_wpre_lloq_ok', 'is_rpp30_lloq_ok']
    for col in helper_bool_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna(False)
        else: # If helper column itself couldn't be created (e.g. precursor missing)
            summary_df[col] = False # Create it as all False for safety in the 'all' condition

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

# Set the title of the Streamlit app
st.markdown(
    "<h2 style='text-align: left; color: black;'>AZD0120 VCN dPCR Analysis App beta v0.1</h2>",
    unsafe_allow_html=True
)

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an csv file (.csv)", type=['csv'])

# If a file is uploaded
if uploaded_file is not None:
    

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
                st.error("Uploaded file is empty.")
                st.stop() # Stop execution if file is empty
            first_line_str = first_line_bytes.decode(encoding).strip()
            break  # If decoding worked, break from loop
        except UnicodeDecodeError:
            continue # Try next encoding
        except Exception as e:
            st.error(f"Error reading the first line of the file with encoding {encoding}: {e}")
            first_line_str = None # Ensure it's None if this attempt fails
            # Continue to allow other encodings to be tried for the full read, or fail there.
    
    uploaded_file.seek(0) # Reset for the main read

    if first_line_str and first_line_str.lower().startswith("sep="):
        skiprows_val = 1
        # Determine the actual separator from the 'sep=' line
        parts = first_line_str.split('=', 1) # Split only on the first '='
        if len(parts) > 1 and parts[1]:
            actual_sep = parts[1]
            # Basic handling for common named separators or if it's just the character
            if actual_sep.lower() == 'comma': actual_sep = ','
            elif actual_sep.lower() == 'semicolon': actual_sep = ';'
            elif actual_sep.lower() == 'tab': actual_sep = '\t'
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
            df = pd.read_csv(uploaded_file, sep=actual_sep, header=0, skiprows=skiprows_val, encoding=encoding)
            read_successful = True
            break # Exit loop if read is successful
        except UnicodeDecodeError:
            continue # Try next encoding
        except pd.errors.EmptyDataError:
            st.error(f"The file appears to be empty or became empty after skipping rows with encoding {encoding}.")
            df = pd.DataFrame() # Create an empty DataFrame
            read_successful = True # Treat as "handled"
            break
        except Exception as e:
            # Log error for this encoding, but allow trying others
            st.warning(f"Could not parse CSV with encoding {encoding} and separator '{actual_sep}': {e}")
            continue
            
    if not read_successful:
        st.error(f"Failed to read or parse the CSV file with all attempted encodings and separator '{actual_sep}'. Please check the file format.")
        st.stop() # Stop execution if file cannot be read
    
    if df.empty and not read_successful: # Double check if df is empty due to read failure
        st.error("Failed to load data into DataFrame. The DataFrame is empty.")
        st.stop()

    # Rename the first column (which is unnamed) to "Well Position"
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "Well Position"}, inplace=True)

    # Strip whitespace from all column names (helps with hidden/extra spaces)
    df.columns = df.columns.str.strip()

    # Dynamically detect concentration column name
    possible_conc_columns = ["Conc. [copies/µL]", "Conc. [cp/ÂµL] (dPCR reaction)"]
    conc_col = next((col for col in possible_conc_columns if col in df.columns), None)
    if conc_col is None:
        st.error("No recognized concentration column found in CSV. Expected one of: 'Conc. [copies/µL]' or 'Conc. [cp/ÂµL] (dPCR reaction)'.")
        st.stop()

    # Dynamically detect target column name
    possible_target_columns = ["Target", "Target (Name)"]
    target_col = next((col for col in possible_target_columns if col in df.columns), None)
    if target_col is None:
        st.error("No recognized target column found in CSV. Expected one of: 'Target' or 'Target (Name)'.")
        st.stop()

    # Dynamically detect 'Partitions (Valid)' column
    possible_valid_part_cols = ["Partitions (Valid)", "Partitions (valid)"]
    valid_part_col = next((col for col in possible_valid_part_cols if col in df.columns), None)
    if valid_part_col is None:
        st.error("No recognized 'Partitions (Valid)' column found in CSV. Expected one of: 'Partitions (Valid)' or 'Partitions (valid)'.")
        st.stop()

    # Dynamically detect 'Partitions (Positive)' column
    possible_positive_part_cols = ["Partitions (Positive)", "Partitions (positive)"]
    positive_part_col = next((col for col in possible_positive_part_cols if col in df.columns), None)
    if positive_part_col is None:
        st.error("No recognized 'Partitions (Positive)' column found in CSV. Expected one of: 'Partitions (Positive)' or 'Partitions (positive)'.")
        st.stop()

    # Display the DataFrame
# Assay Status Check based on NTC performance
    assay_status_message = "Assay Status: Pass"
    status_color = "green"
    assay_failure_reasons = []

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

    # Finalize Status
    if assay_failure_reasons:
        assay_status_message = "Assay Status: Fail"
        status_color = "red"

    st.markdown(f"<h4 style='color: {status_color};'>{assay_status_message}</h4>", unsafe_allow_html=True)

    # Always display the criteria in an expander
    with st.expander("View Assay Status Criteria"):
        if not assay_failure_reasons:
            st.success("All assay status checks passed (NTC Partitions, Control Valid Partitions, PC %CV).")
        else:
            for reason in assay_failure_reasons:
                st.error(reason)
    # Display threshold values used in analysis
    st.write('**Thresholds Used in Analysis:**')
    st.write(f"MIN_VALID_PARTITION: {MIN_VALID_PARTITION}")
    st.write(f"NTC_POS_PART_CUTOFF: {NTC_POS_PART_CUTOFF}")
    st.write(f"MAX_CV_PERCENTAGE: {MAX_CV_PERCENTAGE}")
    st.write(f"LLOQ_WPRE_COPIES_UL: {LLOQ_WPRE_COPIES_UL}")
    st.write(f"LLOQ_RPP30_COPIES_UL: {LLOQ_RPP30_COPIES_UL}")
    st.write(f"CAR_PC_MIN_COPY_NUMBER_CELL: {CAR_PC_MIN_COPY_NUMBER_CELL}")
    st.write(f"CAR_PC_MAX_COPY_NUMBER_CELL: {CAR_PC_MAX_COPY_NUMBER_CELL}")
    st.write(f"MAX_CV: {MAX_CV}")
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
# Remove NTC samples from the summary table
            summary_df = summary_df[~summary_df["Sample ID"].astype(str).str.startswith("NTC")]

            # Calculate "Copy number/cell"
            # Ensure both concentration columns exist before attempting calculation
            if "WPRE Concentration" in summary_df.columns and "RPP30 Concentration" in summary_df.columns:
                # Convert columns to numeric, coercing errors to NaN. This helps if they are not already numeric.
                wpre_conc = pd.to_numeric(summary_df["WPRE Concentration"], errors='coerce')
                rpp30_conc = pd.to_numeric(summary_df["RPP30 Concentration"], errors='coerce')
                
                # Perform calculation, np.where handles division by zero or NaN in RPP30
                summary_df["Copy number/cell"] = np.where(
                    (rpp30_conc == 0) | (rpp30_conc.isna()),
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
# Calculate Average "Copy number/cell" per Sample Group
                if "Copy number/cell" in summary_df.columns:
                    # Ensure "Copy number/cell" is numeric for the mean calculation
                    copy_number_cell_numeric = pd.to_numeric(summary_df["Copy number/cell"], errors='coerce')
                    summary_df["Avg CnC per Group"] = copy_number_cell_numeric.groupby(summary_df['Sample Group']).transform('mean')
                else:
                    # If "Copy number/cell" column doesn't exist, fill "Avg CnC per Group" with NaN
                    summary_df["Avg CnC per Group"] = np.nan
                # User input for %CD19
                st.subheader("Enter %CD19 Values")
                # Get unique sample groups from the current summary_df and sort them for consistent display order
                unique_sample_groups = sorted(list(summary_df['Sample Group'].unique()))

                # Initialize session state for CD19 inputs if it doesn't exist
                if 'user_cd19_inputs' not in st.session_state:
                    st.session_state.user_cd19_inputs = {}

                # Synchronize session state: ensure it only contains current groups,
                # preserving existing values or defaulting new ones to 0.
                # This handles cases like file re-uploads with different sample groups.
                current_valid_inputs = {}
                for group in unique_sample_groups:
                    current_valid_inputs[group] = st.session_state.user_cd19_inputs.get(group, 0.0) # Default to 0.0
                st.session_state.user_cd19_inputs = current_valid_inputs
                
                # Create input fields for each unique sample group
                for group in unique_sample_groups:
                    # Each input widget needs a unique and stable key
                    input_key = f"cd19_input_for_group_{group}"
                    
                    user_input = st.number_input(
                        label=f"%CD19 for Sample Group '{group}'",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.user_cd19_inputs.get(group, 0.0), # Use value from session state or default
                        step=0.1,
                        key=input_key,
                        help="Enter the percentage of CD19 positive cells (0.0-100.0)."
                    )
                    # Update session state immediately as user types
                    st.session_state.user_cd19_inputs[group] = user_input

                # Map the collected user inputs to the summary_df
                # .get(group, np.nan) could be used if we wanted a non-entered field to be NaN by default
                # but since we default to 0 in session_state, this will map those 0s.
                # .fillna(np.nan) handles any sample groups in df not covered by inputs (should not happen here).
                summary_df["User input %CD19"] = summary_df['Sample Group'].map(st.session_state.user_cd19_inputs).fillna(np.nan)

                # WPRE Concentration %CV
                if "WPRE Concentration" in summary_df.columns:
                    wpre_conc_numeric = pd.to_numeric(summary_df["WPRE Concentration"], errors='coerce')
                    grouped_wpre_mean = wpre_conc_numeric.groupby(summary_df['Sample Group']).transform('mean')
                    grouped_wpre_std = wpre_conc_numeric.groupby(summary_df['Sample Group']).transform('std')
                    
                    summary_df["WPRE Concentration %CV"] = np.where(
                        (grouped_wpre_mean == 0) | grouped_wpre_mean.isna() | grouped_wpre_std.isna(),
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
                        (grouped_rpp30_mean == 0) | grouped_rpp30_mean.isna() | grouped_rpp30_std.isna(),
                        np.nan, # %CV is NaN if mean is 0, or std is NaN
                        (grouped_rpp30_std / grouped_rpp30_mean) * 100
                    ).round(2)
                else:
                    summary_df["RPP30 Concentration %CV"] = np.nan
                
                # Drop the temporary Sample Group column if no longer needed for display
                # summary_df.drop(columns=['Sample Group'], inplace=True) # Or keep it if useful
            else:
                summary_df["WPRE Concentration %CV"] = np.nan
                summary_df["RPP30 Concentration %CV"] = np.nan

# Calculate "Average copy number/transduced cell"
            if "Avg CnC per Group" in summary_df.columns and "User input %CD19" in summary_df.columns:
                # Ensure inputs are numeric for calculation
                avg_cnc_group = pd.to_numeric(summary_df["Avg CnC per Group"], errors='coerce')
                user_input_cd19 = pd.to_numeric(summary_df["User input %CD19"], errors='coerce')

                # Condition for "N/A": CD19 input is 0 or NaN, or Avg CnC per Group is NaN
                condition_na = (user_input_cd19 == 0) | user_input_cd19.isna() | avg_cnc_group.isna()
                
                # Initialize with np.nan for numeric calculations
                avg_copy_transduced_cell_values = np.full(len(summary_df), np.nan, dtype=float)
                
                # Indices where calculation is possible
                valid_indices = ~condition_na
                
                if np.any(valid_indices): # Check if there are any valid entries to calculate
                    # Perform calculation only for valid indices
                    avg_copy_transduced_cell_values[valid_indices] = (
                        avg_cnc_group[valid_indices] / (user_input_cd19[valid_indices] / 100.0)
                    )
                    # Round the calculated numeric values
                    avg_copy_transduced_cell_values[valid_indices] = np.round(avg_copy_transduced_cell_values[valid_indices], 2)

                # Now, create the final column, converting np.nan to "N/A" string where appropriate
                # and applying "N/A" based on the original condition_na
                summary_df["Average copy number/transduced cell"] = "N/A" # Default to "N/A"
                # Assign calculated rounded values where valid_indices are true
                summary_df.loc[valid_indices, "Average copy number/transduced cell"] = avg_copy_transduced_cell_values[valid_indices]
                # Ensure that any original condition_na (like division by zero) explicitly results in "N/A"
                # This also handles cases where avg_cnc_group might be non-NaN but cd19 input is zero.
                summary_df.loc[condition_na, "Average copy number/transduced cell"] = "N/A"
            else:
                summary_df["Average copy number/transduced cell"] = "N/A" # Default if precursor columns are missing

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
                helper_cols_for_empty_df = [
                    'is_wpre_cv_ok', 'is_rpp30_cv_ok', 'is_partitions_valid_ok',
                    'is_wpre_lloq_ok', 'is_rpp30_lloq_ok'
                ]
                for col_name in helper_cols_for_empty_df:
                    if col_name not in summary_df.columns: # Add if missing
                       summary_df[col_name] = False # Default to False
            
            # Define helper columns to drop after calculations are done
            helper_columns_to_drop = [
                'is_wpre_cv_ok', 'is_rpp30_cv_ok', 'is_partitions_valid_ok',
                'is_wpre_lloq_ok', 'is_rpp30_lloq_ok',
                # 'Sample ID Key' # Uncomment if 'Sample ID Key' should also be dropped
            ]
            # Drop helper columns if they exist, ignore errors if they don't
            summary_df.drop(columns=[col for col in helper_columns_to_drop if col in summary_df.columns], inplace=True, errors='ignore')

            # Select and reorder columns for the final display
            final_columns = ["Sample ID"]
            if "WPRE Concentration" in summary_df.columns:
                final_columns.append("WPRE Concentration")
            if "RPP30 Concentration" in summary_df.columns:
                final_columns.append("RPP30 Concentration")
            if "Copy number/cell" in summary_df.columns:
                final_columns.append("Copy number/cell")
            if "User input %CD19" in summary_df.columns:
                final_columns.append("User input %CD19")
            if "Average copy number/transduced cell" in summary_df.columns:
                final_columns.append("Average copy number/transduced cell")
            if "WPRE Concentration %CV" in summary_df.columns: # Add new %CV column
                final_columns.append("WPRE Concentration %CV")
            if "RPP30 Concentration %CV" in summary_df.columns: # Add new %CV column
                final_columns.append("RPP30 Concentration %CV")
            if "Designation" in summary_df.columns:
                final_columns.append("Designation")
            if "Designation Summary" in summary_df.columns:
                final_columns.append("Designation Summary")
            
            # Ensure only existing columns are selected
            summary_df = summary_df[[col for col in final_columns if col in summary_df.columns]]

# Convert 'Average copy number/transduced cell' to string to handle "N/A" for display
            if "Average copy number/transduced cell" in summary_df.columns:
                summary_df["Average copy number/transduced cell"] = summary_df["Average copy number/transduced cell"].astype(str)
            st.dataframe(summary_df)

            # --- Export Summary Table as Excel ---
            excel_buffer = io.BytesIO()
            summary_df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"Summary_Table_{timestamp}.xlsx"
            st.download_button(
                label="Export Summary Table as Excel",
                data=excel_buffer,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error creating summary table: {e}")
            st.write("Please ensure the CSV contains 'Sample/NTC/Control', a target column ('Target' or 'Target (Name)'), and a concentration column ('Conc. [copies/µL]' or 'Conc. [cp/ÂµL] (dPCR reaction)') and that the target column contains 'WPRE' and 'RPP30' values.")
    else:
        st.warning("Could not create summary table. Required columns ('Sample/NTC/Control', a target column ['Target' or 'Target (Name)'], and a concentration column ['Conc. [copies/µL]' or 'Conc. [cp/ÂµL] (dPCR reaction)']) not found in the uploaded CSV.")