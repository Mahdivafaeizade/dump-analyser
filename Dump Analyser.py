import matplotlib.pyplot as plt
import os
import tarfile
import shutil
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import time
from datetime import datetime
import streamlit as st
import pymysql
from sqlalchemy import create_engine

# ---------------- first---------------- #
def extract_and_copy_files(directory, target_directory):
    extracted_files = []
    copied_files = []
    log_messages = []

    for filename in os.listdir(directory):
        if filename.endswith(".tar.gz"):
            file_path = os.path.join(directory, filename)
            try:
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=directory)
                    extracted_files.append(filename)
            except Exception as e:
                log_messages.append(f"❌ خطا در استخراج {filename}: {str(e)}")

    for folder in os.listdir(directory):
        if folder.startswith(("QN", "BU", "AG", "KB", "HN", "AR", "KS", "ZN")):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                latest_folder = None
                latest_time = None

                for subfolder in os.listdir(folder_path):
                    if subfolder.startswith("AUTOBAKDATA"):
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if os.path.isdir(subfolder_path):
                            try:
                                date_str = subfolder[11:18]
                                time_str = subfolder[18:24]
                                full_datetime_str = date_str + time_str
                                folder_datetime = datetime.strptime(full_datetime_str, "%Y%m%d%H%M%S")
                                if latest_time is None or folder_datetime > latest_time:
                                    latest_folder = subfolder
                                    latest_time = folder_datetime
                            except ValueError as e:
                                log_messages.append(f"⛔ تاریخ نامعتبر در پوشه: {subfolder} - {e}")

                if latest_folder:
                    latest_subfolder_path = os.path.join(folder_path, latest_folder)
                    for file in os.listdir(latest_subfolder_path):
                        file_path = os.path.join(latest_subfolder_path, file)
                        new_file_name = f"{os.path.splitext(file)[0]}_{folder}.XML.gz"
                        new_file_path = os.path.join(target_directory, new_file_name)
                        shutil.copy2(file_path, new_file_path)
                        copied_files.append(new_file_name)
                else:
                    log_messages.append(f"🚫 پوشه معتبر AUTOBAKDATA در {folder} یافت نشد.")

    return extracted_files, copied_files, log_messages

def clean_temp_directory(temp_directory):
    deleted_files = []
    for filename in os.listdir(temp_directory):
        if filename.startswith("neinfo_") and filename.endswith(".XML.gz"):
            file_path = os.path.join(temp_directory, filename)
            os.remove(file_path)
            deleted_files.append(filename)
    return deleted_files

def extract_gz_files_in_temp(temp_directory):
    extracted_xml_files = []
    for filename in os.listdir(temp_directory):
        if filename.endswith(".XML.gz"):
            gz_path = os.path.join(temp_directory, filename)
            xml_filename = filename[:-3]
            xml_path = os.path.join(temp_directory, xml_filename)
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(xml_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                extracted_xml_files.append(xml_filename)
                os.remove(gz_path)
            except Exception as e:
                extracted_xml_files.append(f"❌ خطا در اکسترکت {filename}: {str(e)}")
    return extracted_xml_files

# ---------------- second---------------- #
def convert_xml_to_csv(xml_dir):
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.XML')]
    all_logs = []
    if not xml_files:
        return 0, 0.0, ["هیچ فایل XML یافت نشد."]

    start_time = time.time()

    for xml_file in xml_files:
        data = []
        tag_stack = []
        current_mo = None
        inside_attributes_section = False
        current_function_type = None
        current_local_cell_id = current_mcc = current_mnc = current_RncId = "-"
        current_CellId = current_localCellName = current_neighbourCellName = current_group_id = "-"
        pending_row = None
        waiting_for_comment = False
        site_name = xml_file.split('_')[-1].split('.')[0]
        file_path = os.path.join(xml_dir, xml_file)
        try:
            for event, element in ET.iterparse(file_path, events=("start", "end", "comment")):
                if event == "comment":
                    if waiting_for_comment and pending_row is not None:
                        pending_row['Comment'] = element.text.strip() if element.text else "-"
                        data.append(pending_row)
                        pending_row = None
                        waiting_for_comment = False
                    continue
                tag_name = element.tag.split('}')[-1]
                if event == "start":
                    if tag_name == "syndata" and "FunctionType" in element.attrib:
                        current_function_type = element.attrib["FunctionType"]
                    tag_stack.append(tag_name)
                    if tag_name == "attributes":
                        inside_attributes_section = True
                        if len(tag_stack) > 1:
                            current_mo = tag_stack[-2]
                elif event == "end":
                    tag_value = element.text.strip() if element.text else "-"
                    if inside_attributes_section and tag_name != "attributes":
                        if tag_name == "LocalCellId": current_local_cell_id = tag_value
                        elif tag_name == "Mcc": current_mcc = tag_value
                        elif tag_name == "Mnc": current_mnc = tag_value
                        elif tag_name == "RncId": current_RncId = tag_value
                        elif tag_name == "CellId": current_CellId = tag_value
                        elif tag_name == "LocalCellName": current_localCellName = tag_value
                        elif tag_name == "NeighbourCellName": current_neighbourCellName = tag_value
                        elif "GroupId" in tag_name and tag_value.isdigit() and len(tag_value) == 1:
                            current_group_id = tag_value
                        pending_row = {
                            'Date': datetime.now().strftime('%Y-%m-%d'),
                            'Site': site_name,
                            'FunctionType': current_function_type,
                            'MO': current_mo,
                            'Tag': tag_name,
                            'LocalCellId': current_local_cell_id,
                            'MCC': current_mcc,
                            'MNC': current_mnc,
                            'RNC ID': current_RncId,
                            'CellId': current_CellId,
                            'LocalCellName': current_localCellName,
                            'NeighbourCellName': current_neighbourCellName,
                            'GroupId': current_group_id,
                            'Value': tag_value,
                            'Comment': "-"
                        }
                        waiting_for_comment = True
                    if tag_name == "attributes":
                        inside_attributes_section = False
                    tag_stack.pop()
                    if pending_row and not waiting_for_comment:
                        data.append(pending_row)
                        pending_row = None
            if pending_row:
                data.append(pending_row)
            df = pd.DataFrame(data)
            output_csv = os.path.join(xml_dir, f"{xml_file.replace('.XML', '')}.csv")
            df.to_csv(output_csv, index=False)
        except Exception as e:
            all_logs.append(f"❌ خطا در پردازش {xml_file}: {e}")
    total_time = time.time() - start_time
    return len(xml_files), total_time, all_logs


# ---------------- (Accuracy---------------- #
def calculate_accuracy(baseline_file, csv_folder):
    try:
        baseline_df = pd.read_csv(baseline_file, dtype=str)
    except Exception as e:
        return None, f"❌ خطا در خواندن فایل baseline: {e}"

    baseline_df.columns = [col.strip() for col in baseline_df.columns]
    baseline_df = baseline_df[['MO', 'Tag', 'Key', 'Parameter', 'Target Value']]
    main_key_lookup = baseline_df.set_index('Key')['Target Value'].to_dict()

    csv_files = [
        f for f in os.listdir(csv_folder)
        if f.endswith(".csv") and not f.startswith("discrepancy") and "Accuracy" not in f and "Final_Verified" not in f
    ]

    all_rows = []
    province_map = {'AG': 'AG', 'AR': 'AR', 'BU': 'BU', 'HN': 'HN', 'QN': 'QN', 'ZN': 'ZN', 'KS': 'KS'}

    for file in csv_files:
        file_path = os.path.join(csv_folder, file)
        try:
            df = pd.read_csv(file_path, dtype=str)
        except Exception as e:
            continue

        df.columns = [col.strip() for col in df.columns]
        if 'GroupId' not in df.columns or 'MO' not in df.columns or 'Tag' not in df.columns:
            continue

        base_name = file.replace(".csv", "")
        province_code = next((code for code in province_map if code in base_name), '-')

        df['GroupId'] = df['GroupId'].astype(str).str.strip()
        df['Key'] = (df['MO'].str.strip() + "_" + df['Tag'].str.strip() + "_" + df['GroupId']).str.upper()
        df = df[df['GroupId'].notna() & (df['GroupId'] != '')]

        for idx, row in df.iterrows():
            mo = row['MO'].strip()
            tag = row['Tag'].strip()
            groupid = row['GroupId'].strip()
            key = row['Key']
            value = str(row.get('Value', '')).strip()
            site = str(row.get('Site', '-')).strip()

            if key in main_key_lookup:
                target_value = str(main_key_lookup[key]).strip()
                result = "OK" if value == target_value else "Mismatch"

                base_info = {
                    'Province': province_code,
                    'Site': site,
                    'MO': mo,
                    'Tag': tag,
                    'GroupId': groupid,
                    'Value': value,
                    'Target Value': target_value,
                    'Result': result
                }

                comment = str(row.get('Comment', '')).strip()
                if comment and ':' in comment:
                    for item in comment.split(';'):
                        if ':' in item:
                            param_name, param_value = item.split(':', 1)
                            param_name = param_name.strip()
                            param_value = param_value.strip().lower()
                            if param_value == 'on': param_value = '1'
                            elif param_value == 'off': param_value = '0'
                            else: continue

                            param_target_value = '-'
                            param_result = 'Not Found'

                            full_param_key = f"{mo.strip().upper()}_{tag.strip().upper()}_{groupid.strip().upper()}"
                            match = baseline_df[
                                (baseline_df['MO'].str.upper() == mo.upper()) &
                                (baseline_df['Tag'].str.upper() == tag.upper()) &
                                (baseline_df['Parameter'].str.upper() == param_name.upper()) &
                                (baseline_df['Key'].str.upper() == full_param_key)
                            ]

                            if not match.empty:
                                param_target_value = str(match.iloc[0]['Target Value']).strip()
                                param_result = 'OK' if param_value == param_target_value else 'Mismatch'

                            full_row = base_info.copy()
                            full_row.update({
                                'Parameter': param_name,
                                'Parameter Value': param_value,
                                'Parameter Target Value': param_target_value,
                                'Parameter Result': param_result
                            })
                            all_rows.append(full_row)
                else:
                    full_row = base_info.copy()
                    full_row.update({
                        'Parameter': '-',
                        'Parameter Value': '-',
                        'Parameter Target Value': '-',
                        'Parameter Result': '-'
                    })
                    all_rows.append(full_row)

    final_df = pd.DataFrame(all_rows)
    output_verified = os.path.join(csv_folder, "All_Provinces_Final_Verified_LTE_with_Parameters.csv")
    output_parameter_accuracy = os.path.join(csv_folder, "All_Provinces_Parameter_Accuracy.csv")
    output_tag_accuracy = os.path.join(csv_folder, "All_Provinces_Tag_Accuracy.csv")
    final_df.to_csv(output_verified, index=False)

    final_df['MO_Tag_Parameter_GroupId'] = (
        final_df['MO'].str.strip() + '_' +
        final_df['Tag'].str.strip() + '_' +
        final_df['Parameter'].str.strip() + '_' +
        final_df['GroupId'].str.strip()
    )

    param_df = final_df[final_df['Parameter'] != '-'].copy()
    param_group = param_df.groupby(['MO_Tag_Parameter_GroupId', 'Province'])['Parameter Result']\
        .apply(lambda x: (x == 'OK').mean() * 100).reset_index(name='Accuracy')
    param_pivot = param_group.pivot(index='MO_Tag_Parameter_GroupId', columns='Province', values='Accuracy').reset_index()
    param_pivot.to_csv(output_parameter_accuracy, index=False)

    tag_df = final_df[final_df['Parameter'] == '-'].copy()
    tag_df['MO_Tag_GroupId'] = (
        tag_df['MO'].str.strip() + '_' +
        tag_df['Tag'].str.strip() + '_' +
        tag_df['GroupId'].str.strip()
    )
    tag_group = tag_df.groupby(['MO_Tag_GroupId', 'Province'])['Result']\
        .apply(lambda x: (x == 'OK').mean() * 100).reset_index(name='Accuracy')
    tag_pivot = tag_group.pivot(index='MO_Tag_GroupId', columns='Province', values='Accuracy').reset_index()
    tag_pivot.to_csv(output_tag_accuracy, index=False)

    return [output_verified, output_parameter_accuracy, output_tag_accuracy], "✅ محاسبه Accuracy با موفقیت انجام شد."

# ---------------- UI ---------------- #
st.set_page_config(page_title="Backup XML Accuracy Tool", layout="wide")
st.title("📊 Dump Analyzer Tool")

with st.expander("📦 Step 1: Extract and Prepare Data", expanded=True):
    dir1 = st.text_input("📁 Source Path:", value=r"E:\\Streamlit", key="main_path")
    temp_dir = st.text_input("📂 Temp Folder:", value=r"E:\\Streamlit\\Temp", key="temp_path")
    if st.button("🚀 Run Step 1"):
        if os.path.isdir(dir1) and os.path.isdir(temp_dir):
            with st.spinner("Running Step 1..."):
                extracted, copied, logs = extract_and_copy_files(dir1, temp_dir)
                clean_temp_directory(temp_dir)
                extract_gz_files_in_temp(temp_dir)
            st.success("✅ Step 1 completed successfully.")
            for f in extracted:
                st.code(f)
            if logs:
                for l in logs:
                    st.warning(l)
        else:
            st.error("❌ Please enter valid paths.")

with st.expander("📄 Step 2: Convert XML to CSV", expanded=False):
    xml_path = st.text_input("📂 Path to XML files:", value=r"E:\\Streamlit\\Temp", key="xml_path")
    if st.button("🔄 Convert XML to CSV"):
        if os.path.isdir(xml_path):
            with st.spinner("Converting XML files..."):
                count, duration, logs = convert_xml_to_csv(xml_path)
            st.success(f"✅ {count} files processed in {duration:.2f} seconds.")
            if logs:
                for l in logs:
                    st.warning(l)
        else:
            st.error("❌ The provided path is invalid.")

with st.expander("🎯 Step 3: Calculate Accuracy", expanded=False):
    baseline_file = st.text_input("📘 Path to Baseline CSV file:", value=r"F:\\dump4G\\New folder\\New folder\\discrepancy.csv", key="baseline")
    csv_folder = st.text_input("📂 Path to CSV folder:", value=r"E:\\Streamlit\\Temp", key="csv_folder")
    if st.button("📊 Run Accuracy Calculation"):
        with st.spinner("Processing Accuracy..."):
            outputs, message = calculate_accuracy(baseline_file, csv_folder)
        if outputs:
            st.success(message)
            for file in outputs:
                st.info(f"📁 {file}")
        else:
            st.error(message)

# --- Show Accuracy Outputs ---
with st.expander("📈 View Accuracy Output Files", expanded=False):
    if st.button("📂 Show All_Provinces_Parameter_Accuracy"):
        param_file = os.path.join(csv_folder, "All_Provinces_Parameter_Accuracy.csv")
        if os.path.isfile(param_file):
            df_param = pd.read_csv(param_file)
            st.dataframe(df_param)
        else:
            st.warning("⚠️ File not found: All_Provinces_Parameter_Accuracy.csv")

    if st.button("📂 Show All_Provinces_Final_Verified_LTE_with_Parameters"):
        tag_file = os.path.join(csv_folder, "All_Provinces_Final_Verified_LTE_with_Parameters.csv")
        if os.path.isfile(tag_file):
            df_tag = pd.read_csv(tag_file)
            st.dataframe(df_tag)
        else:
            st.warning("⚠️ File not found: All_Provinces_Final_Verified_LTE_with_Parameters.csv")


# ---------------- Visualization: Parameter Accuracy by Province ---------------- #
import matplotlib.pyplot as plt

with st.expander("📊 Visualize Parameter Accuracy per Province", expanded=False):
    param_file = os.path.join(csv_folder, "All_Provinces_Parameter_Accuracy.csv")
    if os.path.isfile(param_file):
        df_param = pd.read_csv(param_file)

        # Melt DataFrame to have Province and Accuracy columns
        melted = df_param.melt(id_vars=['MO_Tag_Parameter_GroupId'], var_name='Province', value_name='Accuracy')
        melted = melted.dropna(subset=['Accuracy'])

        # Group by Province to compute average accuracy
        summary = melted.groupby('Province')['Accuracy'].mean().sort_values(ascending=False).round(2)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(summary.index, summary.values, color='skyblue')
        ax.set_title("PARAMETER ACCURACY PER PROVINCE", fontsize=14, weight='bold')
        ax.set_ylabel("ACCURACY (%)")
        ax.set_ylim(0, 105)

        # Label each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        st.pyplot(fig)
    else:
        st.warning("⚠️ File All_Provinces_Parameter_Accuracy.csv not found.")





# 1️⃣ Date range selection from user
start_date, end_date = st.date_input(
    "📅 Select date range (for database filtering):",
    value=(datetime(2024, 1, 1), datetime.today()),
    key="site_date_range"
)

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# 2️⃣ Database query function with date range filter
@st.cache_data(show_spinner="📡 Connecting to database and fetching sites...")
def fetch_sites_from_mysql_range(start_str: str, end_str: str):
    try:
        engine = create_engine("mysql+pymysql://MCI_Importer_3:8KqtPADaqMpcQku2@192.168.15.31:3306/imported_data")
        query = """
            SELECT `Date`,
                   LEFT(`cell`,2) AS Province,
                   CONCAT(LEFT(`cell`,2), SUBSTRING(`cell`,5,4)) AS Site
            FROM new_cell_report_daily
            WHERE `Date` BETWEEN %s AND %s
        """
        df = pd.read_sql(query, con=engine, params=(start_str, end_str))
        return df
    except Exception as e:
        st.error(f"❌ Error connecting or executing SQL: {e}")
        return pd.DataFrame()

# 3️⃣ Step 4 UI - New Site Detection
with st.expander("🛰️ Step 4: New Site Detection", expanded=False):
    if st.button("🔍 Check new sites from database", key="fetch_sites_btn"):
        site_df = fetch_sites_from_mysql_range(start_str, end_str)
        if not site_df.empty:
            st.success(f"✅ {len(site_df)} sites found between {start_str} and {end_str}.")
            st.dataframe(site_df.head())

            # Check against final CSV
            csv_path = os.path.join(csv_folder, "All_Provinces_Final_Verified_LTE_with_Parameters.csv")
            if os.path.isfile(csv_path):
                try:
                    main_df = pd.read_csv(csv_path)
                    matched = main_df[main_df['Site'].isin(site_df['Site'])]

                    st.markdown(f"### 📋 Matching records found in final CSV: {len(matched)} rows")
                    st.dataframe(matched)

                    # Save output
                    output_path = os.path.join(csv_folder, "New_Site.csv")
                    matched.to_csv(output_path, index=False)
                    st.success(f"📁 Output saved: {output_path}")
                except Exception as e:
                    st.error(f"❌ Error reading or saving CSV: {e}")
            else:
                st.warning("⚠️ Final CSV not found: All_Provinces_Final_Verified_LTE_with_Parameters.csv")

# 4️⃣ Free SQL Query Executor
with st.expander("💻 Execute custom SQL query on MySQL", expanded=False):
    query_text = st.text_area("📝 Enter your SQL query below:", height=200)

    if st.button("▶️ Run SQL query"):
        if query_text.strip():
            try:
                engine = create_engine("mysql+pymysql://MCI_Importer_3:8KqtPADaqMpcQku2@192.168.15.31:3306/imported_data")
                df_query_result = pd.read_sql(query_text, con=engine)
                st.success(f"✅ Query executed successfully. Rows returned: {len(df_query_result)}")
                st.dataframe(df_query_result)

                # Download button
                csv_bytes = df_query_result.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download result as CSV", data=csv_bytes, file_name="custom_query_result.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Query execution error: {e}")
        else:
            st.warning("⚠️ Please enter a valid SQL query.")
