import pandas as pd
import numpy as np
from stratify import stratified_subsets # make sure stratify.py is placed in the same folder

def load_and_preprocess():
    # Load the data
    df_ecg = pd.read_csv('machine_measurements.csv') # ECG measurements
    cxr_labels = pd.read_csv("cxr-lt-2024/labels.csv")  # Chest X-ray labels
    cxr_meta   = pd.read_csv("mimic-cxr-2.0.0-metadata.csv.gz") # # Metadata with CXR time info
    patients = pd.read_csv('mimic_iv_icu/patients.csv.gz', compression='gzip') # Patient demographics

    # unique CXR target columns
    target_cols = [
        'Adenopathy','Atelectasis','Azygos Lobe','Bulla',
        'Calcification of the Aorta','Cardiomegaly','Cardiomyopathy',
        'Clavicle Fracture','Consolidation','Edema','Emphysema',
        'Enlarged Cardiomediastinum','Fibrosis','Fissure','Fracture',
        'Granuloma','Hernia','Hilum','Hydropneumothorax','Infarction',
        'Infiltration','Kyphosis','Lobar Atelectasis','Lung Lesion',
        'Lung Opacity','Mass','Nodule','Normal','Osteopenia',
        'Pleural Effusion','Pleural Other','Pleural Thickening',
        'Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax',
        'Pulmonary Embolism','Pulmonary Hypertension','Rib Fracture',
        'Round(ed) Atelectasis','Scoliosis','Subcutaneous Emphysema',
        'Support Devices','Tortuous Aorta','Tuberculosis'
    ]

    # CXR preprocessing
    cxr_labels["study_id"] = cxr_labels["study_id"].str.lstrip("s")

    # add cxr time metadata
    cxr_meta = cxr_meta.loc[cxr_meta["ViewPosition"].isin(["PA", "AP"])]
    cxr_meta["cxr_time"] = pd.to_datetime(
        cxr_meta["StudyDate"].astype(str)
        + cxr_meta["StudyTime"].astype(str).str.split(".").str[0].str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )
    cxr_meta = cxr_meta[["dicom_id", "subject_id", "cxr_time"]]

    cxr_labels.drop(columns=["subject_id"], inplace=True, errors="ignore")
    cxr = cxr_labels.merge(cxr_meta, on="dicom_id", how="left")
    cxr = cxr.dropna(subset=["cxr_time"])

    cols_final = ["subject_id", "cxr_time", "dicom_id"] + target_cols
    cxr = cxr[cols_final]

    # ECG preprocessing

    # degree error
    df_ecg.loc[(df_ecg['qrs_axis'] < -360) | (df_ecg['qrs_axis'] > 360), 'qrs_axis'] = np.nan
    df_ecg.loc[(df_ecg['t_axis'] < -360) | (df_ecg['t_axis'] > 360), 't_axis'] = np.nan
    df_ecg.loc[(df_ecg['p_axis'] < -360) | (df_ecg['p_axis'] > 360), 'p_axis'] = np.nan

    # msec error
    df_ecg.loc[(df_ecg['p_onset'] < 0) | (df_ecg['p_onset'] > 5000), 'p_onset'] = np.nan
    df_ecg.loc[(df_ecg['p_end'] < 0) | (df_ecg['p_end'] > 5000), 'p_end'] = np.nan
    df_ecg.loc[(df_ecg['qrs_onset'] < 0) | (df_ecg['qrs_onset'] > 5000), 'qrs_onset'] = np.nan
    df_ecg.loc[(df_ecg['qrs_end'] < 0) | (df_ecg['qrs_end'] > 5000), 'qrs_end'] = np.nan
    df_ecg.loc[(df_ecg['t_end'] < 0) | (df_ecg['t_end'] > 5000), 't_end'] = np.nan
    df_ecg.loc[(df_ecg['rr_interval'] <= 0) | (df_ecg['rr_interval'] > 5000), 'rr_interval'] = np.nan

    # feature engineering
    df_ecg['p_wave_duration'] = df_ecg['p_end'] - df_ecg['p_onset']
    df_ecg['qrs_duration'] = df_ecg['qrs_end'] - df_ecg['qrs_onset']
    df_ecg['pr_segment'] = df_ecg['qrs_onset'] - df_ecg['p_onset']
    df_ecg['qt_segment'] = df_ecg['t_end'] - df_ecg['qrs_onset']
    df_ecg['qrs_t_interval'] = df_ecg['t_end'] - df_ecg['qrs_end']
    df_ecg['pt_interval'] = df_ecg['t_end'] - df_ecg['p_onset']

    # outliers with intervals < 0
    duration_cols = ['p_wave_duration', 'qrs_duration', 'pr_segment', 'qt_segment', 'qrs_t_interval', 'pt_interval']
    for col in duration_cols:
        df_ecg.loc[df_ecg[col] <= 0, col] = np.nan


    df_ecg['QTc'] = np.where(
        (df_ecg['rr_interval'] > 0),
        df_ecg['qt_segment'] / np.sqrt(df_ecg['rr_interval'] / 1000),
        np.nan
    )

    # Ratios of intervals to RR interval
    df_ecg['p_to_rr_ratio'] = np.where(
        (df_ecg['p_wave_duration'] > 0) & (df_ecg['rr_interval'] > 0),
        df_ecg['p_wave_duration'] / df_ecg['rr_interval'],
        np.nan
    )

    df_ecg['qrs_to_rr_ratio'] = np.where(
        (df_ecg['qrs_duration'] > 0) & (df_ecg['rr_interval'] > 0),
        df_ecg['qrs_duration'] / df_ecg['rr_interval'],
        np.nan
    )

    df_ecg['qt_to_rr_ratio'] = np.where(
        (df_ecg['qt_segment'] > 0) & (df_ecg['rr_interval'] > 0),
        df_ecg['qt_segment'] / df_ecg['rr_interval'],
        np.nan
    )

    df_ecg['pr_to_qt_ratio'] = np.where(
        (df_ecg['pr_segment'] > 0) & (df_ecg['qt_segment'] > 0),
        df_ecg['pr_segment'] / df_ecg['qt_segment'],
        np.nan
    )


    df_ecg.replace([np.inf, -np.inf], np.nan, inplace=True)

    # axes diff
    def angle_diff(a, b):
        # calculate the smallest absolute difference between two angles
        diff = (a - b).abs() % 360
        return diff.apply(lambda x: x if x <= 180 else 360 - x)

    df_ecg['p_qrs_axis_diff'] = angle_diff(df_ecg['p_axis'], df_ecg['qrs_axis'])
    df_ecg['qrs_t_axis_diff'] = angle_diff(df_ecg['qrs_axis'], df_ecg['t_axis'])
    df_ecg['p_t_axis_diff'] = angle_diff(df_ecg['p_axis'], df_ecg['t_axis'])

    df_ecg.drop(columns=['p_onset','p_end','qrs_onset', 'qrs_end', 't_end'], inplace=True)

    # extract CXR-ECG pairs
    WINDOW = pd.Timedelta(hours=24) # set max time window between cxr and ecg

    cxr["subject_id"] = cxr["subject_id"].astype(int)
    df_ecg["subject_id"] = df_ecg["subject_id"].astype(int)
    cxr["cxr_time"] = pd.to_datetime(cxr["cxr_time"])
    df_ecg["ecg_time"] = pd.to_datetime(df_ecg["ecg_time"])
    cxr = cxr.rename(columns={"study_id": "study_id_cxr"})

    merged = cxr.merge(df_ecg, on="subject_id", how="inner", suffixes=("", "_ecg"))
    merged["timediff"] = merged["cxr_time"] - merged["ecg_time"]

    # Keep the closest ECG for each CXR
    valid_pairs = merged.loc[
        (merged["timediff"] > pd.Timedelta(0)) & (merged["timediff"] <= WINDOW)
    ].copy()
    valid_pairs["time_diff_sec"] = valid_pairs["timediff"].dt.total_seconds()

    valid_pairs = (
        valid_pairs
        .sort_values(["dicom_id", "time_diff_sec"])
        .groupby("dicom_id", as_index=False)
        .first()
    )

    if "study_id_ecg" not in valid_pairs.columns and "study_id" in valid_pairs.columns:
        valid_pairs = valid_pairs.rename(columns={"study_id": "study_id_ecg"})

    cols_to_add = [c for c in valid_pairs.columns if c not in cxr.columns and c != "dicom_id"]
    df = cxr.merge(valid_pairs[["dicom_id"] + cols_to_add], on="dicom_id", how="inner")

    # get the demographics(age, gender)
    patients_demog = patients[["subject_id", "anchor_age", "anchor_year", "gender"]].drop_duplicates("subject_id")

    df = df.merge(patients_demog, on="subject_id", how="left", validate="m:1")
    # Calculate actual age at exam time
    exam_year = (
        pd.to_datetime(df["cxr_time"], errors="coerce")
        .fillna(pd.to_datetime(df["ecg_time"], errors="coerce"))
        .dt.year
    )

    df["age"] = df["anchor_age"] + (exam_year - df["anchor_year"])
    df.loc[df["anchor_age"] >= 300, "age"] = 90
    df.drop(columns=["anchor_age", "anchor_year"], inplace=True)

    df["gender"] = df["gender"].apply(lambda x: 1 if x == "M" else (0 if x == "F" or x == 0 else np.nan))

    keep_cols = [
        "subject_id", "age", "gender", "time_diff_sec",
        "rr_interval", "qrs_axis", "p_axis", "t_axis",
        'p_wave_duration',
        'qrs_duration',
        'pr_segment',
        'qt_segment',
        'qrs_t_interval',
        'pt_interval',
        'QTc',
        'p_to_rr_ratio',
        'qrs_to_rr_ratio',
        'qt_to_rr_ratio',
        'pr_to_qt_ratio',
        'p_qrs_axis_diff',
        'qrs_t_axis_diff',
        'p_t_axis_diff'
    ] + target_cols


    df = df[keep_cols]

    df["age"].fillna(df["age"].median(), inplace=True)
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0]).astype(int)

    # create stratification labels and assign folds

    def get_active_labels(row, target_cols):
        # return list of target columns where value is 1
        active_labels = [col for col in target_cols if row[col] == 1]
        return active_labels

    # create quartile-based age bins
    df['age_bin'] = pd.qcut(df['age'], q=4, duplicates='drop')
    unique_intervals = df['age_bin'].cat.categories
    bin_labels = {interval: f'{interval.left:.0f}-{interval.right:.0f}' for interval in unique_intervals}
    df['age_bin'] = df['age_bin'].map(bin_labels)

    df['gender'] = df['gender'].astype(str)

    # Combine active labels with age bin and gender into a merged stratification label
    df['merged_strat'] = df.apply(
        lambda row: get_active_labels(row, target_cols) + [row['age_bin'], row['gender']],
        axis=1
    )
    df['merged_strat'] = df['merged_strat'].apply(lambda lst: [x for x in lst if pd.notna(x)])
    col_label = "merged_strat"
    col_group = "subject_id"

    # Perform stratification based on merged labels, make sure stratify.py is placed in the same folder
    res = stratified_subsets(
        df,
        col_label,
        [0.05]*20,
        col_group=col_group,
        label_multi_hot=False,
        random_seed=42
    )

    df['strat_fold'] = res
    df.drop(columns= ["merged_strat", "subject_id", "age_bin"], inplace=True, errors='ignore')
    
    df['gender'] = df['gender'].astype(int)
    #train-val-test split 18:1:1

    train_df = df[df["strat_fold"] <= 17].reset_index(drop=True)
    val_df   = df[df["strat_fold"] == 18].reset_index(drop=True)
    test_df  = df[df["strat_fold"] == 19].reset_index(drop=True)

    for d in (train_df, val_df, test_df):
        d.drop(columns=["strat_fold"], inplace=True)

    feature_cols = [c for c in df.columns if c not in target_cols + ["strat_fold"]]

    X_train, y_train = train_df[feature_cols], train_df[target_cols]
    X_val,   y_val   = val_df[feature_cols],   val_df[target_cols]
    X_test,  y_test  = test_df[feature_cols],  test_df[target_cols]


    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, target_cols




