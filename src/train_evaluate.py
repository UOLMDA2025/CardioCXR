from sklearn.feature_selection import RFE
from sklearn.isotonic import IsotonicRegression
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapz
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import os
from tqdm import tqdm
from preprocessing import load_and_preprocess # make sure prpeprocessing.py is placed in the same folder


x_train, y_train, x_val, y_val, x_test, y_test, feature_cols, target_cols = load_and_preprocess()

alpha = 0.05
pos_threshold = 100
ecg_features = [f for f in feature_cols if f not in ['age', 'gender', 'time_diff_sec']]


def select_best_features_rfe(X_train, y_train, X_val, y_val, ecg_features):
    """
    Runs RFE (Recursive Feature Elimination) to pick the optimal number of features
    for an XGBoost model, making sure at least 4 ECG-related features are included.
    """
    best_score = 0
    best_n = 4
    best_features = None
    
    # Try selecting between 4 features up to all features
    for n in range(4, X_train.shape[1] + 1):
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        rfe = RFE(model, n_features_to_select=n)
        X_train_sel = rfe.fit_transform(X_train, y_train)
        X_val_sel = rfe.transform(X_val)

        selected = X_train.columns[rfe.get_support()].tolist()
        n_ecg = len([f for f in selected if f in ecg_features])
        if n_ecg < 4:
            continue

        model.fit(X_train_sel, y_train)
        y_val_pred_proba = model.predict_proba(X_val_sel)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_proba)

        if auc > best_score:
            best_score = auc
            best_n = n
            best_features = selected

    return best_features, best_score, best_n

auc_results = {}
# Loop over each target variable and train an optimized model
for target_name in target_cols:
    positive_count = y_train[target_name].sum()
    if positive_count < pos_threshold:
        print(f"{target_name} skipped (not enough positive samples: {positive_count})")
        continue

    print(f"Processing target: {target_name}")

    y_train_i = y_train[target_name].values
    y_val_i = y_val[target_name].values
    y_test_i = y_test[target_name].values

    # Feature selection step
    features_selected, val_auc, n_features = select_best_features_rfe(x_train, y_train_i, x_val, y_val_i, ecg_features)
    print(f"{target_name} selected {n_features} features, Val AUC={val_auc:.4f}: {features_selected}")

    x_train_sub = x_train[features_selected]
    x_val_sub = x_val[features_selected]
    x_test_sub = x_test[features_selected]

    model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        max_depth=1,
        n_estimators=1000,
        early_stopping_rounds=10
    )

    model.fit(x_train_sub, y_train_i, eval_set=[(x_val_sub, y_val_i)], verbose=False)

    y_test_preds = model.predict_proba(x_test_sub)[:, 1]
    base_auc = roc_auc_score(y_test_i, y_test_preds)

    # Bootstrapping for confidence intervals
    iterations = 1000
    bootstrap_aucs_internal = []
    for _ in tqdm(range(iterations), desc=f"Bootstrap {target_name}"):
        sample_indices = np.random.choice(len(y_test_i), len(y_test_i), replace=True)
        y_test_bootstrap = y_test_i[sample_indices]
        y_test_preds_bootstrap = y_test_preds[sample_indices]
        if len(np.unique(y_test_bootstrap)) > 1:
            auc_bootstrap_int = roc_auc_score(y_test_bootstrap, y_test_preds_bootstrap)
            bootstrap_aucs_internal.append(auc_bootstrap_int)

    bootstrap_aucs_internal = np.array(bootstrap_aucs_internal)
    auc_diff_int = bootstrap_aucs_internal - base_auc
    low_auc_int = base_auc + np.percentile(auc_diff_int, ((1.0 - alpha) / 2.0) * 100)
    high_auc_int = base_auc + np.percentile(auc_diff_int, (alpha + ((1.0 - alpha) / 2.0)) * 100)

    # Store results
    auc_results[target_name] = {
        'AUROC': base_auc,
        'CI_lower': low_auc_int,
        'CI_upper': high_auc_int
    }

    prevalence_int = y_train_i.sum() + y_val_i.sum() + y_test_i.sum()
    prevalence_int = round((prevalence_int / (len(y_train_i) + len(y_val_i) + len(y_test_i))) * 100, 2)

    # AUROC PLOT
    fpr, tpr, _ = roc_curve(y_test_i, y_test_preds)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=3,
             label=(
                f'AUROC: {base_auc:.3f} ({low_auc_int:.3f}, {high_auc_int:.3f})\n'
                f'Prevalence: {prevalence_int}% | {n_features} features'
             ))
    plt.plot([0, 1], [0, 1], 'k-.', label='Random classifier', lw=3)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'{target_name}', fontsize=16)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.tight_layout()
    fig.patch.set_alpha(0.0)
    fig.set_facecolor('none')
    plt.savefig(f'figures/AUROC_{target_name}.png', dpi=600, transparent=True)
    plt.show()


    # Calibration (isotonic)
    probs_val = model.predict_proba(x_val_sub)[:, 1]
    sorted_idx = np.argsort(probs_val)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs_val[sorted_idx], y_val_i[sorted_idx])

    y_test_preds_calibrated = iso.transform(y_test_preds)

    prob_true, prob_pred = calibration_curve(y_test_i, y_test_preds_calibrated, n_bins=10, strategy='quantile')
    bin_counts = np.histogram(y_test_preds_calibrated, bins=len(prob_true))[0]
    total = np.sum(bin_counts)
    ece = np.sum((bin_counts / total) * np.abs(prob_true - prob_pred))

    print(f"Calibration error (ECE) for {target_name}: {ece:.4f}")

    fig = plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linestyle=':', label='Calibration', color='red', linewidth=3, markersize=10)
    plt.plot([0, 1], [0, 1], label='Perfectly calibrated', color='black', linewidth=3)
    max_val = max(prob_pred.max(), prob_true.max())
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.xlabel('Mean predicted probability', fontsize=16)
    plt.ylabel('True probability', fontsize=16)
    plt.title(f"{target_name}", fontsize=16)
    plt.legend(fontsize=16, framealpha=0)
    plt.grid(True)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=12)
    fig.patch.set_alpha(0.0)
    fig.set_facecolor('none')
    plt.savefig(f"figures/calibration_{target_name}.png", dpi=600, transparent=True)
    plt.show()

    # NET BENEFIT ANALYSIS
    def compute_net_benefit(y_true, y_prob, thresholds):
        N = len(y_true)
        net_benefits = []

        for pt in thresholds:
            y_pred = y_prob >= pt
            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            net_benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
            net_benefits.append(net_benefit)

        return net_benefits


    max_prob = y_test_preds.max()
    thresholds = np.linspace(0, max_prob+0.1, 100)


    # Compute net benefit for the model int
    nb_model = compute_net_benefit(y_test_i, y_test_preds, thresholds)
    nb_all = compute_net_benefit(y_test_i, np.ones_like(y_test_preds), thresholds)
    nb_none = compute_net_benefit(y_test_i, np.zeros_like(y_test_preds), thresholds)


    from scipy.ndimage import gaussian_filter1d

    nb_model = gaussian_filter1d(nb_model, sigma=1)
    nb_none = gaussian_filter1d(nb_none, sigma=1)
    nb_all = gaussian_filter1d(nb_all, sigma=1)

    #nb_model_e = gaussian_filter1d(nb_model_e, sigma=1)
    #nb_none_e = gaussian_filter1d(nb_none_e, sigma=1)
    #nb_all_e = gaussian_filter1d(nb_all_e, sigma=1)

    # AUNBCs
    aunbc_int = trapz(nb_model, thresholds)
    aunbc_int_all = trapz(nb_all, thresholds)
    aunbc_int_none = trapz(nb_none, thresholds)


    fig = plt.figure(figsize=(8, 6))
    plt.plot(thresholds, nb_model, label="Model", linestyle=':', color='black', linewidth=3)
    plt.plot(thresholds, nb_all, label="Refer All", linestyle='-.', color='blue', linewidth=3)
    plt.plot(thresholds, nb_none, label="Refer None", linestyle='--', color='red', linewidth=3)

    plt.axhline(0, color='grey', linewidth=3)


    nb_model = np.array(nb_model)
    idx_neg = np.where(nb_model < 0)[0]

    # Set y_min to the second negative value, or to the first if only one exists, or to a default
    if len(idx_neg) >= 2:
        y_min = nb_model[idx_neg[1]]
    elif len(idx_neg) == 1:
        y_min = nb_model[idx_neg[0]]
    else:
        y_min = -0.0001  # or another default

    max_val = max(np.max(nb_model), np.max(nb_all), np.max(nb_none))

    plt.ylim(y_min, max_val)


    idx_x = np.where(nb_model <= 0)[0]
    if len(idx_x) > 0:
        x_max = thresholds[idx_x[0]]
    else:
        x_max = thresholds[-1]
    plt.xlim(thresholds[0], x_max)


    plt.xlabel("Threshold Probability (pt)", fontsize=16)
    plt.ylabel("Net Benefit", fontsize=16)
    plt.title(f"{target_name}", fontsize=16)
    plt.legend(fontsize=16, framealpha=0)
    plt.grid(True)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set both x and y tick label font size
    # Remove background
    fig.patch.set_alpha(0.0)  # Make figure background transparent
    fig.set_facecolor('none')  # Make axes background transparent
    plt.savefig(f"figures/net_benefit_{target_name}.png", dpi=600, transparent=True)
    plt.show()


    # SHAP ANALYSIS
    feature_name_map = {
        'rr_interval': 'RR Interval',
        'qrs_axis': 'QRS Axis',
        'p_axis': 'P Axis',
        't_axis': 'T Axis',
        'p_wave_duration': 'P Wave Duration',
        'qrs_duration': 'QRS Duration',
        'pr_segment': 'PR Segment',
        'qt_segment': 'QT Segment',
        'qrs_t_interval': 'QRS-T Interval',
        'pt_interval': 'PT Interval',
        'QTc': 'QTc',
        'p_to_rr_ratio': 'P-to-RR Ratio',
        'qrs_to_rr_ratio': 'QRS-to-RR Ratio',
        'qt_to_rr_ratio': 'QT-to-RR Ratio',
        'pr_to_qt_ratio': 'PR-to-QT Ratio',
        'p_qrs_axis_diff': 'P-QRS Axis Diff',
        'qrs_t_axis_diff': 'QRS-T Axis Diff',
        'p_t_axis_diff': 'P-T Axis Diff',
        'time_diff_sec': 'Time Difference (s)'
    }

    # Rename columns for better display
    x_train_sub_renamed = x_train_sub.rename(columns=feature_name_map)

    explainer = shap.TreeExplainer(model, x_train_sub_renamed)
    shap_values = explainer(x_train_sub_renamed)

    # Ensure the SHAP plot is saved correctly
    fig, ax = plt.subplots(figsize=(6, 6))
    shap.plots.beeswarm(
        shap_values,
        max_display=min(6, shap_values.shape[1]),
        color=plt.get_cmap("RdBu_r"),
        show=False
    )

    plt.xlabel('SHAP Value (impact on model output)', fontsize=18)
    plt.ylabel('Features', fontsize=18)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)

    plt.title(f'{target_name}', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    plt.tight_layout()
    plt.savefig(f'figures/shap_{target_name}.png', dpi=600, transparent=True)
    plt.show()
    plt.close()
# Summary table for all targets
print("\nAUROC + Confidence Intervals for all targets:\n")
for target, res in auc_results.items():
    print(f"{target:30} | AUROC: {res['AUROC']:.3f} [{res['CI_lower']:.3f}, {res['CI_upper']:.3f}]")



