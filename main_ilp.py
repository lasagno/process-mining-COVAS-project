import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pm4py

# Robust fallback imports for simplicity evaluation
try:
    from pm4py.evaluation.simplicity.variants import arc_degree as simplicity_metric
except ImportError:
    try:
        from pm4py.algo.evaluation.simplicity.variants import arc_degree as simplicity_metric
    except ImportError:
        simplicity_metric = None


def tune_ilp_miner(sublogs, unique_groups, results_dir="."):
    """
    Performs grid search to find the optimal hyperparameter for the ILP Miner.
    Objective: Maximize sum of (Fitness + Precision) on the main diagonal (each log on its own model).
    """
    print("\n--- Starting Hyperparameter Tuning ---")

    # Define the search grid
    # alpha: controls how much infrequent/noisy behavior is filtered out
    # during ILP-based discovery (0.0 = no filtering, 1.0 = maximum filtering)
    # Kept coarse (6 values) since each ILP solve is expensive across 8 age groups
    param_grid = {
        # 'alpha': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        'alpha': [0.0, 0.2, 0.4, 0.6]
    }

    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))

    best_score = -1.0
    best_params = {}

    # prepare logging of combo scores
    try:
        os.makedirs(results_dir, exist_ok=True)
    except Exception:
        pass
    tuning_log_path = os.path.join(results_dir, "hyperparam_tuning_results.csv")
    records = []

    print(f"Total configurations to test: {len(combinations)}")

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{idx}/{len(combinations)}] Testing {params}...")

        combo_score = 0.0
        valid_combo = True

        for group in unique_groups:
            log = sublogs[group]
            group_start = pd.Timestamp.now()

            try:
                # 1. Discover Model
                net, im, fm = pm4py.discover_petri_net_ilp(
                    log,
                    alpha=params['alpha']
                )

                # 2. Evaluate Fitness (Main Diagonal)
                fit_res = pm4py.fitness_token_based_replay(log, net, im, fm)
                f_val = fit_res.get('log_fitness', fit_res.get('average_trace_fitness', 0.0))

                # 3. Evaluate Precision (Main Diagonal)
                p_val = pm4py.precision_token_based_replay(log, net, im, fm)

                combo_score += (f_val + p_val)

                elapsed = (pd.Timestamp.now() - group_start).total_seconds()
                print(f"    group {group}: fitness={f_val:.3f}, precision={p_val:.3f} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  -> [Warning] Failed on group {group}: {e}")
                valid_combo = False
                break

        # record the result for this combination
        record = {
            'alpha': params['alpha'],
            'combo_score': combo_score if valid_combo else np.nan,
            'valid': bool(valid_combo),
            'tested_at': pd.Timestamp.now()
        }
        records.append(record)

        if valid_combo:
            print(f"  -> Diagonal Sum Score: {combo_score:.4f}")
            if combo_score > best_score:
                best_score = combo_score
                best_params = params
        else:
            print("  -> Skipped configuration due to errors.")

    print(f"\n=== Tuning Complete ===")
    # save tuning log
    try:
        df_log = pd.DataFrame.from_records(records)
        # ensure consistent column order
        cols = ['tested_at', 'alpha', 'combo_score', 'valid']
        df_log = df_log[cols]
        df_log.to_csv(tuning_log_path, index=False)
        print(f"Tuning log saved to: {tuning_log_path}")
    except Exception as e:
        print(f"Failed to save tuning log to {tuning_log_path}: {e}")

    if best_score > 0:
        print(f"Optimal Parameters: {best_params} (Score: {best_score:.4f})")
        return best_params
    else:
        print("No valid configurations found. Falling back to default parameters.")
        return {'alpha': 0.0}


def discover_models_and_simplicity(sublogs, unique_groups, discover_fn, results_dir, miner_label):
    """
    Discovers a Petri net per group using the given discover_fn, computes
    arc-degree simplicity, saves visualizations, and returns the models dict.

    discover_fn: a callable taking a single sublog and returning (net, im, fm)
    """
    models = {}
    simplicity_scores = {}

    print(f"\n--- Discovering Process Models ({miner_label}) & Calculating Simplicity ---")
    for group in unique_groups:
        print(f"Processing group {group}...")
        sublog = sublogs[group]

        net, im, fm = discover_fn(sublog)
        models[group] = (net, im, fm)

        # Calculate Arc-Degree Simplicity
        if simplicity_metric is not None:
            try:
                simplicity_val = simplicity_metric.apply(net)
            except Exception:
                simplicity_val = np.nan
        else:
            simplicity_val = np.nan
        simplicity_scores[group] = simplicity_val

        vis_filename = os.path.join(results_dir, f"petri_net_{group}.png")
        pm4py.save_vis_petri_net(net, im, fm, vis_filename)
        print(f" -> Saved Petri Net to {vis_filename}")

    # Save Simplicity Results
    simplicity_df = pd.DataFrame.from_dict(simplicity_scores, orient='index', columns=['arc_degree_simplicity'])
    simplicity_df.index.name = 'age_group'
    simplicity_df.to_csv(os.path.join(results_dir, "simplicity_metrics.csv"))
    visualize_simplicity(simplicity_df, results_dir, miner_label)

    return models


def cross_evaluate(sublogs, unique_groups, models, results_dir, miner_label):
    """
    Replays every sublog against every discovered model (cross-evaluation),
    producing fitness and precision matrices plus heatmaps.
    """
    fitness_matrix = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)
    precision_matrix = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)

    print(f"\n--- Replaying Logs Against Models ({miner_label} Cross-Evaluation) ---")
    for log_group in unique_groups:
        current_log = sublogs[log_group]
        for model_group in unique_groups:
            net, im, fm = models[model_group]

            try:
                fitness_res = pm4py.fitness_token_based_replay(current_log, net, im, fm)
                fitness_matrix.loc[log_group, model_group] = fitness_res.get('log_fitness',
                                                                             fitness_res.get('average_trace_fitness',
                                                                                             0.0))
            except Exception:
                fitness_matrix.loc[log_group, model_group] = np.nan

            try:
                precision_matrix.loc[log_group, model_group] = pm4py.precision_token_based_replay(current_log, net, im,
                                                                                                  fm)
            except Exception:
                precision_matrix.loc[log_group, model_group] = np.nan

    # Save Matrices
    fitness_matrix.to_csv(os.path.join(results_dir, "fitness_matrix.csv"))
    precision_matrix.to_csv(os.path.join(results_dir, "precision_matrix.csv"))

    # Generate Heatmaps
    save_heatmap(fitness_matrix, f"Cross-Evaluation Fitness Matrix ({miner_label})", "RdBu",
                 os.path.join(results_dir, "fitness_matrix_heatmap.png"))
    save_heatmap(precision_matrix, f"Cross-Evaluation Precision Matrix ({miner_label})", "RdBu",
                 os.path.join(results_dir, "precision_matrix_heatmap.png"))

    return fitness_matrix, precision_matrix


def analyze_process_by_age(xes_file_path, results_dir="process_mining_results"):
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: '{os.path.abspath(results_dir)}'\n")

    print("Loading event log...")
    log_data = pm4py.read_xes(xes_file_path)

    if not isinstance(log_data, pd.DataFrame):
        df = pm4py.convert_to_dataframe(log_data)
    else:
        df = log_data.copy()

    age_col = None
    possible_cols = ["case:age", "age", "case:Age", "Age"]
    for col in possible_cols:
        if col in df.columns:
            age_col = col
            break

    if age_col is None:
        raise ValueError(f"Could not find an age attribute. Checked columns: {possible_cols}.")

    print(f"Found age attribute column: '{age_col}'")

    df = df.dropna(subset=[age_col])
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
    df = df.dropna(subset=[age_col])

    def get_age_bucket(age):
        lower_bound = int(age // 10) * 10
        upper_bound = lower_bound + 9
        return f"{lower_bound}-{upper_bound}"

    df['age_group'] = df[age_col].apply(get_age_bucket)
    unique_groups = sorted(df['age_group'].unique())
    print(f"Identified Age Groups: {unique_groups}")

    sublogs = {group: df[df['age_group'] == group].copy() for group in unique_groups}

    # Add a pooled "Overall" group containing every case, regardless of age,
    # so the tuned model can also be evaluated against an all-ages baseline.
    # Included in tuning too, so alpha is optimized for this group as well.
    # NOTE: "Overall" is by far the largest sublog, and ILP solve time scales
    # steeply with sublog size/complexity -- expect tuning to take noticeably
    # longer per alpha value once this group is included.
    OVERALL_LABEL = "Overall"
    sublogs[OVERALL_LABEL] = df.copy()
    unique_groups = unique_groups + [OVERALL_LABEL]
    print(f"Added combined group: '{OVERALL_LABEL}' (all ages pooled)")
    print(f"Final groups for tuning/discovery/evaluation: {unique_groups}")

    # ===========================================================
    # ILP MINER (tuned via alpha grid search)
    # ===========================================================
    best_params = tune_ilp_miner(sublogs, unique_groups, results_dir)

    def ilp_discover_fn(sublog):
        return pm4py.discover_petri_net_ilp(
            sublog, alpha=best_params['alpha']
        )

    ilp_models = discover_models_and_simplicity(
        sublogs, unique_groups, ilp_discover_fn, results_dir, "ILP Miner"
    )
    cross_evaluate(sublogs, unique_groups, ilp_models, results_dir, "ILP Miner")

    print(f"\nAll tasks completed successfully! Check the output directory: '{results_dir}'")


def save_heatmap(matrix, title, cmap, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap=cmap, fmt=".4f", vmin=0, vmax=1, center=0.5, linewidths=.5,
                cbar_kws={'label': 'Score'})
    plt.title(f"{title}\n(Rows: Evaluated Log | Columns: Discovered Model)", fontsize=13, pad=15, weight='bold')
    plt.xlabel("Model Age Group", fontsize=11, labelpad=10)
    plt.ylabel("Log Age Group", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_simplicity(df, results_dir):
    plt.figure(figsize=(9, 5))
    norm = plt.Normalize(0, 1)
    colors = plt.cm.RdBu(norm(df['arc_degree_simplicity']))
    bars = plt.bar(df.index, df['arc_degree_simplicity'], color=colors, edgecolor='grey', width=0.6)

    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02, f"{height:.4f}", ha='center', va='bottom',
                     fontsize=10, weight='bold')

    plt.title(f"Model Simplicity (Arc-Degree Metric) per Age Group - ILP\n(Red: Low Simplicity | Blue: High Simplicity)",
              fontsize=12, pad=15, weight='bold')
    plt.xlabel("Age Group", fontsize=11, labelpad=10)
    plt.ylabel("Simplicity Score", fontsize=11, labelpad=10)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "simplicity_chart.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    XES_FILE = "Data/all_waves_merged.xes"
    OUTPUT_DIRECTORY = "Results/ilp_age_group_analysis_outputs"

    if os.path.exists(XES_FILE):
        analyze_process_by_age(XES_FILE, results_dir=OUTPUT_DIRECTORY)
    else:
        print(f"Error: Target file '{XES_FILE}' not found.")
