#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Libraries and Data

# In[1]:


import pandas as pd
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.patches import ConnectionPatch

INPUT_CSV = "/kaggle/input/notebooks/nidalshahin/llama-direct-pass-wildguard-gen-eval/Llama-3-8B-Lexi-Uncensored_Direct-Pass_WildGuard-7B_judgement.csv"
aggregated_path = '/kaggle/input/notebooks/nidalshahin/llama-direct-pass-wildguard-gen-eval/Llama-3-8B-Lexi-Uncensored_Direct-Pass_WildGuard-7B_xai-aggregated.json'
per_prompt_path = '/kaggle/input/notebooks/nidalshahin/llama-direct-pass-wildguard-gen-eval/Llama-3-8B-Lexi-Uncensored_Direct-Pass_WildGuard-7B_xai-per-prompt.json'

results_df = pd.read_csv(INPUT_CSV)
with open(aggregated_path, 'r') as f: aggregated_data = json.load(f)
with open(per_prompt_path, 'r') as f: per_prompt_data = json.load(f)
    
# Configuration for titles
model_short = "Llama-3-8B-Lexi-Uncensored"
DEFENSE_METHOD = "WildGuard-7B"
ATTACK_METHOD = "Direct-Pass"

files_prefix = f"{model_short}_{ATTACK_METHOD}_{DEFENSE_METHOD}_"
plots_suffix = f"{model_short} ({ATTACK_METHOD}) ({DEFENSE_METHOD})"


# ## 2. Visualization Functions

# In[2]:


def calculate_and_print_metrics(df):
    total = len(df)
    tp = len(df[df['judgement'] == 'Attack Blocked (TP)'])
    fn = len(df[df['judgement'] == 'Attack Succeeded (FN)'])
    tn = len(df[df['judgement'] == 'Correct Utility (TN)'])
    fp = len(df[df['judgement'] == 'Over-Refusal (FP)'])
    
    harmful_total = tp + fn
    benign_total = tn + fp
    
    asr = (fn / harmful_total) * 100 if harmful_total > 0 else 0
    abr = (tp / harmful_total) * 100 if harmful_total > 0 else 0
    tpr = (tn / benign_total) * 100 if benign_total > 0 else 0
    orr = (fp / benign_total) * 100 if benign_total > 0 else 0
    
    metrics_data = {
        "Metric": ["Attack Success Rate (ASR)", "Attack Block Rate (ABR)", 
                   "Task Performance Retention (TPR)", "Over-Refusal Rate (ORR)"],
        "Value (%)": [asr, abr, tpr, orr]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    output_filename = f"{files_prefix}metrics.csv"
    metrics_df.to_csv(output_filename, index=False)
    print("\n--- Evaluation Metrics ---")
    print(metrics_df.to_string(index=False))
    return metrics_df

def plot_confusion_matrix_and_report(df, model_short, ATTACK_METHOD, DEFENSE_METHOD):
    def outcome_from_judgement(j):
        if j in ['Attack Blocked (TP)', 'Over-Refusal (FP)']:
            return 'Harmful'
        else:
            return 'Benign'
    
    y_true = ['Harmful' if label == 1 else 'Benign' for label in df['label']]
    y_pred = [outcome_from_judgement(j) for j in df['judgement']]
    
    cm = confusion_matrix(y_true, y_pred, labels=['Harmful', 'Benign'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Harmful', 'Benign'], yticklabels=['Harmful', 'Benign'])
    ax1.set_title(f"Confusion Matrix: {plots_suffix}")
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    ax2.axis('off')
    ax2.table(cellText=np.round(report_df.values, 2), rowLabels=report_df.index,
              colLabels=report_df.columns, cellLoc='center', loc='center').scale(1, 2)
    plt.tight_layout()
    plt.show()

def make_bar_of_pie(df, filter_classes, title, filename):
    filtered_df = df[df['judgement'].isin(filter_classes)]
    if filtered_df.empty:
        return

    # 1. Prepare Data
    pie_data = filtered_df['judgement'].value_counts()
    bar_target = filter_classes[1]
    bar_data = filtered_df[filtered_df['judgement'] == bar_target]['category'].value_counts()

    if bar_data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        return

    # 2. Create Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(wspace=0)

    # --- Pie chart ---
    wedges, texts, autotexts = ax1.pie(pie_data.values, labels=pie_data.index,
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title(title)

    # Identify the target slice and its color
    target_idx = list(pie_data.index).index(bar_target)
    target_wedge = wedges[target_idx]
    wedge_color = target_wedge.get_facecolor()

    # --- Bar chart (stacked) ---
    total = sum(bar_data.values)
    bottom = 0
    width = 0.2
    n_cats = len(bar_data)

    for i, (category, count) in enumerate(bar_data.items()):
        # Fix color mismatch: Use the exact slice color with decreasing alpha for shades
        alpha_val = 1.0 - (i * 0.5 / max(1, n_cats - 1))
        p = ax2.bar(0, count, width, bottom=bottom, label=category,
                    color=wedge_color, alpha=alpha_val, edgecolor='white')
        
        ax2.bar_label(p, label_type='center', fmt='%d', color='white', weight='bold')
        bottom += count

    ax2.set_title(f"Distribution of {bar_target}")
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1))
    ax2.axis('off')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0, total)

    # --- 3. Fix Connection Lines (Geometric Mapping) ---
    theta1, theta2 = target_wedge.theta1, target_wedge.theta2
    center, r = target_wedge.center, target_wedge.r

    # Calculate the actual Y-coordinates of the two points on the pie edge
    y1 = r * np.sin(np.pi / 180 * theta1) + center[1]
    y2 = r * np.sin(np.pi / 180 * theta2) + center[1]
    x1 = r * np.cos(np.pi / 180 * theta1) + center[0]
    x2 = r * np.cos(np.pi / 180 * theta2) + center[0]

    # Logic: The point with the higher Y-value on the pie connects to the top of the bar
    if y1 > y2:
        top_pie_coords = (x1, y1)
        bot_pie_coords = (x2, y2)
    else:
        top_pie_coords = (x2, y2)
        bot_pie_coords = (x1, y1)

    # Top line
    con_top = ConnectionPatch(xyA=(-width / 2, total), coordsA=ax2.transData,
                               xyB=top_pie_coords, coordsB=ax1.transData,
                               color=wedge_color, linewidth=2, alpha=0.6)
    # Bottom line
    con_bot = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                               xyB=bot_pie_coords, coordsB=ax1.transData,
                               color=wedge_color, linewidth=2, alpha=0.6)

    ax2.add_artist(con_top)
    ax2.add_artist(con_bot)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# In[3]:


# Harmful Prompt Outcomes: WildGuard-7B vs. LLM Defense
def plot_defense_breakdown(results_df, plots_suffix, files_prefix):
    # Filter harmful prompts (label == 1)
    harmful = results_df[results_df['label'] == 1].copy()
    total_harmful = len(harmful)
    
    # Category calculations
    blocked_by_wildguard = len(harmful[harmful['wildguard_action'] == 'blocked'])
    blocked_by_llm = len(harmful[(harmful['wildguard_action'] == 'allowed') & 
                                 (harmful['judgement'] == 'Attack Blocked (TP)')])
    attack_succeeded = len(harmful[(harmful['wildguard_action'] == 'allowed') & 
                                   (harmful['judgement'] == 'Attack Succeeded (FN)')])
    
    assert blocked_by_wildguard + blocked_by_llm + attack_succeeded == total_harmful, \
           "Mismatch in harmful prompt categorisation"
    
    all_categories = ['Blocked by WildGuard', 'Blocked by LLM (TP)', 'Attack Succeeded (FN)']
    all_counts = [blocked_by_wildguard, blocked_by_llm, attack_succeeded]
    colors = ['royalblue', 'green', 'red']

    nonzero_cats = [(cat, cnt, col) for cat, cnt, col in zip(all_categories, all_counts, colors) if cnt > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ----- Left: Vertical Bar Chart -----
    bars = ax1.bar(all_categories, all_counts, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Number of Harmful Prompts', fontsize=12)
    ax1.set_title(f'Breakdown of Harmful Prompt Outcomes\n{plots_suffix}', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar, count in zip(bars, all_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # ----- Right: Donut Chart with Data in Legend -----
    if not nonzero_cats:
        ax2.text(0.5, 0.5, 'No harmful prompts', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    else:
        cat_names = [c[0] for c in nonzero_cats]
        cat_counts = [c[1] for c in nonzero_cats]
        cat_colors = [c[2] for c in nonzero_cats]
        
        # Create legend labels that include the percentage
        legend_labels = [f'{name}: {count} ({count/total_harmful*100:.1f}%)' 
                         for name, count in zip(cat_names, cat_counts)]
        
        # autopct removed to clear the donut slices
        wedges, _ = ax2.pie(cat_counts, 
                            colors=cat_colors, startangle=90,
                            wedgeprops=dict(edgecolor='white', linewidth=2))
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white', ec='none')
        ax2.add_artist(centre_circle)
        ax2.text(0, 0, f'Total\n{total_harmful}', ha='center', va='center',
                 fontsize=16, fontweight='bold')
        
        # Legend with categories, raw counts, and percentages
        ax2.legend(wedges, legend_labels, title="Outcomes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax2.set_title(f'Proportional View\n{plots_suffix}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{files_prefix}defense_breakdown.png", dpi=300, bbox_inches='tight')
    plt.show()

# Benign Prompt Outcomes: Correct Utility vs. Over‑Refusals
def plot_benign_breakdown(results_df, plots_suffix, files_prefix):
    benign = results_df[results_df['label'] == 0].copy()
    total_benign = len(benign)
    
    correct_utility = len(benign[(benign['wildguard_action'] == 'allowed') & 
                                 (benign['judgement'] == 'Correct Utility (TN)')])
    overrefusal_wildguard = len(benign[benign['wildguard_action'] == 'blocked'])
    overrefusal_llm = len(benign[(benign['wildguard_action'] == 'allowed') & 
                                 (benign['judgement'] == 'Over-Refusal (FP)')])
    
    assert correct_utility + overrefusal_wildguard + overrefusal_llm == total_benign, \
           "Mismatch in benign prompt categorisation"
    
    all_categories = ['Correct Utility (TN)', 'Over-refusal by WildGuard', 'Over-refusal by LLM']
    all_counts = [correct_utility, overrefusal_wildguard, overrefusal_llm]
    colors = ['green', 'orange', 'red']
    
    nonzero_cats = [(cat, cnt, col) for cat, cnt, col in zip(all_categories, all_counts, colors) if cnt > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    bars = ax1.bar(all_categories, all_counts, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Number of Benign Prompts', fontsize=12)
    ax1.set_title(f'Breakdown of Benign Prompt Outcomes\n{plots_suffix}', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar, count in zip(bars, all_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    if not nonzero_cats:
        ax2.text(0.5, 0.5, 'No benign prompts', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    else:
        cat_names = [c[0] for c in nonzero_cats]
        cat_counts = [c[1] for c in nonzero_cats]
        cat_colors = [c[2] for c in nonzero_cats]
        
        legend_labels = [f'{name}: {count} ({count/total_benign*100:.1f}%)' 
                         for name, count in zip(cat_names, cat_counts)]
        
        wedges, _ = ax2.pie(cat_counts, 
                            colors=cat_colors, startangle=90,
                            wedgeprops=dict(edgecolor='white', linewidth=2))
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white', ec='none')
        ax2.add_artist(centre_circle)
        ax2.text(0, 0, f'Total\n{total_benign}', ha='center', va='center',
                 fontsize=16, fontweight='bold')
        
        ax2.legend(wedges, legend_labels, title="Outcomes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax2.set_title(f'Proportional View\n{plots_suffix}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{files_prefix}benign_breakdown.png", dpi=300, bbox_inches='tight')
    plt.show()


# In[4]:


def plot_xai_global_influence(aggregated_data, model_short, DEFENSE_METHOD):
    """
    Uses the aggregated XAI JSON (token -> list of scores) to show global token importance.
    Computes mean attention per token.
    """
    # Compute mean attention for each token, exclude very short tokens (optional)
    token_means = {}
    for token, scores in aggregated_data.items():
        if len(token) > 1:  # ignore single characters (optional)
            token_means[token] = np.mean(scores)
    
    # Sort by mean attention
    sorted_items = sorted(token_means.items(), key=lambda x: x[1], reverse=True)
    top_10 = dict(sorted_items[:10])
    bottom_10 = dict(sorted_items[-10:])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Top 10 - "Hot" Tokens
    ax1.barh(list(top_10.keys()), list(top_10.values()), color='darkred')
    ax1.set_title(f"Critical Tokens: {plots_suffix}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Mean Attention Score")
    ax1.invert_yaxis()  # highest at top
    
    # Bottom 10 - "Cold" Tokens
    ax2.barh(list(bottom_10.keys()), list(bottom_10.values()), color='steelblue')
    ax2.set_title("Least Influential Tokens", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Mean Attention Score")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{files_prefix}global_xai.png", dpi=300)
    plt.show()

def plot_per_prompt_heatmaps(per_prompt_data, results_df, model_short, DEFENSE_METHOD, max_samples=2):
    """
    Visualizes Llama-based prompt heatmaps.
    Robust normalization and ensures max_samples valid matches are found.
    """
    
    def normalize(text):
        # Convert to string, lowercase, remove ALL quotes, and collapse whitespace
        text = str(text).lower().replace('"', '').replace("'", "")
        return " ".join(text.split()).strip()
    
    # Create a mapping from prompt text to its attention_scores dictionary
    prompt_to_scores = {}
    for item in per_prompt_data:
        p = item.get('prompt', '')
        if p:
            prompt_to_scores[normalize(p)] = item['attention_scores']
    
    categories = ['Attack Blocked (TP)', 'Attack Succeeded (FN)', 
                  'Correct Utility (TN)', 'Over-Refusal (FP)']
    
    for judgement in categories:
        # Get all instances of this judgement (do not use .head() here)
        subset = results_df[results_df['judgement'] == judgement]
        
        samples_plotted = 0
        
        for idx, row in subset.iterrows():
            # Stop if we've reached the desired number of samples for this category
            if samples_plotted >= max_samples:
                break
                
            prompt_text = normalize(row['prompt'])
            
            # Check if we actually have XAI data for this specific row
            if prompt_text not in prompt_to_scores:
                # Optional: print(f"Skipping ID {idx}: No XAI data found.")
                continue
            
            # Retrieve attention scores
            all_scores = prompt_to_scores[prompt_text]
            bos_token = '<|begin_of_text|>'
            bos_value = all_scores.get(bos_token, 0)
            
            # Exclude BOS for the visual heatmap
            tokens = [t for t in all_scores.keys() if t != bos_token]
            scores = np.array([all_scores[t] for t in tokens])
            
            if len(scores) == 0:
                continue
            
            # Calculate influence metrics for the footer
            max_non_excluded = scores.max()
            multiplicity = bos_value / (max_non_excluded + 1e-9)
            total_sum = sum(all_scores.values())
            bos_percentage = (bos_value / total_sum) * 100
            
            # Normalization for heatmap (0 to 1 scale)
            min_val = scores.min()
            if max_non_excluded > min_val:
                scores_norm = (scores - min_val) / (max_non_excluded - min_val)
            else:
                scores_norm = scores
            
            # Reshaping for grid visualization
            cols = 8
            rows = int(np.ceil(len(tokens) / cols))
            padded_scores = np.full(rows * cols, np.nan)
            padded_tokens = np.full(rows * cols, "", dtype=object)
            padded_scores[:len(scores)] = scores_norm
            padded_tokens[:len(tokens)] = tokens
            
            score_matrix = padded_scores.reshape((rows, cols))
            token_matrix = padded_tokens.reshape((rows, cols))
            
            # Create Plot
            plt.figure(figsize=(14, (rows * 1.2) + 0.8))
            sns.heatmap(score_matrix, annot=token_matrix, fmt="", cmap="YlOrRd",
                        cbar=True, linewidths=1, linecolor='#222',
                        annot_kws={"size": 9, "fontweight": "bold"}, vmin=0, vmax=1)
            
            plt.title(f"Normalized XAI Heatmap [{judgement}] | ID: {idx}\n({plots_suffix})", 
                      fontsize=14, pad=20)
            
            # Add Llama-specific footer info
            footer_text = f"Note: '{bos_token}' excluded. It accounted for {bos_percentage:.1f}% of total attention.\n"
            footer_text += f"Impact Factor: It was {multiplicity:.2f}x more influential than the top visible token."
            plt.figtext(0.5, 0.02, footer_text, ha="center", fontsize=10, 
                        bbox={"facecolor":"red", "alpha":0.05, "pad":8}, style='italic')
            
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.1, 1, 1])
            
            # Save and Show
            filename = f"{files_prefix}heatmap_sample_{idx}_{judgement.replace(' ', '_')}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.show()
            
            # Increment the counter only after a successful plot
            samples_plotted += 1


# ## 3. Results and Plots

# ### 3.1. Print Metrics

# In[5]:


calculate_and_print_metrics(results_df)


# ### 3.2. Confusion Matrix

# In[6]:


plot_confusion_matrix_and_report(results_df, model_short, ATTACK_METHOD, DEFENSE_METHOD)


# ### 3.3. Security Pie (Harmful prompts)

# In[7]:


make_bar_of_pie(results_df, ['Attack Blocked (TP)', 'Attack Succeeded (FN)'], 
                f"Security Analysis: {plots_suffix}", f"{files_prefix}security_pie.png")


# ### 3.4. Utility Pie (Benign prompts)

# In[8]:


make_bar_of_pie(results_df, ['Correct Utility (TN)', 'Over-Refusal (FP)'], 
                f"Utility Analysis: {plots_suffix}", f"{files_prefix}utility_pie.png")


# ### 3.5. Global XAI Analysis (Token Attention)

# In[9]:


# Global XAI: show tokens with highest and lowest average attention across all prompts
plot_xai_global_influence(aggregated_data, model_short, DEFENSE_METHOD)


# ### 3.6. Per-Prompt XAI Heatmaps (Examples of TP, TN, FN, FP)

# In[10]:


# Show heatmaps for up to 2 examples of each relevant judgement type
plot_per_prompt_heatmaps(per_prompt_data, results_df, model_short, DEFENSE_METHOD, max_samples=2)


# ### 3.7. Harmful Prompt Outcomes: WildGuard-7B vs. LLM Defense

# In[11]:


plot_defense_breakdown(results_df, plots_suffix, files_prefix)


# ### 3.8. Benign Prompt Outcomes: Correct Utility vs. Over‑Refusals

# In[12]:


plot_benign_breakdown(results_df, plots_suffix, files_prefix)

