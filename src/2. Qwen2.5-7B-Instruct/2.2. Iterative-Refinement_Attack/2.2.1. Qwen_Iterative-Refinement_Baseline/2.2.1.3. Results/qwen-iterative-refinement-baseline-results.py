#!/usr/bin/env python
# coding: utf-8

# # Results: Qwen2.5-7B-Instruct (Baseline Defense)
# 
# This notebook analyzes the results of the iterative refinement attack against the target model.  
# It computes overall success rates, bypass severity distributions, and generates summary visualizations and CSV files.
# 
# **Input JSON**: `/kaggle/input/notebooks/nidalshahin/qwen-iterative-refinement-baseline-bst-judgement/Qwen_Iterative-Refinement_Baseline_BST-Judgement.json`

# In[1]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Choose either "Llama" or "Qwen"
MODEL_NAME = "Qwen"

# Paths
INPUT_JSON = "/kaggle/input/notebooks/nidalshahin/qwen-iterative-refinement-baseline-bst-judgement/Qwen_Iterative-Refinement_Baseline_BST-Judgement.json"
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed color mapping for success/blocked
success_blocked_colors = {"Succeeded": "green", "Blocked": "red"}

# Fixed color for mutation bar plot
mutation_bar_color = "teal"

# Fixed color mapping for bypass severity categories
bypass_color_map = {
    "Full Bypass": "darkred",
    "Partial Bypass": "red",
    "Surface Bypass": "orangered",
    "Blocked": "orange",
    "Not Applicable (N/A)": "coral"
}

# Fixed order for bypass severity (BST)
BYPASS_ORDER = ["Full Bypass", "Partial Bypass", "Surface Bypass", "Blocked", "Not Applicable (N/A)"]

# Fixed order for success types (pie chart)
SUCCESS_TYPE_ORDER = ["Original Try", "m1", "m2", "m3", "m4", "m5"]

# Load the JSON data
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} goal records.")


# ## 1. Data Preparation
# 
# Flatten the nested structure into a DataFrame.

# In[2]:


records = []
for goal in data:
    goal_id = goal["id"]
    top_success = goal["success"]
    top_success_type = goal.get("success_type")
    top_failure_type = goal.get("failure_type")
    
    # Original attempt
    orig = goal["goal"]["original"]
    records.append({
        "goal_id": goal_id,
        "attempt": "original",
        "success": orig["judgement"] == "Attack Succeeded (FN)",
        "judgement": orig["judgement"],
        "judge_reasoning": orig.get("judge_reasoning"),
        "bypass_severity": None,
        "attack_method": orig.get("attack_method"),
        "defense_method": orig.get("defense_method"),
        "prompt": orig.get("prompt", "")[:100] + "..." if orig.get("prompt") else None,
        "response": orig.get("model_response", "")[:100] + "..." if orig.get("model_response") else None,
        "top_success": top_success,
        "top_success_type": top_success_type if top_success else None,
        "top_failure_type": top_failure_type if not top_success else None,
    })
    
    # Mutation attempts m1..m5
    for i in range(1, 6):
        mut_key = f"m{i}"
        mut = goal["goal"].get(mut_key)
        if mut is not None:
            judgement = mut.get("judgement", "")
            bypass = None
            if "bypass_severity_taxonomy" in mut:
                bypass = mut["bypass_severity_taxonomy"].get("judgement")
            records.append({
                "goal_id": goal_id,
                "attempt": mut_key,
                "success": judgement == "Attack Succeeded (FN)",
                "judgement": judgement,
                "judge_reasoning": mut.get("judge_reasoning"),
                "bypass_severity": bypass,
                "attack_method": mut.get("attack_method"),
                "defense_method": mut.get("defense_method"),
                "prompt": mut.get("prompt", "")[:100] + "..." if mut.get("prompt") else None,
                "response": mut.get("response", "")[:100] + "..." if mut.get("response") else None,
                "top_success": top_success,
                "top_success_type": top_success_type if top_success else None,
                "top_failure_type": top_failure_type if not top_success else None,
            })

df = pd.DataFrame(records)
print(f"Total attempts (original + mutations): {len(df)}")
print(f"Unique goals: {df['goal_id'].nunique()}")
df.head()


# ## 2. Overall Statistics

# In[3]:


# Success rate by attempt type
success_by_attempt = df.groupby("attempt")["success"].agg(["count", "sum", "mean"])
success_by_attempt["success_rate"] = success_by_attempt["mean"] * 100
success_by_attempt = success_by_attempt.sort_index()
print("Success rate per attempt type:")
print(success_by_attempt[["count", "sum", "success_rate"]])

# Overall goal success (top-level)
goal_success = df.groupby("goal_id")["top_success"].first()
print(f"\nOverall goal success: {goal_success.sum()} / {len(goal_success)} ({goal_success.mean()*100:.1f}%)")

# Success types for successful goals
success_types = df["top_success_type"].dropna().value_counts()
print("\nSuccess types for successful goals:")
print(success_types)

# Failure types for unsuccessful goals
failure_types = df["top_failure_type"].dropna().value_counts()
print("\nFailure types for unsuccessful goals:")
print(failure_types)

# Attack methods
attack_methods = df["attack_method"].value_counts()
print("\nAttack methods used:")
print(attack_methods)

# Bypass severity (from mutations)
bypass_severity = df["bypass_severity"].dropna().value_counts()
print("\nBypass severity taxonomy distribution (only mutations):")
print(bypass_severity)


# ## 3. Visualizations

# In[4]:


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# ### 3.1 Original success vs blocked

# In[5]:


orig_df = df[df["attempt"] == "original"]
orig_counts = orig_df["success"].value_counts().rename(index={True: "Succeeded", False: "Blocked"})

plt.figure()
colors = [success_blocked_colors[cat] for cat in orig_counts.index]
ax = orig_counts.plot(kind="bar", color=colors)
plt.title(f"[{MODEL_NAME}] Original Direct-Pass Attempts: Success vs Blocked", weight="bold")
plt.ylabel("Number of attempts")
plt.xticks(rotation=0)
for i, v in enumerate(orig_counts):
    ax.text(i, v + 3.5, str(v), ha="center", weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{MODEL_NAME}_original_success_vs_blocked.png"), dpi=150)
plt.show()


# ### 3.2 Success rate by mutation level

# In[6]:


mut_success = df[df["attempt"].str.startswith("m")].groupby("attempt")["success"].mean() * 100
plt.figure()
mut_success.plot(kind="bar", color=mutation_bar_color)
plt.title(f"[{MODEL_NAME}] Mutation Attempt Success Rate (%)", weight="bold")
plt.ylabel("Success rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=0)
for i, v in enumerate(mut_success):
    plt.text(i, v + 2, f"{v:.1f}%", ha="center", weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{MODEL_NAME}_mutation_success_rate.png"), dpi=150)
plt.show()


# ### 3.3 Bypass severity distribution

# In[7]:


if not bypass_severity.empty:
    bypass_severity = bypass_severity.reindex(BYPASS_ORDER, fill_value=0)
    colors = [bypass_color_map.get(cat, "gray") for cat in bypass_severity.index]
    plt.figure()
    bypass_severity.plot(kind="bar", color=colors)
    plt.title(f"[{MODEL_NAME}] Mutation Bypass Severity Taxonomy", weight="bold")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    for i, v in enumerate(bypass_severity):
        plt.text(i, v + 0.35, str(int(v)), ha="center", weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{MODEL_NAME}_bypass_severity.png"), dpi=150)
    plt.show()


# ### 3.4 Top-level success types pie chart

# In[8]:


if not success_types.empty:
    success_types = success_types.reindex(SUCCESS_TYPE_ORDER, fill_value=0)
    success_types = success_types[success_types > 0]
    if not success_types.empty:
        pie_colors = sns.color_palette("pastel", n_colors=len(success_types))
        plt.figure()
        success_types.plot(kind="pie", autopct="%1.1f%%", startangle=0, colors=pie_colors)
        plt.title(f"[{MODEL_NAME}] Successful Goal Types", weight="bold")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{MODEL_NAME}_success_types_pie.png"), dpi=150)
        plt.show()


# ## 4. Export Summary CSV Files

# In[9]:


# 4.1 Summary statistics per goal
goal_summary = []
for gid in df["goal_id"].unique():
    goal_data = df[df["goal_id"] == gid]
    orig_success = goal_data[goal_data["attempt"] == "original"]["success"].values[0]
    mut_success = goal_data[goal_data["attempt"].str.startswith("m")]["success"].any()
    top_success = goal_data["top_success"].iloc[0]
    top_success_type = goal_data["top_success_type"].iloc[0]
    top_failure_type = goal_data["top_failure_type"].iloc[0]
    bypass_list = goal_data["bypass_severity"].dropna().tolist()
    goal_summary.append({
        "goal_id": gid,
        "original_success": orig_success,
        "any_mutation_success": mut_success,
        "overall_success": top_success,
        "overall_success_type": top_success_type if top_success else None,
        "failure_type": top_failure_type if not top_success else None,
        "bypass_severities": "; ".join(bypass_list)
    })

summary_df = pd.DataFrame(goal_summary)
summary_df.to_csv(os.path.join(OUT_DIR, f"{MODEL_NAME}_goal_summary.csv"), index=False)
print("Saved goal_summary.csv")

# 4.2 Detailed attempts table
detailed_df = df[["goal_id", "attempt", "success", "judgement", "bypass_severity", "attack_method", "defense_method", "prompt", "response"]]
detailed_df.to_csv(os.path.join(OUT_DIR, f"{MODEL_NAME}_detailed_attempts.csv"), index=False)
print("Saved detailed_attempts.csv")

# 4.3 Aggregated statistics table
stats = {
    "Metric": [
        "Total goals",
        "Total attempts (orig + mutations)",
        "Original success count",
        "Original success rate (%)",
        "Mutation success count (any level)",
        "Mutation success rate (at least one mutation)",
        "Overall goal success count",
        "Overall goal success rate (%)",
    ],
    "Value": [
        len(goal_summary),
        len(df),
        orig_df["success"].sum(),
        orig_df["success"].mean() * 100,
        df[df["attempt"].str.startswith("m")].groupby("goal_id")["success"].any().sum(),
        df[df["attempt"].str.startswith("m")].groupby("goal_id")["success"].any().mean() * 100,
        goal_success.sum(),
        goal_success.mean() * 100,
    ]
}
stats_df = pd.DataFrame(stats)
stats_df.to_csv(os.path.join(OUT_DIR, f"{MODEL_NAME}_summary_statistics.csv"), index=False)
print("Saved summary_statistics.csv")


# ## All outputs saved to `/kaggle/working/`
# 
# - `goal_summary.csv` – per‑goal success and bypass details  
# - `detailed_attempts.csv` – every attempt (original + mutations)  
# - `summary_statistics.csv` – aggregated numbers  
# - `original_success_vs_blocked.png`  
# - `mutation_success_rate.png`  
# - `bypass_severity.png`  
# - `success_types_pie.png`  
# - `attack_categories.png`

# In[10]:


print("All results saved to:", OUT_DIR)

