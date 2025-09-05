import ast
import pandas as pd
from paths import runs_dir

pred_csv = runs_dir / "20250829121120532455_vpcls-graphhb-512-8/vpcls-graphhb-512-8_0000505_0.85.csv"
pred_df = pd.read_csv(pred_csv)
data_df = pd.read_csv("/p/vast1/LExperts/protorch/NeuroVectorizer/heterodatas_toss_4_x86_64_ib_cray_O3/tests/best_VF_IF.csv")
rt_df = pd.read_csv("/p/vast1/LExperts/protorch/NeuroVectorizer/neuro-vectorizer/tests/IF_VF_times_toss_4_x86_64_ib_cray.csv")

pred_rt_csv = pred_csv.with_stem(f"{pred_csv.stem}_w_rt")
pred_rt_df = pd.DataFrame(columns=[
    "pt_file", "c_file", "llvm_runtime", "pred_runtime",
    "best_runtime", "best_ratio", "pred_ratio", "pred_VF_IF", "best_VF_IF"
])

total_best_ratio = 0
total_pred_ratio = 0

for index, pred_row in pred_df.iterrows():
    data_rows = data_df[data_df['pt_file'] == pred_row['file_name']].to_dict(orient='records')
    assert len(data_rows) == 1, f"Expected 1 row in data_df for {pred_row['file_name']}, but got {len(data_rows)}"
    data_row = data_rows[0]
    
    rt_rows = rt_df[rt_df['c_file'] == data_row['c_file']]

    llvm_runtimes = rt_rows[(rt_rows['VF'] == -1) & (rt_rows['IF'] == -1)]
    assert len(llvm_runtimes) == 1, f"Expected 1 row with IF == -1 and VF == -1 for {data_row['c_file']}, but got {len(llvm_runtimes)}"
    llvm_runtime = llvm_runtimes.iloc[0]['runtime']

    runtimes = rt_rows[~((rt_rows['VF'] == -1) & (rt_rows['IF'] == -1))]
    pred = ast.literal_eval(pred_row['pred'])
    pred_runtime_row = runtimes[(runtimes['VF'] == pred[0]) & (runtimes['IF'] == pred[1])]
    assert len(pred_runtime_row) == 1, f"Expected 1 row in runtimes for {pred_row['file_name']}, but got {len(pred_runtime_row)}"
    pred_runtime = pred_runtime_row.iloc[0]['runtime']

    best_runtime = runtimes['runtime'].min()
    best_ratio = float(llvm_runtime) / float(best_runtime)
    pred_ratio = float(llvm_runtime) / float(pred_runtime)

    total_best_ratio += best_ratio
    total_pred_ratio += pred_ratio

    new_row = pd.Series({
        "pt_file": pred_row['file_name'],
        "c_file": data_row['c_file'],
        "llvm_runtime": llvm_runtime,
        "pred_runtime": pred_runtime,
        "best_runtime": best_runtime,
        "best_ratio": best_ratio,
        "pred_ratio": pred_ratio,
        "pred_VF_IF": tuple(pred),
        "best_VF_IF": pred_row['target']
    })
    pred_rt_df = pd.concat([pred_rt_df, new_row.to_frame().T], ignore_index=True)

pred_rt_df.to_csv(pred_rt_csv, index=False)

avg_best_ratio = total_best_ratio / len(pred_df)
avg_pred_ratio = total_pred_ratio / len(pred_df)

speedups_text = "\t Speedups | Best: {:.4f} | Pred: {:.4f}\n".format(avg_best_ratio, avg_pred_ratio)
with open(pred_csv.with_suffix(".txt"), "a") as f:
    f.write(speedups_text)

print(f"\nCSV File: {pred_rt_csv}")
print(f"\nText File: {pred_csv.with_suffix('.txt')}")
print(speedups_text)
