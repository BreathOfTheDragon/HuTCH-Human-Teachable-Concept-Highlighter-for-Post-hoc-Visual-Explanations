import pandas as pd
from scipy.stats import sem, t
import os

modes = ["blackened", "segmented"]

for mode in modes:
    # Load the data from your file
    file_path = f"./highlighted_images_overlap/iou_dice_results_{mode}.csv" 
    df = pd.read_csv(file_path)

    # Extract image base names (e.g., "Fur", "PinchWaist", "Stripes")
    df['base_name'] = df['image'].str.extract(r'([A-Za-z]+)')

    # Group by the base name and calculate statistics
    results = []


    for base_name, group in df.groupby('base_name'):
        for metric in ['iou', 'dice']:
            values = group[metric]
            mean_val = values.mean()
            std_val = values.std()
            n = len(values)
            ci95 = t.ppf(0.975, n - 1) * sem(values) if n > 1 else 0  # 95% CI

            results.append({
                'base_name': base_name,
                'metric': metric,
                'mean': round(mean_val,3),
                'std': round(std_val, 3),
                'ci95': round(ci95,3)
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    overlap_dir = "highlighted_images_overlap"
    results_csv_path = os.path.join(overlap_dir, f"iou_dice_stats_{mode}.csv")
    results_df.to_csv(results_csv_path, index=False)


