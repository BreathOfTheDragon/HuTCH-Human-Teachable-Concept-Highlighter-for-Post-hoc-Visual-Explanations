from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import pandas as pd


df_saliency = pd.read_csv('./highlighted_saliency_overlap/iou_dice_results_saliency.csv')
df_blackened = pd.read_csv('./highlighted_images_overlap/iou_dice_results_blackened.csv')
df_segmented = pd.read_csv('./highlighted_images_overlap/iou_dice_results_segmented.csv')


# print(df_saliency['iou'])
# print("\n")

# IoU Saliency vs Segmented:
t_stat_iou, p_value_iou = ttest_ind(df_saliency['iou'], df_segmented['iou'], equal_var=False)
print("t-test IoU Saliency vs Segmented: t =", t_stat_iou, "p =", p_value_iou)

print("\n***************************************\n")

# IoU Saliency vs Blackened:
t_stat_iou_blackened, p_value_iou_blackened = ttest_ind(df_saliency['iou'], df_blackened['iou'], equal_var=False)
print("t-test IoU Saliency vs Blackened: t =", t_stat_iou_blackened, "p =", p_value_iou_blackened)

print("\n***************************************\n")


# Dice Saliency vs Segmented:
t_stat_dice, p_value_dice = ttest_ind(df_saliency['dice'], df_segmented['dice'], equal_var=False)
print("t-test Dice Saliency vs Segmented: t =", t_stat_dice, "p =", p_value_dice)

print("\n***************************************\n")


# Dice Saliency vs Blackened:
t_stat_dice_blackened, p_value_dice_blackened = ttest_ind(df_saliency['dice'], df_blackened['dice'], equal_var=False)
print("t-test Dice Saliency vs Blackened: t =", t_stat_dice_blackened, "p =", p_value_dice_blackened)

print("\n***************************************\n")



anova_one_way_iou = f_oneway(df_saliency['iou'], df_segmented['iou'], df_blackened['iou'])

anova_one_way_dice = f_oneway(df_saliency['dice'], df_segmented['dice'], df_blackened['dice'])


print(anova_one_way_iou)

print(anova_one_way_dice)


