import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker



df_saliency_1  = pd.read_csv('./stats/iou_dice_stats_saliency_1.csv')
df_blackened_1 = pd.read_csv('./stats/iou_dice_stats_blackened_1.csv')
df_segmented_1 = pd.read_csv('./stats/iou_dice_stats_segmented_1.csv')


df_saliency_2  = pd.read_csv('./stats/iou_dice_stats_saliency_2.csv')
df_blackened_2 = pd.read_csv('./stats/iou_dice_stats_blackened_2.csv')
df_segmented_2 = pd.read_csv('./stats/iou_dice_stats_segmented_2.csv')



df_saliency_3  = pd.read_csv('./stats/iou_dice_stats_saliency_3.csv')
df_blackened_3 = pd.read_csv('./stats/iou_dice_stats_blackened_3.csv')
df_segmented_3 = pd.read_csv('./stats/iou_dice_stats_segmented_3.csv')




iou_mean_saliency_1   = df_saliency_1.loc[df_saliency_1['metric'] == 'iou', 'mean'].iloc[0]
iou_mean_saliency_2   = df_saliency_2.loc[df_saliency_2['metric'] == 'iou', 'mean'].iloc[0]
iou_mean_saliency_3   = df_saliency_3.loc[df_saliency_3['metric'] == 'iou', 'mean'].iloc[0]




iou_mean_blackened_1  = df_blackened_1.loc[df_blackened_1['metric'] == 'iou', 'mean'].iloc[0]
iou_mean_blackened_2  = df_blackened_2.loc[df_blackened_2['metric'] == 'iou', 'mean'].iloc[0]
iou_mean_blackened_3  = df_blackened_3.loc[df_blackened_3['metric'] == 'iou', 'mean'].iloc[0]



iou_mean_segmented_1  = df_segmented_1.loc[df_segmented_1['metric'] == 'iou', 'mean'].iloc[0]
iou_mean_segmented_2  = df_segmented_2.loc[df_segmented_2['metric'] == 'iou', 'mean'].iloc[0]
iou_mean_segmented_3  = df_segmented_3.loc[df_segmented_3['metric'] == 'iou', 'mean'].iloc[0]



iou_ci_saliency_1     = df_saliency_1.loc[df_saliency_1['metric'] == 'iou', 'ci95'].iloc[0]
iou_ci_saliency_2     = df_saliency_2.loc[df_saliency_2['metric'] == 'iou', 'ci95'].iloc[0]
iou_ci_saliency_3     = df_saliency_3.loc[df_saliency_3['metric'] == 'iou', 'ci95'].iloc[0]



iou_ci_blackened_1    = df_blackened_1.loc[df_blackened_1['metric'] == 'iou', 'ci95'].iloc[0]
iou_ci_blackened_2    = df_blackened_2.loc[df_blackened_2['metric'] == 'iou', 'ci95'].iloc[0]
iou_ci_blackened_3    = df_blackened_3.loc[df_blackened_3['metric'] == 'iou', 'ci95'].iloc[0]



iou_ci_segmented_1    = df_segmented_1.loc[df_segmented_1['metric'] == 'iou', 'ci95'].iloc[0]
iou_ci_segmented_2    = df_segmented_2.loc[df_segmented_2['metric'] == 'iou', 'ci95'].iloc[0]
iou_ci_segmented_3    = df_segmented_3.loc[df_segmented_3['metric'] == 'iou', 'ci95'].iloc[0]






dice_mean_saliency_1  = df_saliency_1.loc[df_saliency_1['metric'] == 'dice', 'mean'].iloc[0]
dice_mean_saliency_2  = df_saliency_2.loc[df_saliency_2['metric'] == 'dice', 'mean'].iloc[0]
dice_mean_saliency_3  = df_saliency_3.loc[df_saliency_3['metric'] == 'dice', 'mean'].iloc[0]


dice_mean_blackened_1 = df_blackened_1.loc[df_blackened_1['metric'] == 'dice', 'mean'].iloc[0]
dice_mean_blackened_2 = df_blackened_2.loc[df_blackened_2['metric'] == 'dice', 'mean'].iloc[0]
dice_mean_blackened_3 = df_blackened_3.loc[df_blackened_3['metric'] == 'dice', 'mean'].iloc[0]


dice_mean_segmented_1 = df_segmented_1.loc[df_segmented_1['metric'] == 'dice', 'mean'].iloc[0]
dice_mean_segmented_2 = df_segmented_2.loc[df_segmented_2['metric'] == 'dice', 'mean'].iloc[0]
dice_mean_segmented_3 = df_segmented_3.loc[df_segmented_3['metric'] == 'dice', 'mean'].iloc[0]



dice_ci_saliency_1    = df_saliency_1.loc[df_saliency_1['metric'] == 'dice', 'ci95'].iloc[0]
dice_ci_saliency_2    = df_saliency_2.loc[df_saliency_2['metric'] == 'dice', 'ci95'].iloc[0]
dice_ci_saliency_3    = df_saliency_3.loc[df_saliency_3['metric'] == 'dice', 'ci95'].iloc[0]



dice_ci_blackened_1   = df_blackened_1.loc[df_blackened_1['metric'] == 'dice', 'ci95'].iloc[0]
dice_ci_blackened_2   = df_blackened_2.loc[df_blackened_2['metric'] == 'dice', 'ci95'].iloc[0]
dice_ci_blackened_3   = df_blackened_3.loc[df_blackened_3['metric'] == 'dice', 'ci95'].iloc[0]


dice_ci_segmented_1   = df_segmented_1.loc[df_segmented_1['metric'] == 'dice', 'ci95'].iloc[0]
dice_ci_segmented_2   = df_segmented_2.loc[df_segmented_2['metric'] == 'dice', 'ci95'].iloc[0]
dice_ci_segmented_3   = df_segmented_3.loc[df_segmented_3['metric'] == 'dice', 'ci95'].iloc[0]




iou_means_1  = [iou_mean_saliency_1, iou_mean_segmented_1, iou_mean_blackened_1]
iou_means_2  = [iou_mean_saliency_2, iou_mean_segmented_2, iou_mean_blackened_2]
iou_means_3  = [iou_mean_saliency_3, iou_mean_segmented_3, iou_mean_blackened_3]


iou_cis_1    = [iou_ci_saliency_1,    iou_ci_segmented_1 , iou_ci_blackened_1]
iou_cis_2    = [iou_ci_saliency_2,    iou_ci_segmented_2 , iou_ci_blackened_2]
iou_cis_3    = [iou_ci_saliency_3,    iou_ci_segmented_3 , iou_ci_blackened_3]


dice_means_1 = [dice_mean_saliency_1,  dice_mean_segmented_1 , dice_mean_blackened_1]
dice_means_2 = [dice_mean_saliency_2,  dice_mean_segmented_2 , dice_mean_blackened_2]
dice_means_3 = [dice_mean_saliency_3,  dice_mean_segmented_3 , dice_mean_blackened_3]


dice_cis_1   = [dice_ci_saliency_1,      dice_ci_segmented_1 , dice_ci_blackened_1]
dice_cis_2   = [dice_ci_saliency_2,      dice_ci_segmented_2 , dice_ci_blackened_2]
dice_cis_3   = [dice_ci_saliency_3,      dice_ci_segmented_3 , dice_ci_blackened_3]













methods    = ['SR',  'HS' , 'HR']






fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 12))

# --- Plot 1: IoU Comparison ---
ax1.errorbar(
    methods,         # x positions for each method
    iou_means_1,       # y values (means)
    yerr=iou_cis_1,    # error bars (confidence intervals)
    fmt='o',         # marker style
    capsize=15,      # size of the cap at each error bar end
    markersize=15,   # marker size
    color='black',   # color of markers and error bars
    elinewidth=3,    # thickness of the error bar lines
    capthick=3       # thickness of the cap lines
)

# ax1.set_title("IoU", fontsize = 30)
ax1.tick_params(axis='x', labelsize=30)  # For x-axis numbers
ax1.tick_params(axis='y', labelsize=20)  # For y-axis numbers
ax1.margins(x=0.3)
ax1.set_ylim(0, 0.8)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.05)) 
ax1.grid(True)

ax1.set_title("Top 1", fontsize=30)



# --- Plot 2: IoU Comparison ---
ax2.errorbar(
    methods,         # x positions for each method
    iou_means_2,      # y values (means)
    yerr=iou_cis_2,   # error bars (confidence intervals)
    fmt='o',         # marker style
    capsize=15,      # size of the cap at each error bar end
    markersize=15,   # marker size
    color='black',   # color of markers and error bars
    elinewidth=3,    # thickness of the error bar lines
    capthick=3       # thickness of the cap lines
)


# ax2.set_title("IoU", fontsize = 30)
ax2.tick_params(axis='x', labelsize=30)  # For x-axis numbers
ax2.tick_params(axis='y', labelsize=0)  # For y-axis numbers
ax2.margins(x=0.3)
ax2.set_ylim(0, 0.8)
ax2.grid(True)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax2.grid(True)


ax2.set_ylabel("")
ax2.set_title("Top 2" , fontsize=30)


# --- Plot 3: IoU Comparison ---

ax3.errorbar(
    methods,         # x positions for each method
    iou_means_3,      # y values (means)
    yerr=iou_cis_3,   # error bars (confidence intervals)
    fmt='o',         # marker style
    capsize=15,      # size of the cap at each error bar end
    markersize=15,   # marker size
    color='black',   # color of markers and error bars
    elinewidth=3,    # thickness of the error bar lines
    capthick=3       # thickness of the cap lines
)


# ax3.set_title("IoU", fontsize = 30)
ax3.tick_params(axis='x', labelsize=30)  # For x-axis numbers
ax3.tick_params(axis='y', labelsize=0)  # For y-axis numbers
ax3.margins(x=0.3)
ax3.set_ylim(0, 0.8)
ax3.grid(True)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax3.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax3.grid(True)


ax3.set_ylabel("")
ax3.set_title("Top 3" , fontsize=30)





# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("./combined_iou.png")
plt.show()












fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 12))

# --- Plot 1: IoU Comparison ---
ax1.errorbar(
    methods,         # x positions for each method
    dice_means_1,       # y values (means)
    yerr=dice_cis_1,    # error bars (confidence intervals)
    fmt='o',         # marker style
    capsize=15,      # size of the cap at each error bar end
    markersize=15,   # marker size
    color='black',   # color of markers and error bars
    elinewidth=3,    # thickness of the error bar lines
    capthick=3       # thickness of the cap lines
)

# ax1.set_title("Dice", fontsize = 30)
ax1.tick_params(axis='x', labelsize=30)  # For x-axis numbers
ax1.tick_params(axis='y', labelsize=20)  # For y-axis numbers
ax1.margins(x=0.3)
ax1.set_ylim(0, 0.8)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.05)) 
ax1.grid(True)

ax1.set_title("Top 1", fontsize = 30)



# --- Plot 2: Dice Comparison ---
ax2.errorbar(
    methods,         # x positions for each method
    dice_means_2,      # y values (means)
    yerr=dice_cis_2,   # error bars (confidence intervals)
    fmt='o',         # marker style
    capsize=15,      # size of the cap at each error bar end
    markersize=15,   # marker size
    color='black',   # color of markers and error bars
    elinewidth=3,    # thickness of the error bar lines
    capthick=3       # thickness of the cap lines
)


# ax2.set_title("Dice", fontsize = 30)
ax2.tick_params(axis='x', labelsize=30)  # For x-axis numbers
ax2.tick_params(axis='y', labelsize=0)  # For y-axis numbers
ax2.margins(x=0.3)
ax2.set_ylim(0, 0.8)
ax2.grid(True)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax2.grid(True)

ax2.yaxis.label.set_visible(False)
ax2.set_title("Top 2" , fontsize=30)


# --- Plot 3: Dice Comparison ---
ax3.errorbar(
    methods,         # x positions for each method
    dice_means_3,      # y values (means)
    yerr=dice_cis_3,   # error bars (confidence intervals)
    fmt='o',         # marker style
    capsize=15,      # size of the cap at each error bar end
    markersize=15,   # marker size
    color='black',   # color of markers and error bars
    elinewidth=3,    # thickness of the error bar lines
    capthick=3       # thickness of the cap lines
)


# ax3.set_title("Dice", fontsize = 30)
ax3.tick_params(axis='x', labelsize=30)  # For x-axis numbers
ax3.tick_params(axis='y', labelsize=0)  # For y-axis numbers
ax3.margins(x=0.3)
ax3.set_ylim(0, 0.8)
ax3.grid(True)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax3.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax3.grid(True)

ax3.yaxis.label.set_visible(False)
ax3.set_title("Top 3" , fontsize=30)





# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("./combined_dice.png")
plt.show()



