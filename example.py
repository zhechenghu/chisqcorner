from corner import corner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chisqcorner import plot_corner_chi2

# --- Configuration ---
data_file_path = 'data/samples.csv'
output_plot_path = 'figures/corner_chi2.png'

parameter_column_names = ['m3_msun', 'log_p2_days', 'omega_1_deg', 'i_tot_deg']
chi2_column_name = 'chi2'

plot_param_labels = [r'$m_3$ [M$_\odot$]', r'$\log P_2$ [days]', r'$\omega_1$ [deg]', r'$i_{tot}$ [deg]']

num_plot_bins = 30
plot_colormap = plt.cm.coolwarm

# --- Load data ---
samples_raw = pd.read_csv(data_file_path)
samples_for_plot = samples_raw[parameter_column_names + [chi2_column_name]]
samples_for_plot.replace([np.inf, -np.inf], np.nan, inplace=True)
samples_for_plot.dropna(inplace=True)

# --- Plot ---
figure_chi2 = plot_corner_chi2(
    samples_df=samples_for_plot,
    param_labels=plot_param_labels,
    colormap=plot_colormap,
    nbins=num_plot_bins,
    chi2_col=chi2_column_name
)
figure_chi2.savefig(output_plot_path, dpi=300, bbox_inches='tight')
plt.close()

figure_density = corner(samples_for_plot, labels=plot_param_labels + [chi2_column_name])
figure_density.savefig('figures/corner_density.png', dpi=300, bbox_inches='tight')
plt.close()
