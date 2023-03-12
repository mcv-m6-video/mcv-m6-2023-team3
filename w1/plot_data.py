import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
def plot_ap_miou(study_noise_pos):
     # Separate x and y values into separate lists
        distance = [x for x, _ in study_noise_pos]
        AP_values = [0 for _, y in study_noise_pos] #TODO
        MIOU_values = [y for _, y in study_noise_pos]

        # Create a scatter plot
        plt.plot(distance, AP_values, label='AP')
        plt.plot(distance, MIOU_values, label='MIOU')

        # Add axis labels and title
        plt.xlabel('Max distance')
        plt.title('AP vs MIOU')
        plt.legend(loc='upper right')
        # Show the plot
        plt.show()