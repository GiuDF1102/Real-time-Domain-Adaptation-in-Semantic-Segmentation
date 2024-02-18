import matplotlib.pyplot as plt

mIoU_values = [0.152, 0.194, 0.183, 0.187, 0.186, 0.214, 0.219, 0.211, 0.203, 0.243, 0.241, 0.237, 0.220, 0.233, 0.246, 0.260, 0.242, 0.242, 0.265, 0.263, 0.259, 0.260, 0.247, 0.260, 0.259, 0.248, 0.259, 0.253, 0.258, 0.264, 0.263, 0.259, 0.267, 0.264, 0.256, 0.266, 0.256, 0.261, 0.264, 0.265, 0.270, 0.262, 0.250, 0.263, 0.264, 0.268, 0.258, 0.264, 0.263, 0.263]

# Number of epochs
epochs = range(1, len(mIoU_values) + 1)

# Create the plot
plt.figure(figsize=(9, 6))
plt.plot(epochs, mIoU_values, linestyle='-')
plt.title('mIoU vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.grid(True)
plt.grid(which='major', color='#666666', linestyle='-')
plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.ylim(0.15, 0.28)
plt.xlim(1, 50)
plt.tight_layout()

# Show the plot
plt.savefig("mIoU_ADV.png")