import matplotlib.pyplot as plt
import tifffile

# आपकी बनी हुई फाइल
img = tifffile.imread('final_superres_thermal.tif')

plt.figure(figsize=(12, 10))
plt.imshow(img, cmap='inferno')  # 'hot', 'plasma', या 'viridis' भी इस्तेमाल कर सकते हैं
plt.colorbar(label='Normalized Thermal Intensity')
plt.title('Super-Resolved Thermal Map (100m)')
plt.savefig('thermal_visualization.png', dpi=300)
plt.show()

print(f"Image shape: {img.shape}")
print(f"Min value: {img.min():.4f}, Max value: {img.max():.4f}")