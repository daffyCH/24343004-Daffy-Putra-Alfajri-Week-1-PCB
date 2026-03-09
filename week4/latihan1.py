# ============================================
# LATIHAN 1: IMPLEMENTASI MANUAL HISTOGRAM EQUALIZATION
# ============================================

# --------------------------------------------
# Import Library
# --------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------
# Fungsi Manual Histogram Equalization
# --------------------------------------------
def manual_histogram_equalization(image):

    # 1. Hitung histogram
    histogram = np.zeros(256)

    for pixel in image.flatten():
        histogram[pixel] += 1

    # 2. Hitung cumulative histogram (CDF)
    cdf = histogram.cumsum()

    # Normalisasi CDF
    cdf_normalized = cdf / cdf[-1]

    # 3. Hitung transformation function
    transform = np.floor(255 * cdf_normalized).astype(np.uint8)

    # 4. Transformasi piksel
    equalized_image = transform[image]

    return equalized_image, histogram, cdf, transform


# --------------------------------------------
# Main Program
# --------------------------------------------
def main():

    print("=== LATIHAN 1: MANUAL HISTOGRAM EQUALIZATION ===")

    # Load image
    image = cv2.imread("C:/Users/manta/Pictures/miring.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: gambar tidak ditemukan")
        return

    # Proses equalization
    equalized_image, histogram, cdf, transform = manual_histogram_equalization(image)

    # --------------------------------------------
    # Print sebagian nilai hasil perhitungan
    # --------------------------------------------
    print("\n10 Nilai Histogram pertama:")
    print(histogram[:10])

    print("\n10 Nilai CDF pertama:")
    print(cdf[:10])

    print("\n10 Nilai Transformation Function pertama:")
    print(transform[:10])


    # --------------------------------------------
    # Visualisasi
    # --------------------------------------------
    plt.figure(figsize=(14,8))

    # Original Image
    plt.subplot(2,3,1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Equalized Image
    plt.subplot(2,3,2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title("Equalized Image")
    plt.axis('off')

    # Histogram Original
    plt.subplot(2,3,3)
    plt.hist(image.flatten(), bins=256)
    plt.title("Histogram Original")

    # Histogram Equalized
    plt.subplot(2,3,4)
    plt.hist(equalized_image.flatten(), bins=256)
    plt.title("Histogram Equalized")

    # CDF
    plt.subplot(2,3,5)
    plt.plot(cdf)
    plt.title("Cumulative Histogram (CDF)")

    # Transformation Function
    plt.subplot(2,3,6)
    plt.plot(transform)
    plt.title("Transformation Function")

    plt.tight_layout()
    plt.show()


# --------------------------------------------
# Run Program
# --------------------------------------------
if __name__ == "__main__":
    main()