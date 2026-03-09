import cv2
import numpy as np
import matplotlib.pyplot as plt

def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement for medical images
    
    Parameters:
    medical_image: Input medical image
    modality: Image modality ('X-ray', 'MRI', 'CT', 'Ultrasound')
    
    Returns:
    Enhanced image and enhancement report
    """

    # Pastikan grayscale
    if len(medical_image.shape) == 3:
        medical_image = cv2.cvtColor(medical_image, cv2.COLOR_BGR2GRAY)

    original = medical_image.copy()

    # ======================
    # ENHANCEMENT PIPELINE
    # ======================

    if modality == 'X-ray':
        denoised = cv2.GaussianBlur(medical_image,(5,5),0)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

    elif modality == 'MRI':
        denoised = cv2.medianBlur(medical_image,5)
        enhanced = cv2.normalize(denoised,None,0,255,cv2.NORM_MINMAX)

    elif modality == 'CT':
        window_center = 128
        window_width = 256

        min_val = window_center - window_width//2
        max_val = window_center + window_width//2

        windowed = np.clip(medical_image,min_val,max_val)
        enhanced = cv2.normalize(windowed,None,0,255,cv2.NORM_MINMAX)

    elif modality == 'Ultrasound':
        denoised = cv2.medianBlur(medical_image,5)
        enhanced = cv2.bilateralFilter(denoised,9,75,75)

    else:
        enhanced = medical_image

    # ======================
    # METRICS REPORT
    # ======================

    mean_before = np.mean(original)
    mean_after = np.mean(enhanced)

    std_before = np.std(original)
    std_after = np.std(enhanced)

    hist_before = cv2.calcHist([original],[0],None,[256],[0,256])
    hist_after = cv2.calcHist([enhanced],[0],None,[256],[0,256])

    hist_before = hist_before/hist_before.sum()
    hist_after = hist_after/hist_after.sum()

    entropy_before = -np.sum(hist_before*np.log2(hist_before+1e-7))
    entropy_after = -np.sum(hist_after*np.log2(hist_after+1e-7))

    report = {
        "Modality": modality,
        "Mean Before": float(mean_before),
        "Mean After": float(mean_after),
        "Std Before": float(std_before),
        "Std After": float(std_after),
        "Entropy Before": float(entropy_before),
        "Entropy After": float(entropy_after)
    }

    return enhanced, report


# ==============================
# PROGRAM UTAMA
# ==============================

image = cv2.imread("C:/Users/manta/Pictures/ddca3f92-4b8e-4672-bb6b-f3594ad4e304.jpg",0)

if image is None:
    print("Error: gambar tidak ditemukan")
    exit()

enhanced_image, report = medical_image_enhancement(image,'X-ray')

# tampilkan report
print("\nEnhancement Report")
print("===================")
for k,v in report.items():
    print(k,":",v)

# tampilkan gambar
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image,cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Enhanced Image")
plt.imshow(enhanced_image,cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()