import nibabel as nib
import numpy as np
from skimage.util import random_noise
# Import new denoise functions
from skimage.restoration import denoise_bilateral, denoise_wavelet # Keep these
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os # Needed for Deep Learning model path
# Import TensorFlow and Keras components
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from skimage import filters

# --- Image Loading ---
def load_nifti_file(file_path):
    """Loads a NIfTI file and returns the image data."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(f"Loaded image shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None

def get_slice(data, slice_index, axis=2):
    """Extracts a 2D slice from 3D data."""
    if data is None or data.ndim != 3:
        return None
    try:
        if axis == 0: # Sagittal
            slice_2d = data[slice_index, :, :]
        elif axis == 1: # Coronal
            slice_2d = data[:, slice_index, :]
        else: # Axial (default)
            slice_2d = data[:, :, slice_index]
        return np.rot90(slice_2d) # Rotate for standard display
    except IndexError:
        print(f"Slice index {slice_index} out of bounds for axis {axis}")
        return None

def normalize_image(image):
    """Normalizes image data to the range [0, 1]."""
    if image is None:
        return None
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        normalized = (image - min_val) / (max_val - min_val)
        return normalized
    else:
        return np.zeros_like(image) # Handle constant image

# --- Noise Addition ---
def add_gaussian_noise_func(image, sigma=25):
    """Adds Gaussian noise to a normalized image."""
    if image is None: return None
    var = (sigma / 255.0)**2
    noisy_image = random_noise(image, mode='gaussian', var=var, clip=True)
    return noisy_image

def add_salt_pepper_noise(image, amount=0.05):
    """Adds salt & pepper noise to a normalized image."""
    if image is None: return None
    noisy_image = random_noise(image, mode='s&p', amount=amount, salt_vs_pepper=0.5, clip=True)
    return noisy_image

# --- Filtering ---
def apply_gaussian_filter(image, sigma=1.0):
    """Applies Gaussian filtering to denoise an image."""
    if image is None: return None
    try:
        denoised = filters.gaussian(image, sigma=sigma, channel_axis=None)
        return np.clip(denoised, 0, 1)
    except Exception as e:
        print(f"Error applying Gaussian filter: {e}")
        return None

def apply_bilateral_filter(image, sigma_spatial=1, sigma_color=0.1):
    """Applies a bilateral filter for edge-preserving smoothing."""
    if image is None: return None
    denoised_image = denoise_bilateral(image, sigma_spatial=sigma_spatial, 
                                      sigma_color=sigma_color, channel_axis=None)
    return denoised_image

def apply_wavelet_filter(image, wavelet='db4', mode='soft', method='BayesShrink'):
    """Applies wavelet-based denoising."""
    if image is None: return None
    denoised_image = denoise_wavelet(image, wavelet=wavelet, mode=mode, 
                                    method=method, channel_axis=None)
    return denoised_image

def create_unet_model(input_size=(256, 256, 1)):
    """Creates a U-Net model for image denoising."""
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Middle
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([up1, conv2], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([up2, conv1], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def get_model_path():
    """Returns the path to the denoising model."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    return os.path.join(script_dir, 'denoising_model.h5')

def apply_deep_learning_filter(image, use_memory_optimization=True):
    """Applies deep learning-based denoising with memory optimization."""
    if image is None: return None
    
    try:
        model_path = get_model_path()
        model_input_shape = (256, 256)
        
        if not os.path.exists(model_path):
            print(f"Pre-trained model not found at {model_path}. Creating and saving a new model...")
            model = create_unet_model(input_size=(*model_input_shape, 1))
            model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Loading model from {model_path}")
            model = load_model(model_path)
        
        orig_shape = image.shape
        resized = np.zeros(model_input_shape, dtype=image.dtype)
        min_h = min(orig_shape[0], model_input_shape[0])
        min_w = min(orig_shape[1], model_input_shape[1])
        resized[:min_h, :min_w] = image[:min_h, :min_w]
        
        input_img = resized.reshape(1, *model_input_shape, 1)
        denoised = model.predict(input_img, verbose=0)
        denoised = denoised[0, :, :, 0]
        
        if orig_shape != model_input_shape:
            result = np.zeros(orig_shape, dtype=denoised.dtype)
            result[:min_h, :min_w] = denoised[:min_h, :min_w]
        else:
            result = denoised
        
        result = np.clip(result, 0, 1)
        
        if use_memory_optimization:
            tf.keras.backend.clear_session()
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        
        return result
        
    except Exception as e:
        print(f"Error applying deep learning filter: {e}")
        print("Falling back to Bilateral filter.")
        return apply_bilateral_filter(image)

def calculate_metrics(original, processed):
    """Calculates PSNR and SSIM between two images."""
    if original is None or processed is None:
        return None, None
    if original.shape != processed.shape:
        print("Error: Image shapes do not match for metric calculation.")
        return None, None
        
    try:
        original = original.astype(np.float64)
        processed = processed.astype(np.float64)
        data_range = 1.0
        
        psnr_val = psnr(original, processed, data_range=data_range)
        ssim_val = ssim(original, processed, data_range=data_range, channel_axis=None)
        return psnr_val, ssim_val
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None

def convert_to_qimage(image_array):
    """Converts a numpy array (normalized 0-1) to a QImage."""
    if image_array is None: return None
    try:
        from PyQt6.QtGui import QImage, qRgb
        
        image_8bit = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        height, width = image_8bit.shape
        bytes_per_line = width
        
        q_image = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        return q_image.copy()
    except ImportError:
        return None
    except Exception as e:
        print(f"Error converting numpy array to QImage: {e}")
        return None


# --- Medical Diagnostic Support Functions ---

def calculate_diagnostic_score(psnr_val, ssim_val):
    """
    Calculate a diagnostic aid score (1-5) based on image quality metrics.
    
    Args:
        psnr_val: Peak Signal-to-Noise Ratio value
        ssim_val: Structural Similarity Index value
        
    Returns:
        tuple: (score, quality_text) where score is an integer 1-5 and
               quality_text is a string description
    """
    if psnr_val is None or ssim_val is None:
        return 0, "N/A"
        
    # Calculate score based on PSNR and SSIM
    if ssim_val > 0.6 and psnr_val > 25:
        return 5, "Excellent"
    elif ssim_val > 0.5 and psnr_val > 22:
        return 4, "Good"
    elif ssim_val > 0.4 and psnr_val > 18:
        return 3, "Acceptable"
    elif ssim_val > 0.2:
        return 2, "Poor"
    else:
        return 1, "Very Poor"

def generate_system_suggestions(filter_type, noise_level, psnr_val, ssim_val):
    """
    Generate helpful suggestions based on image quality and processing.
    
    Args:
        filter_type: String indicating the filter method used
        noise_level: Float indicating the noise level applied
        psnr_val: Peak Signal-to-Noise Ratio value
        ssim_val: Structural Similarity Index value
        
    Returns:
        list: List of suggestion strings
    """
    suggestions = []
    
    # Check noise level
    if noise_level > 30:
        suggestions.append("High noise level detected. Consider re-acquiring image or applying stronger denoising.")
    
    # Check filter performance
    if psnr_val is not None and ssim_val is not None:
        if psnr_val < 15:
            suggestions.append(f"{filter_type} filter performed poorly. Try a different denoising method.")
        elif psnr_val > 25 and ssim_val > 0.6:
            suggestions.append(f"{filter_type} filtering performed excellently. Recommended for diagnostic use.")
            
        # Filter-specific suggestions
        if filter_type == "Gaussian" and psnr_val < 20:
            suggestions.append("Consider using Bilateral filter for better edge preservation.")
        elif filter_type == "Bilateral" and ssim_val < 0.4:
            suggestions.append("Try adjusting sigma_spatial parameter for better results.")
        elif filter_type == "Wavelet" and psnr_val > 22:
            suggestions.append("Wavelet filtering is performing well for this image type.")
        elif filter_type == "Deep Learning":
            if psnr_val > 25:
                suggestions.append("Deep learning model is effective for this noise pattern.")
            else:
                suggestions.append("Deep learning model may need retraining for this specific noise type.")
    
    return suggestions

def find_best_filter_method(results_dict, original_slice):
    """
    Find the best performing filter method from a dictionary of results.
    
    Args:
        results_dict: Dictionary mapping filter names to processed images
        original_slice: The original image slice for comparison
        
    Returns:
        tuple: (best_method, best_psnr, best_ssim) or (None, None, None) if no results
    """
    if not results_dict or original_slice is None:
        return None, None, None
        
    best_method = None
    best_score = -1
    best_psnr = None
    best_ssim = None
    
    for method, result in results_dict.items():
        psnr_val, ssim_val = calculate_metrics(original_slice, result)
        
        if psnr_val is not None and ssim_val is not None:
            # Combined score (weighted toward SSIM)
            score = (psnr_val / 50.0) * 0.4 + ssim_val * 0.6
            
            if score > best_score:
                best_score = score
                best_method = method
                best_psnr = psnr_val
                best_ssim = ssim_val
                
    return best_method, best_psnr, best_ssim

def extract_patient_metadata(nifti_file):
    """
    Extract patient metadata from a NIfTI file if available.
    
    Args:
        nifti_file: Path to the NIfTI file
        
    Returns:
        dict: Dictionary containing patient metadata or empty dict if none found
    """
    try:
        img = nib.load(nifti_file)
        header = img.header
        
        # Extract available metadata
        metadata = {}
        
        # Try to get standard DICOM fields that might be stored in NIfTI header
        if hasattr(header, 'get_value_label'):
            # This is a simplified example - actual implementation would depend
            # on how metadata is stored in your specific NIfTI files
            for field in ['PatientAge', 'PatientSex', 'PatientID', 'StudyDescription']:
                value = header.get_value_label(field)
                if value:
                    metadata[field] = value
        
        # Extract any available description
        if hasattr(img, 'get_filename'):
            metadata['Filename'] = os.path.basename(img.get_filename())
            
        return metadata
        
    except Exception as e:
        print(f"Error extracting patient metadata: {e}")
        return {}
