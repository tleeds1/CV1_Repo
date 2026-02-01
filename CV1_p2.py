"""
=============================================================================
BÀI TẬP LỚN COMPUTER VISION - PHẦN 2: LỌC ẢNH
Biểu diễn ảnh màu và lọc tín hiệu
=============================================================================

Nội dung:
- Low-pass Filtering: Mean Filter, Gaussian Filter
- High-pass Filtering: Laplacian, Sobel, Unsharp Masking
=============================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Tạo thư mục output
# Tạo thư mục output
OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# =============================================================================
# PHẦN 1: ĐỊNH NGHĨA CÁC KERNEL
# =============================================================================

# --- LOW-PASS FILTERS ---

# Mean Filter (Box Filter) 5x5
# Mỗi pixel = trung bình cộng của 25 pixel lân cận
# Ý nghĩa: Làm mờ đều, giảm nhiễu nhưng làm mất chi tiết biên
KERNEL_MEAN_5x5 = np.ones((5, 5), dtype=np.float32) / 25

# Mean Filter 3x3 (để so sánh)
KERNEL_MEAN_3x3 = np.ones((3, 3), dtype=np.float32) / 9

# --- HIGH-PASS FILTERS ---

# Laplacian Filter
# Phát hiện biên đẳng hướng (mọi hướng)
# Tổng các hệ số = 0 (đặc trưng của high-pass filter)
KERNEL_LAPLACIAN = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

# Laplacian mở rộng (8-connected)
KERNEL_LAPLACIAN_8 = np.array([
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1]
], dtype=np.float32)

# Sobel X - Phát hiện biên theo hướng dọc (vertical edges)
# Gradient theo trục X
KERNEL_SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Sobel Y - Phát hiện biên theo hướng ngang (horizontal edges)
# Gradient theo trục Y
KERNEL_SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)


# =============================================================================
# PHẦN 2: CÁC HÀM TẠO NHIỄU (NOISE GENERATION)
# =============================================================================

def add_salt_pepper_noise(image, amount=0.02):
    """
    Thêm nhiễu muối tiêu (Salt and Pepper Noise) vào ảnh
    
    Nguyên lý:
    - Nhiễu muối tiêu là dạng nhiễu xung (impulse noise)
    - Một số pixel ngẫu nhiên bị thay bằng giá trị cực đại (255 - muối/trắng)
      hoặc cực tiểu (0 - tiêu/đen)
    - Thường xuất hiện do lỗi truyền dữ liệu, sensor bị hỏng
    
    Args:
        image: Ảnh đầu vào
        amount: Tỷ lệ pixel bị nhiễu (0.0-1.0), mặc định 2%
    
    Returns:
        Ảnh đã thêm nhiễu
    """
    noisy = image.copy()
    
    # Tính số pixel bị ảnh hưởng
    num_salt = int(amount * image.size / 2)
    num_pepper = int(amount * image.size / 2)
    
    # Thêm nhiễu muối (trắng)
    coords_salt = [
        np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]
    ]
    if len(image.shape) == 3:
        noisy[coords_salt[0], coords_salt[1], :] = 255
    else:
        noisy[coords_salt[0], coords_salt[1]] = 255
    
    # Thêm nhiễu tiêu (đen)
    coords_pepper = [
        np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]
    ]
    if len(image.shape) == 3:
        noisy[coords_pepper[0], coords_pepper[1], :] = 0
    else:
        noisy[coords_pepper[0], coords_pepper[1]] = 0
    
    return noisy


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Thêm nhiễu Gaussian vào ảnh
    
    Nguyên lý:
    - Nhiễu Gaussian là dạng nhiễu phổ biến nhất trong xử lý ảnh
    - Mỗi pixel được cộng thêm một giá trị ngẫu nhiên theo phân phối Gaussian
    - Thường xuất hiện do nhiễu điện tử trong sensor, ánh sáng yếu
    
    Args:
        image: Ảnh đầu vào
        mean: Giá trị trung bình của nhiễu (thường = 0)
        sigma: Độ lệch chuẩn (sigma càng lớn → nhiễu càng mạnh)
    
    Returns:
        Ảnh đã thêm nhiễu
    """
    # Tạo nhiễu Gaussian
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    
    # Cộng nhiễu vào ảnh và clip về [0, 255]
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


# =============================================================================
# PHẦN 3: CÁC HÀM LỌC ẢNH
# =============================================================================

def apply_median_filter(image, kernel_size=3):
    """
    Áp dụng Median Filter - GIẢI PHÁP TỐI ƯU CHO NHIỄU MUỐI TIÊU
    
    Nguyên lý:
    - Thay thế mỗi pixel bằng GIÁ TRỊ TRUNG VỊ của các pixel lân cận
    - Khác với Mean Filter (trung bình cộng), Median loại bỏ outlier hoàn toàn
    - Ví dụ: [0, 100, 102, 98, 255] → Median = 100 (loại 0 và 255)
                                    → Mean = 111 (bị ảnh hưởng bởi 0 và 255)
    
    Ưu điểm:
    - Xóa sạch nhiễu muối tiêu (outlier bị loại khi sắp xếp)
    - Giữ sắc cạnh tốt hơn Mean/Gaussian Filter
    - Không làm nhòe biên như Mean Filter
    
    Args:
        image: Ảnh đầu vào (grayscale hoặc color)
        kernel_size: Kích thước kernel (3, 5, 7, ...) - phải là số lẻ
    
    Returns:
        Ảnh sau khi lọc
    """
    return cv2.medianBlur(image, kernel_size)


def apply_custom_convolution(image, kernel, name="Custom"):
    """
    Áp dụng phép tích chập (Convolution) với kernel tự định nghĩa
    
    Nguyên lý tích chập:
    - Kernel (ma trận nhỏ) trượt qua từng pixel của ảnh
    - Tại mỗi vị trí: nhân từng phần tử kernel với pixel tương ứng,
      sau đó cộng tất cả lại → giá trị pixel mới
    - Đây là phép toán cơ bản của mọi filter trong xử lý ảnh
    
    Args:
        image: Ảnh đầu vào
        kernel: Ma trận kernel (numpy array)
        name: Tên kernel (để in thông tin)
    
    Returns:
        Ảnh sau khi tích chập
    """
    print(f"   Áp dụng kernel {name}:")
    print(f"   {kernel}")
    
    # cv2.filter2D thực hiện phép tích chập
    # -1 nghĩa là output có cùng depth với input
    result = cv2.filter2D(image, -1, kernel)
    
    return result


def apply_mean_filter(image, kernel_size=5, use_custom=False):
    """
    Áp dụng Mean Filter (Box Filter) - Low-pass filter
    
    Nguyên lý:
    - Thay thế mỗi pixel bằng trung bình cộng của các pixel lân cận
    - Kích thước kernel càng lớn → ảnh càng mờ
    
    Args:
        image: Ảnh đầu vào (grayscale hoặc color)
        kernel_size: Kích thước kernel (3, 5, 7, ...)
        use_custom: Nếu True, dùng kernel tự tạo với cv2.filter2D
    
    Returns:
        Ảnh sau khi lọc
    """
    if use_custom:
        # DEMO: Sử dụng kernel tự định nghĩa
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        return apply_custom_convolution(image, kernel, f"Mean {kernel_size}x{kernel_size}")
    else:
        return cv2.blur(image, (kernel_size, kernel_size))


def apply_gaussian_filter(image, kernel_size=5, sigma=1.5):
    """
    Áp dụng Gaussian Filter - Low-pass filter
    
    Nguyên lý:
    - Tương tự Mean Filter nhưng trọng số giảm dần từ tâm ra ngoài
    - Tuân theo phân phối Gaussian (hình chuông)
    - Sigma (σ) điều chỉnh độ rộng của Gaussian → độ mờ
    
    Ưu điểm so với Mean Filter:
    - Làm mờ tự nhiên hơn
    - Bảo toàn biên tốt hơn
    - Giảm hiệu ứng "blocky"
    
    Args:
        image: Ảnh đầu vào
        kernel_size: Kích thước kernel (phải là số lẻ)
        sigma: Độ lệch chuẩn của Gaussian
    
    Returns:
        Ảnh sau khi lọc
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_laplacian_filter(image, use_8_connected=False):
    """
    Áp dụng Laplacian Filter - High-pass filter
    
    Nguyên lý:
    - Tính đạo hàm bậc 2 của ảnh
    - Phát hiện vùng có sự thay đổi cường độ nhanh (biên)
    - Đẳng hướng: phát hiện biên theo mọi hướng
    
    Đặc điểm:
    - Nhạy với nhiễu (thường cần làm mờ trước)
    - Tổng các hệ số trong kernel = 0
    
    Args:
        image: Ảnh đầu vào (nên là grayscale)
        use_8_connected: Sử dụng kernel 8-connected (mạnh hơn)
    
    Returns:
        Ảnh biên (có thể có giá trị âm)
    """
    if use_8_connected:
        kernel = KERNEL_LAPLACIAN_8
    else:
        kernel = KERNEL_LAPLACIAN
    
    # Chuyển sang grayscale nếu là ảnh màu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Áp dụng filter với kernel tự định nghĩa (DEMO tích chập)
    # cv2.filter2D là hàm tích chập tổng quát
    laplacian = cv2.filter2D(gray, cv2.CV_64F, kernel)
    
    # Chuẩn hóa với cv2.convertScaleAbs (cách chuẩn OpenCV)
    # - Lấy trị tuyệt đối
    # - Clip về 0-255 (không phụ thuộc vào max cục bộ)
    laplacian_8u = cv2.convertScaleAbs(laplacian)
    
    return laplacian_8u


def apply_sobel_filter(image):
    """
    Áp dụng Sobel Filter - High-pass filter
    
    Nguyên lý:
    - Tính đạo hàm bậc 1 theo hướng X và Y
    - Sobel X: phát hiện biên dọc (vertical edges)
    - Sobel Y: phát hiện biên ngang (horizontal edges)
    - Magnitude = sqrt(Gx² + Gy²): tổng hợp biên cả 2 hướng
    
    Args:
        image: Ảnh đầu vào
    
    Returns:
        Tuple (sobel_x, sobel_y, sobel_magnitude)
    """
    # Chuyển sang grayscale nếu là ảnh màu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Cách 1: Dùng hàm OpenCV cv2.Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Cách 2 (DEMO): Dùng kernel tự định nghĩa với cv2.filter2D
    # sobel_x_custom = cv2.filter2D(gray, cv2.CV_64F, KERNEL_SOBEL_X)
    # sobel_y_custom = cv2.filter2D(gray, cv2.CV_64F, KERNEL_SOBEL_Y)
    
    # Tính magnitude (độ lớn gradient)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Chuẩn hóa với cv2.convertScaleAbs (cách chuẩn OpenCV)
    # Ưu điểm: Không bị ảnh hưởng bởi outlier như Min-Max scaling
    sobel_x_8u = cv2.convertScaleAbs(sobel_x)
    sobel_y_8u = cv2.convertScaleAbs(sobel_y)
    magnitude_8u = cv2.convertScaleAbs(magnitude)
    
    return sobel_x_8u, sobel_y_8u, magnitude_8u


def apply_unsharp_masking(image, sigma=1.0, strength=1.5):
    """
    Áp dụng Unsharp Masking - Làm sắc nét ảnh
    
    Nguyên lý:
    - High-frequency = Original - Low-pass (Gaussian blur)
    - Sharpened = Original + strength × High-frequency
    
    Công thức: sharpened = original + k × (original - blurred)
             = (1 + k) × original - k × blurred
    
    Args:
        image: Ảnh đầu vào
        sigma: Sigma cho Gaussian blur
        strength: Hệ số làm sắc (k), thường 0.5-2.0
    
    Returns:
        Ảnh sau khi làm sắc
    """
    # Làm mờ với Gaussian
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Tính high-frequency component
    # sharpened = original + strength * (original - blurred)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    
    return sharpened


# =============================================================================
# PHẦN 3: HÀM VISUALIZATION
# =============================================================================

def show_comparison(original, processed, title_original, title_processed, 
                    save_name=None, cmap_processed=None):
    """Hiển thị so sánh 2 ảnh cạnh nhau"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Ảnh gốc
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title(title_original, fontsize=18)
    axes[0].axis('off')
    
    # Ảnh sau xử lý
    if cmap_processed:
        axes[1].imshow(processed, cmap=cmap_processed)
    elif len(processed.shape) == 3:
        axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(processed, cmap='gray')
    axes[1].set_title(title_processed, fontsize=18)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(os.path.join(IMAGES_DIR, save_name), dpi=150, bbox_inches='tight')
        print(f"  → Saved: {save_name}")
    
    plt.show()


def show_multiple_results(original, results, titles, save_name=None):
    """Hiển thị nhiều kết quả trong 1 figure"""
    n = len(results) + 1
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 10*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    # Ảnh gốc
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Ảnh gốc", fontsize=20)
    axes[0].axis('off')
    
    # Các kết quả
    for i, (result, title) in enumerate(zip(results, titles)):
        ax = axes[i + 1]
        if len(result.shape) == 3:
            ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(result, cmap='gray')
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    
    # Ẩn axes thừa
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_name:
        plt.savefig(os.path.join(IMAGES_DIR, save_name), dpi=150, bbox_inches='tight')
        print(f"  → Saved: {save_name}")
    
    plt.show()


def print_kernel_info():
    """In thông tin về các kernel sử dụng"""
    print("\n" + "="*60)
    print("THÔNG TIN CÁC KERNEL SỬ DỤNG")
    print("="*60)
    
    print("\n--- LOW-PASS FILTERS ---")
    print("\n1. Mean Filter 5x5:")
    print("   Kernel:")
    print("   [[1/25, 1/25, 1/25, 1/25, 1/25],")
    print("    [1/25, 1/25, 1/25, 1/25, 1/25],")
    print("    [1/25, 1/25, 1/25, 1/25, 1/25],")
    print("    [1/25, 1/25, 1/25, 1/25, 1/25],")
    print("    [1/25, 1/25, 1/25, 1/25, 1/25]]")
    print("   Ý nghĩa: Lấy trung bình cộng 25 pixel → làm mờ đều")
    
    print("\n2. Gaussian Filter:")
    print("   Kernel: trọng số theo phân phối Gaussian")
    print("   Trung tâm có trọng số lớn nhất, giảm dần ra ngoài")
    print("   Ý nghĩa: Làm mờ tự nhiên, bảo toàn biên tốt hơn Mean")
    
    print("\n--- HIGH-PASS FILTERS ---")
    print("\n3. Laplacian Filter:")
    print("   Kernel:")
    print("   [[ 0,  1,  0],")
    print("    [ 1, -4,  1],")
    print("    [ 0,  1,  0]]")
    print("   Ý nghĩa: Đạo hàm bậc 2, phát hiện biên đẳng hướng")
    
    print("\n4. Sobel X:")
    print("   Kernel:")
    print("   [[-1,  0,  1],")
    print("    [-2,  0,  2],")
    print("    [-1,  0,  1]]")
    print("   Ý nghĩa: Gradient theo X, phát hiện biên dọc")
    
    print("\n5. Sobel Y:")
    print("   Kernel:")
    print("   [[-1, -2, -1],")
    print("    [ 0,  0,  0],")
    print("    [ 1,  2,  1]]")
    print("   Ý nghĩa: Gradient theo Y, phát hiện biên ngang")
    
    print("\n6. Unsharp Masking:")
    print("   Công thức: sharpened = original + k × (original - blurred)")
    print("   Ý nghĩa: Làm sắc nét bằng cách cộng thêm high-frequency")
    print("="*60 + "\n")


# =============================================================================
# PHẦN 4: XỬ LÝ VÀ PHÂN TÍCH TỪNG ẢNH
# =============================================================================

def process_image(image_path, image_name):
    """
    Xử lý một ảnh với tất cả các loại filter
    
    Args:
        image_path: Đường dẫn đến ảnh
        image_name: Tên để đặt cho file output
    """
    print(f"\n{'='*60}")
    print(f"XỬ LÝ ẢNH: {image_name}")
    print(f"{'='*60}")
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return
    
    print(f"Kích thước ảnh: {image.shape}")
    
    # =========================================================================
    # PHẦN A: DEMO KHỬ NHIỄU (DENOISING)
    # Quy trình: Ảnh gốc → Thêm nhiễu → Lọc → So sánh
    # =========================================================================
    print("\n" + "-"*60)
    print("DEMO KHỬ NHIỄU (DENOISING)")
    print("-"*60)
    
    # --- A1: Nhiễu muối tiêu (Salt & Pepper Noise) ---
    print("\n--- A1: Khử nhiễu Muối Tiêu (Salt & Pepper) ---")
    print("   Nhiễu muối tiêu: pixel ngẫu nhiên thành trắng (255) hoặc đen (0)")
    print("   → GIẢI PHÁP TỐI ƯU: Median Filter (lọc trung vị)")
    
    # Thêm nhiễu muối tiêu
    noisy_sp = add_salt_pepper_noise(image, amount=0.02)
    
    # So sánh các phương pháp khử nhiễu
    denoised_mean = apply_mean_filter(noisy_sp, kernel_size=3)
    denoised_median = apply_median_filter(noisy_sp, kernel_size=3)  # GIẢI PHÁP TỐI ƯU
    denoised_median_5 = apply_median_filter(noisy_sp, kernel_size=5)
    
    # Hiển thị so sánh: Gốc → Nhiễu → Các phương pháp khử nhiễu
    show_multiple_results(
        image,
        [noisy_sp, denoised_mean, denoised_median, denoised_median_5],
        ["+ Nhiễu S&P 2%", "Mean 3x3 (nhòe)", "Median 3x3 ✓", "Median 5x5 ✓"],
        "lowpass_denoise_sp.png" if image_name == "chandung" else f"{image_name}_denoise_sp.png"
    )
    
    # --- A2: Nhiễu Gaussian ---
    print("\n--- A2: Khử nhiễu Gaussian ---")
    print("   Nhiễu Gaussian: mỗi pixel cộng thêm giá trị ngẫu nhiên theo phân phối Gaussian")
    
    # Thêm nhiễu Gaussian
    noisy_gaussian = add_gaussian_noise(image, mean=0, sigma=25)
    
    # Khử nhiễu
    denoised_mean_g = apply_mean_filter(noisy_gaussian, kernel_size=5)
    denoised_gaussian_g = apply_gaussian_filter(noisy_gaussian, kernel_size=5, sigma=1.5)
    
    # Hiển thị so sánh
    show_multiple_results(
        image,
        [noisy_gaussian, denoised_mean_g, denoised_gaussian_g],
        ["+ Nhiễu Gaussian σ=25", "Khử với Mean 5x5", "Khử với Gaussian σ=1.5"],
        "lowpass_denoise_gaussian.png" if image_name == "chandung" else f"{image_name}_denoise_gaussian.png"
    )
    
    # =========================================================================
    # PHẦN B: DEMO TÍCH CHẬP VỚI KERNEL TỰ ĐỊNH NGHĨA
    # Chứng minh hiểu bản chất phép Convolution
    # =========================================================================
    print("\n" + "-"*60)
    print("DEMO TÍCH CHẬP VỚI KERNEL TỰ ĐỊNH NGHĨA (cv2.filter2D)")
    print("-"*60)
    
    # B1: Mean Filter với kernel tự tạo
    print("\n--- B1: Mean Filter với kernel tự định nghĩa ---")
    mean_custom = apply_mean_filter(image, kernel_size=5, use_custom=True)
    
    # B2: Laplacian với kernel tự định nghĩa (đã có trong apply_laplacian_filter)
    print("\n--- B2: Laplacian với kernel tự định nghĩa ---")
    print(f"   Kernel Laplacian 4-connected:")
    print(f"   {KERNEL_LAPLACIAN}")
    laplacian_custom = cv2.filter2D(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 
        cv2.CV_64F, 
        KERNEL_LAPLACIAN
    )
    laplacian_custom = cv2.convertScaleAbs(laplacian_custom)
    
    # B3: Sobel với kernel tự định nghĩa
    print("\n--- B3: Sobel với kernel tự định nghĩa ---")
    print(f"   Kernel Sobel X:")
    print(f"   {KERNEL_SOBEL_X}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x_custom = cv2.filter2D(gray, cv2.CV_64F, KERNEL_SOBEL_X)
    sobel_x_custom = cv2.convertScaleAbs(sobel_x_custom)
    
    # Hiển thị kết quả custom convolution
    show_multiple_results(
        image,
        [mean_custom, laplacian_custom, sobel_x_custom],
        ["Mean 5x5 (custom)", "Laplacian (custom)", "Sobel X (custom)"],
        f"{image_name}_custom_conv.png"
    )
    
    # =========================================================================
    # PHẦN C: LOW-PASS FILTERS (trên ảnh gốc sạch)
    # =========================================================================
    print("\n--- Áp dụng Low-pass Filters trên ảnh gốc ---")
    
    # Mean Filter với các kernel size khác nhau
    mean_3 = apply_mean_filter(image, kernel_size=3)
    mean_5 = apply_mean_filter(image, kernel_size=5)
    mean_7 = apply_mean_filter(image, kernel_size=7)
    
    # Gaussian Filter
    gaussian = apply_gaussian_filter(image, kernel_size=5, sigma=1.5)
    gaussian_strong = apply_gaussian_filter(image, kernel_size=9, sigma=3.0)
    
    # Hiển thị so sánh Low-pass
    show_multiple_results(
        image,
        [mean_3, mean_5, mean_7, gaussian, gaussian_strong],
        ["Mean 3x3", "Mean 5x5", "Mean 7x7", "Gaussian σ=1.5", "Gaussian σ=3.0"],
        "lowpass_kernel_compare.png" if image_name == "chandung" else f"{image_name}_lowpass.png"
    )
    
    # =========================================================================
    # PHẦN D: HIGH-PASS FILTERS
    # =========================================================================
    print("\n--- Áp dụng High-pass Filters ---")
    
    # Laplacian
    laplacian = apply_laplacian_filter(image)
    laplacian_8 = apply_laplacian_filter(image, use_8_connected=True)
    
    # Sobel
    sobel_x, sobel_y, sobel_mag = apply_sobel_filter(image)
    
    # Hiển thị kết quả High-pass
    show_multiple_results(
        image,
        [laplacian, laplacian_8, sobel_x, sobel_y, sobel_mag],
        ["Laplacian 4-conn", "Laplacian 8-conn", "Sobel X", "Sobel Y", "Sobel Magnitude"],
        "highpass_laplacian.png" if image_name == "toanha" else f"{image_name}_highpass.png"
    )
    
    # =========================================================================
    # PHẦN E: UNSHARP MASKING
    # =========================================================================
    print("\n--- Áp dụng Unsharp Masking ---")
    
    sharpened_weak = apply_unsharp_masking(image, sigma=1.0, strength=0.5)
    sharpened_medium = apply_unsharp_masking(image, sigma=1.0, strength=1.0)
    sharpened_strong = apply_unsharp_masking(image, sigma=1.0, strength=2.0)
    
    # Custom save for highpass_sobel.png (using toanha)
    if image_name == "toanha":
        show_multiple_results(
            image,
            [sobel_x, sobel_y, sobel_mag],
            ["Sobel X", "Sobel Y", "Sobel Magnitude"],
            "highpass_sobel.png"
        )
    
    show_multiple_results(
        image,
        [sharpened_weak, sharpened_medium, sharpened_strong],
        ["Unsharp k=0.5", "Unsharp k=1.0", "Unsharp k=2.0"],
        f"{image_name}_unsharp.png"
    )
    
    # Lưu một số ảnh riêng lẻ
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_noisy_sp.jpg"), noisy_sp)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_noisy_gaussian.jpg"), noisy_gaussian)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_gaussian.jpg"), gaussian)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_laplacian.jpg"), laplacian)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_sobel.jpg"), sobel_mag)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_sharpened.jpg"), sharpened_medium)
    
    print(f"\n✓ Hoàn thành xử lý {image_name}")


# =============================================================================
# PHẦN 5: MAIN - CHẠY CHƯƠNG TRÌNH
# =============================================================================

def main():
    """Hàm chính - chạy xử lý tất cả các ảnh"""
    
    print("\n" + "="*60)
    print("BÀI TẬP LỚN COMPUTER VISION - PHẦN 2")
    print("LỌC ẢNH VỚI LOW-PASS VÀ HIGH-PASS FILTER")
    print("="*60)
    
    # In thông tin kernel
    print_kernel_info()
    
    # Danh sách ảnh cần xử lý
    images = [
        ("CV1_ChanDung1.jpg", "chandung"),      # Ảnh chân dung - chi tiết mịn
        ("CV1_PhongCanh.jpg", "phongcanh"),     # Ảnh phong cảnh - nhiều texture
        ("CV1_ToaNha.jpg", "toanha"),           # Ảnh tòa nhà - đường nét sắc
    ]
    
    # Xử lý từng ảnh
    for filename, name in images:
        image_path = filename  # Giả sử ảnh ở cùng thư mục
        if os.path.exists(image_path):
            process_image(image_path, name)
        else:
            print(f"Cảnh báo: Không tìm thấy {filename}")
    
    print("\n" + "="*60)
    print("HOÀN THÀNH! Kết quả được lưu trong thư mục 'output'")
    print("="*60)
    
    # --- PHÂN TÍCH VÀ NHẬN XÉT ---
    print("\n" + "="*60)
    print("PHÂN TÍCH VÀ NHẬN XÉT")
    print("="*60)
    
    print("""
1. KHỬ NHIỄU (DENOISING):
   - Nhiễu Muối Tiêu (Salt & Pepper): pixel ngẫu nhiên thành trắng/đen
     → MEDIAN FILTER LÀ GIẢI PHÁP TỐI ƯU!
     → Mean Filter chỉ làm nhòe hạt nhiễu ra xung quanh (ảnh vẫn bẩn)
     → Median lấy giá trị trung vị → loại bỏ outlier hoàn toàn, giữ sắc cạnh
   - Nhiễu Gaussian: cộng thêm giá trị ngẫu nhiên theo phân phối Gaussian
     → Gaussian Filter hiệu quả vì khớp với bản chất của nhiễu
   - Nhận xét: Chọn filter phù hợp với loại nhiễu là rất quan trọng!

2. TÍCH CHẬP (CONVOLUTION):
   - Kernel trượt qua từng pixel, nhân và cộng → giá trị mới
   - cv2.filter2D cho phép dùng kernel tùy ý
   - Hiểu convolution là hiểu bản chất mọi phép lọc ảnh

3. LOW-PASS FILTER (Làm mờ/Giảm nhiễu):
   - Mean Filter: Làm mờ đều, đơn giản nhưng làm mất biên
   - Gaussian Filter: Làm mờ tự nhiên hơn, bảo toàn biên tốt hơn
   - Median Filter: Lấy trung vị → tối ưu cho nhiễu muối tiêu, giữ sắc cạnh
   - Kernel càng lớn → ảnh càng mờ (trừ Median vẫn giữ cạnh)
   - Phù hợp: Giảm nhiễu, làm mịn da trong ảnh chân dung

4. HIGH-PASS FILTER (Phát hiện biên):
   - Laplacian: Phát hiện biên mọi hướng, nhạy với nhiễu
   - Sobel: Phát hiện biên theo hướng cụ thể (X hoặc Y)
   - Sobel Magnitude: Kết hợp cả 2 hướng
   - Phù hợp: Phát hiện cạnh, đường viền trong ảnh tòa nhà

5. UNSHARP MASKING (Làm sắc nét):
   - Công thức: sharpened = original + k × (original - blurred)
   - Tăng cường chi tiết bằng cách cộng high-frequency component
   - Strength càng lớn → càng sắc nét (nhưng có thể quá mức)
   - Phù hợp: Làm rõ chi tiết trong ảnh phong cảnh

6. SO SÁNH TRÊN CÁC LOẠI ẢNH:
   - Ảnh chân dung: Low-pass giúp làm mịn da, High-pass làm nổi nét mắt/môi
   - Ảnh phong cảnh: Nhiều texture → High-pass phức tạp, Unsharp tăng chi tiết
   - Ảnh tòa nhà: Đường thẳng rõ → Sobel rất hiệu quả phát hiện cạnh

7. HẠN CHẾ VÀ LƯU Ý:
   - Low-pass: Làm mất chi tiết nếu kernel quá lớn
   - High-pass: Nhạy với nhiễu, cần làm mờ trước nếu ảnh có nhiễu
   - Unsharp: Strength quá lớn → ảnh bị artifact, halo effect
""")


if __name__ == "__main__":
    main()
