import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from fpdf import FPDF

# Define the scale: meters per pixel (you will need to determine this based on your images)
meters_per_pixel = 0.05  # Example: 1 pixel = 0.05 meters, modify as needed

# Preprocessing function to resize, grayscale, and normalize images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (512, 512))
    normalized_image = resized_image / 255.0
    return normalized_image, resized_image

# Function to compute edge detection
def edge_detection(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), threshold1=100, threshold2=200)
    return edges

# Function to detect utility ducts via line detection
def detect_utility_ducts(image):
    edges = edge_detection(image)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=5)
    if lines is not None:
        return len(lines)
    return 0

# Function to detect lines and calculate road length via Hough Line Transform
def calculate_road_length(image):
    edges = edge_detection(image)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10)
    total_length_pixels = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_length_pixels += line_length_pixels
    
    # Convert road length from pixels to kilometers using the meters_per_pixel scale
    total_length_km = (total_length_pixels * meters_per_pixel) / 1000  # Convert meters to kilometers
    return total_length_km

# Function to calculate the histogram difference between two images
def histogram_difference(image1, image2):
    # Convert images to uint8 format for histogram calculation
    image1_uint8 = (image1 * 255).astype(np.uint8)
    image2_uint8 = (image2 * 255).astype(np.uint8)

    # Calculate histograms for both images
    hist1 = cv2.calcHist([image1_uint8], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2_uint8], [0], None, [256], [0, 256])

    # Normalize histograms to compare them
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Compute the correlation between the histograms
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    if hist_diff < 0:
        raise ValueError("The images are not at all similar based on histogram comparison.")
    
    return hist_diff

# Function to detect keypoints for macadamization and pedestrian infrastructure changes
def keypoint_detection(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

# Function to segment the road components based on color
def analyze_road_components(image):
    # Convert the image to the HSV color space for easier color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for road components (adjust as needed)
    # Asphalt (usually dark)
    asphalt_lower = np.array([0, 0, 0])
    asphalt_upper = np.array([180, 255, 50])

    # Markings (white or yellow)
    markings_lower = np.array([20, 0, 150])
    markings_upper = np.array([30, 255, 255])  # yellow
    white_lower = np.array([0, 0, 200])  # white
    white_upper = np.array([180, 50, 255])

    # Create masks for each component
    asphalt_mask = cv2.inRange(hsv_image, asphalt_lower, asphalt_upper)
    markings_mask = cv2.inRange(hsv_image, markings_lower, markings_upper) | cv2.inRange(hsv_image, white_lower, white_upper)

    # Calculate the percentage of the image covered by each component
    total_pixels = image.shape[0] * image.shape[1]
    asphalt_percentage = np.sum(asphalt_mask > 0) / total_pixels * 100
    markings_percentage = np.sum(markings_mask > 0) / total_pixels * 100

    # Return the analysis results as a dictionary
    components = {
        'asphalt_percentage': asphalt_percentage,
        'markings_percentage': markings_percentage
    }
    
    return components

# Function to compute similarity score using Structural Similarity Index (SSIM)
def compute_similarity(image1, image2):
    similarity, _ = ssim(image1, image2, full=True, data_range=1.0)
    return similarity

# Function to calculate pixel intensity difference between two images
def pixel_intensity_difference(image1, image2):
    return np.mean(np.abs(image1 - image2))

# Function to calculate road coverage percentage based on edge detection
def road_coverage_change(edge_image1, edge_image2):
    road_pixels1 = np.sum(edge_image1 > 0)
    road_pixels2 = np.sum(edge_image2 > 0)
    if road_pixels1 == 0:
        return 0
    return (road_pixels2 - road_pixels1) / road_pixels1 * 100

# Function to save results to a text file
def save_report_txt(report_data, filename='report.txt'):
    with open(filename, 'w') as f:
        f.write("Construction Progress Report\n")
        f.write("============================\n\n")
        f.write(f"Similarity Score: {report_data['similarity_score']:.2f}%\n")
        f.write(f"Road Coverage Change: {report_data['road_coverage_change_percent']:.2f}%\n")
        f.write(f"Pixel Intensity Difference: {report_data['pixel_intensity_difference']:.2f}\n")
        f.write(f"Histogram Difference: {report_data['histogram_difference']:.2f}\n")
        f.write(f"Number of Keypoints in Reference Image: {report_data['num_keypoints_image1']}\n")
        f.write(f"Number of Keypoints in Test Image: {report_data['num_keypoints_image2']}\n")
        f.write(f"Number of Matches: {report_data['num_matches']}\n")
        f.write(f"Road Pixels in Reference Image (edges): {report_data['road_pixels1']}\n")
        f.write(f"Road Pixels in Test Image (edges): {report_data['road_pixels2']}\n")
        f.write(f"Histogram Correlation (1.0 = perfect match): {report_data['histogram_difference']:.2f}\n")
        f.write(f"Utility Ducts Change: {report_data['utility_ducts_change']}\n")
        f.write(f"Road Length (in kilometers): {report_data['road_length']:.6f} km\n")
        f.write(f"Asphalt Coverage: {report_data['asphalt_percentage']:.2f}%\n")
        f.write(f"Markings Coverage: {report_data['markings_percentage']:.2f}%\n")
        f.write(f"Estimated Construction Progress: {report_data['estimated_construction_progress_percent']:.2f}%\n")

# Function to save results to a PDF file
def save_report_pdf(report_data, filename='report.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.cell(200, 10, txt="Construction Progress Report", ln=True, align='C')
    pdf.ln(10)

    # Add general progress information
    pdf.cell(200, 10, txt=f"Similarity Score: {report_data['similarity_score']:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Road Coverage Change: {report_data['road_coverage_change_percent']:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Pixel Intensity Difference: {report_data['pixel_intensity_difference']:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Histogram Difference: {report_data['histogram_difference']:.2f}", ln=True)
    
    if 'error_message' in report_data:
        pdf.cell(200, 10, txt=f"Error: {report_data['error_message']}", ln=True)
    elif abs(report_data['histogram_difference'] - 1.0) < 0.01:
        pdf.cell(200, 10, txt="The images might be the same based on histogram similarity.", ln=True)
    
    # Add utility ducts change and estimated progress
    pdf.cell(200, 10, txt=f"Utility Ducts Change: {report_data['utility_ducts_change']}", ln=True)
    pdf.cell(200, 10, txt=f"Road Length (in kilometers): {report_data['road_length']:.6f} km", ln=True)
    pdf.cell(200, 10, txt=f"Asphalt Coverage: {report_data['asphalt_percentage']:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Markings Coverage: {report_data['markings_percentage']:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Estimated Construction Progress: {report_data['estimated_construction_progress_percent']:.2f}%", ln=True)

    # Add detailed information
    pdf.cell(200, 10, txt=f"Number of Keypoints in Reference Image: {report_data['num_keypoints_image1']}", ln=True)
    pdf.cell(200, 10, txt=f"Number of Keypoints in Test Image: {report_data['num_keypoints_image2']}", ln=True)
    pdf.cell(200, 10, txt=f"Number of Matches: {report_data['num_matches']}", ln=True)
    pdf.cell(200, 10, txt=f"Road Pixels in Reference Image (edges): {report_data['road_pixels1']}", ln=True)
    pdf.cell(200, 10, txt=f"Road Pixels in Test Image (edges): {report_data['road_pixels2']}", ln=True)

    # Output the PDF to the specified filename
    pdf.output(filename)
    print(f"PDF report saved as {filename}")

# Main function to process images and generate report
def calculate_construction_progress(image_path1, image_path2):
    image1, resized_image1 = preprocess_image(image_path1)
    image2, resized_image2 = preprocess_image(image_path2)

    edges_image1 = edge_detection(image1)
    edges_image2 = edge_detection(image2)

    similarity_score = compute_similarity(image1, image2)
    road_coverage_change_percent = road_coverage_change(edges_image1, edges_image2)
    pixel_diff = pixel_intensity_difference(image1, image2)
    
    try:
        hist_diff = histogram_difference(image1, image2)
        error_message = None
    except ValueError as e:
        hist_diff = None
        error_message = str(e)
    
    kp1, kp2, matches = keypoint_detection(resized_image1, resized_image2)
    utility_ducts_change = detect_utility_ducts(image1) - detect_utility_ducts(image2)

    # Calculate road length in kilometers
    road_length_km = calculate_road_length(image1)

    # Analyze the road components
    road_components = analyze_road_components(cv2.imread(image_path1))

    progress_percent = (1 - similarity_score) * 100 + road_coverage_change_percent

    report_data = {
        "similarity_score": similarity_score * 100,
        "road_coverage_change_percent": road_coverage_change_percent,
        "pixel_intensity_difference": pixel_diff,
        "histogram_difference": hist_diff if hist_diff is not None else 0,
        "utility_ducts_change": utility_ducts_change,
        "road_length": road_length_km,
        "asphalt_percentage": road_components['asphalt_percentage'],
        "markings_percentage": road_components['markings_percentage'],
        "estimated_construction_progress_percent": progress_percent,
        "num_keypoints_image1": len(kp1),
        "num_keypoints_image2": len(kp2),
        "num_matches": len(matches),
        "road_pixels1": np.sum(edges_image1 > 0),
        "road_pixels2": np.sum(edges_image2 > 0),
    }

    if error_message:
        report_data["error_message"] = error_message

    # Print the additional parameters
    print(f"Number of Keypoints in Reference Image: {len(kp1)}")
    print(f"Number of Keypoints in Test Image: {len(kp2)}")
    print(f"Number of Matches: {len(matches)}")
    print(f"Road Pixels in Reference Image (edges): {np.sum(edges_image1 > 0)}")
    print(f"Road Pixels in Test Image (edges): {np.sum(edges_image2 > 0)}")
    print(f"Histogram Correlation (1.0 = perfect match): {hist_diff:.2f}")
    print(f"Utility Ducts Change (lines detected): {utility_ducts_change}")
    print(f"Estimated Road Length (in kilometers): {road_length_km:.6f} km")
    print(f"Asphalt Coverage: {road_components['asphalt_percentage']:.2f}%")
    print(f"Markings Coverage: {road_components['markings_percentage']:.2f}%")

    save_report_txt(report_data)  # Generate text report
    save_report_pdf(report_data)  # Generate PDF report

# Paths to the input images (replace with your own paths)
image_path1 = "WhatsApp Image 2024-09-11 at 14.28.28.jpeg"
image_path2 = "WhatsApp Image 2024-09-11 at 14.28.29.jpeg"

# Calculate the progress and generate the report
calculate_construction_progress(image_path1, image_path2)