import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"Image not found or unable to load at path: {image_path}")
    return image

# Display image using matplotlib
def displayImage(image, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Preprocess image to remove obstacles (contours)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for c in contours:
        if cv2.contourArea(c) > 2:
            cv2.drawContours(mask, [c], -1, 0, -1)
    
    result = cv2.bitwise_and(image, image, mask=mask)    
    return result, mask

# Select region of interest to find the coordinates of the bounding box
def selectRegionOfInterest(image):
    roi = cv2.selectROI("Select the area of interest", image)
    if roi == (0, 0, 0, 0):
        raise ValueError("ROI selection canceled or invalid.")
    cv2.destroyAllWindows()
    return roi

# Calculate distance between two points
def calculate_distance(pt1, pt2):
    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return dist

# Draw line view
def line_view(image, mask, roi, spacing,electrode_spacing):
    line_image = image.copy()
    x, y, w, h = roi
    vertical_lines_dist = []
    horizontal_lines_dist = []
    diagonal_lines_dist = []
    displayImage(mask,'Mask')

    #Vertical lines
    for i in range(x, x + w + 1, spacing):
        start_point = None
        for j in range(y, y + h + 1):

            #if area of interest, mark that point as start_point
            if mask[j, i] != 0:
                if start_point is None:
                    start_point = j
            
            #if black area detected, stop growing the line and draw the line
            else:
                if start_point is not None:
                    cv2.line(line_image, (i, start_point), (i, j - 1), (0, 255, 0), 1)
                    dist = calculate_distance((i, start_point), (i, j - 1))
                    electrodes = round(dist / electrode_spacing / 2) * 2  
                    for k in range(0,electrodes,int(electrode_spacing)):
                        cv2.circle(line_image,(i, int(start_point + k * electrode_spacing)),2,(0,0,255),-1)
                    vertical_lines_dist.append(electrodes)
                    start_point = None
        
        #edge case
        if start_point is not None:
            end_point = y + h
            cv2.line(line_image, (i, start_point), (i, end_point), (0, 255, 0), 1)
            dist = calculate_distance((i, start_point), (i, end_point))
            electrodes = round(dist / electrode_spacing / 2) * 2 
            for k in range(0, electrodes, int(electrode_spacing)):
                cv2.circle(line_image, (i, int(start_point + k * electrode_spacing)), 2, (0,0,255), -1)
            vertical_lines_dist.append(electrodes)

    #horizontal lines
    for j in range(y, y + h + 1, spacing):
        start_point = None
        for i in range(x, x + w + 1):
            
            #if area of interest, mark that point as start_point
            if mask[j, i] != 0:
                if start_point is None:
                    start_point = i
            
            #if black area detected, stop growing the line and draw the line
            else:
                if start_point is not None:
                    cv2.line(line_image, (start_point, j), (i - 1, j), (0, 255, 0), 1)
                    dist = calculate_distance((start_point, j), (i - 1, j))
                    electrodes = round(dist / electrode_spacing / 2) * 2  
                    for k in range(0,electrodes,int(electrode_spacing)):
                        cv2.circle(line_image, (int(start_point + k * electrode_spacing), j), 2, (0, 0, 255), -1)
                    horizontal_lines_dist.append(electrodes)
                    start_point = None
        #edge case
        if start_point is not None:
            cv2.line(line_image, (start_point, j), (x + w, j), (0, 255, 0), 1)
            dist = calculate_distance((start_point, j), (x + w, j))
            electrodes = round(dist / electrode_spacing / 2) * 2  
            for k in range(0,electrodes,int(electrode_spacing)):
                cv2.circle(line_image, (int(start_point + k * electrode_spacing), j), 2, (0, 0, 255), -1)

            horizontal_lines_dist.append(electrodes)
    
    # Diagonal lines (top-left to bottom-right)
    for k in range(-(h // spacing), (w // spacing) + 1):
        start_point = None
        for i in range(max(x, x - k * spacing), min(x + w, x + w - k * spacing) + 1, spacing):
            j = y + (i - x) + k * spacing
            if y <= j <= y + h:
                if mask[j, i] != 0:
                    if start_point is None:
                        start_point = (i, j)
                else:
                    if start_point is not None:
                        end_point = (i - spacing, j - spacing)
                        cv2.line(line_image, start_point, end_point, (0, 255, 0), 1)
                        dist = calculate_distance(start_point, end_point)
                        electrodes = round(dist / electrode_spacing / 2) * 2  
                        for m in range(0, electrodes,int(electrode_spacing)):
                            dx = int((end_point[0] - start_point[0]) * m / electrodes + start_point[0])
                            dy = int((end_point[1] - start_point[1]) * m / electrodes + start_point[1])
                            cv2.circle(line_image, (dx, dy), 2, (0, 0, 255), -1)
                        diagonal_lines_dist.append(electrodes)
                        start_point = None
        if start_point is not None:
            end_point = (min(x + w, i), min(y + h, j))
            cv2.line(line_image, start_point, end_point, (0, 255, 0), 1)
            dist = calculate_distance(start_point, end_point)
            electrodes = round(dist / electrode_spacing / 2) * 2  
            for m in range(0, electrodes,int(electrode_spacing)):
                dx = int((end_point[0] - start_point[0]) * m / electrodes + start_point[0])
                dy = int((end_point[1] - start_point[1]) * m / electrodes + start_point[1])
                cv2.circle(line_image, (dx, dy), 2, (0, 0, 255), -1)
            diagonal_lines_dist.append(electrodes)

    # Diagonal lines (top-right to bottom-left)
    for k in range(-(h // spacing), (w // spacing) + 1):
        start_point = None
        for i in range(max(x, x + k * spacing), min(x + w, x + w + k * spacing) + 1, spacing):
            j = y + h - (i - x) + k * spacing
            if y <= j <= y + h:
                if mask[j, i] != 0:
                    if start_point is None:
                        start_point = (i, j)
                else:
                    if start_point is not None:
                        end_point = (i - spacing, j + spacing)
                        cv2.line(line_image, start_point, end_point, (0, 255, 0), 1)
                        dist = calculate_distance(start_point, end_point)
                        electrodes = round(dist / electrode_spacing / 2) * 2  
                        for m in range(0, electrodes,int(electrode_spacing)):
                            dx = int((end_point[0] - start_point[0]) * m / electrodes + start_point[0])
                            dy = int((end_point[1] - start_point[1]) * m / electrodes + start_point[1])
                            cv2.circle(line_image, (dx, dy), 2, (0, 0, 255), -1)
                        diagonal_lines_dist.append(electrodes)
                        start_point = None
        if start_point is not None:
            end_point = (min(x + w, i), max(y, j))
            cv2.line(line_image, start_point, end_point, (0, 255, 0), 1)
            dist = calculate_distance(start_point, end_point)
            electrodes = round(dist / electrode_spacing / 2) * 2  
            for m in range(0, electrodes,int(electrode_spacing)):
                dx = int((end_point[0] - start_point[0]) * m / electrodes + start_point[0])
                dy = int((end_point[1] - start_point[1]) * m / electrodes + start_point[1])
                cv2.circle(line_image, (dx, dy), 2, (0, 0, 255), -1)
            diagonal_lines_dist.append(electrodes)

    displayImage(line_image, 'Line image')
    return vertical_lines_dist, horizontal_lines_dist, diagonal_lines_dist

# Main function
if __name__ == "__main__":
    image_path = r'C:\Users\swath\OneDrive\Desktop\Swathika\MeisterGen\PatternRecognition\static\acre3.jpeg'
    #image_path = r'C:\Users\swath\OneDrive\Desktop\Swathika\MeisterGen\PatternRecognition\static\sample.png'
    file_path = r'C:\Users\swath\OneDrive\Desktop\Swathika\MeisterGen\PatternRecognition\static\electrode_file.txt'
    image = load_image(image_path)
    displayImage(image, 'Original Image')
    
    print(f"Image size:{image.shape}")
    #image = cv2.resize(image, (1189, 1600))

    processed_image, obstacle_mask = preprocess_image(image)
    displayImage(processed_image, 'Obstacles Removed')

    roi = selectRegionOfInterest(processed_image)
    print(f"Coordinates of bounding box: {roi}")

    electrode_spacing = float(input("Enter the electrode spacing value: "))
    vertical_electrodes, horizontal_electrodes,diagonal_electrodes = line_view(processed_image, obstacle_mask, roi, 100 ,electrode_spacing)
    with open(file_path,'a') as file:
        for key,val in enumerate(vertical_electrodes):
            file.write(f"Vertical Line {key}: {val}\n")
        file.write("\n\n")
        for key,val in enumerate(horizontal_electrodes):
            file.write(f"Horizontal Line {key}: {val}\n")
        file.write("\n\n")
        for key,val in enumerate(diagonal_electrodes):
            file.write(f"Diagonal Line {key}: {val}\n")
    file.close()
        



        