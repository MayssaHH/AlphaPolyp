# AlphaPolyp

![logo](https://github.com/LineIntegralx/AlphaPolyp/blob/main/images/front.png)


# Google Drive Link 
The link below contains:
- The full dataset in /extracted_folder/syth-colon 
- The powerpoint presentation 
- The demo 
- The model's weights 

<https://drive.google.com/drive/folders/1NARp2kYtM5-aD8gHWHypCNpK98rDFR7-?usp=sharing>

# Docker: Building and Running

There are two options to run AlphaPolyp: building from the provided Dockerfile and model weights, or pulling pre-built Docker images.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/LineIntegralx/AlphaPolyp.git
    cd AlphaPolyp
    ```

2.  **Download Model Weights:**

    *   Download the model weights from the [Google Drive link ](https://drive.google.com/drive/folders/1NARp2kYtM5-aD8gHWHypCNpK98rDFR7-?usp=sharing).
    *   Place the downloaded model weights into the `model` folder within the cloned repository.
      
## Option 1: Building from Source and Local Model Weights

3.  **Build and Run with Docker Compose:**

    ```bash
    docker compose up --build
    ```

    This command builds the Docker images and starts the containers.

## Option 2: Pulling Pre-built Docker Images

3.  **Docker Login:**

    *   In your terminal, run `docker login` and log in to your Docker account.

4.  **Pull Docker Images:**

    ```bash
    docker pull lineintegral/alphapolyp-model:latest
    docker pull lineintegral/alphapolyp-flask:latest
    ```

5.  **Run with Docker Compose:**

    ```bash
    docker compose up
    ```

    This command starts the containers in the foreground and streams their logs to your terminal.

---
# Usage

1. **Upload an Image**
   - On the main page, click on the Browse Files button or drag and drop an endoscopy image into the upload area. 
   - Supported formats: JPG, PNG, JPEG. 

2. **Analyze the Image**
   - After uploading, click the Analyze Image button. 
   - The system will process the image and run the polyp segmentation and volume estimation model. 

3. **View Results**
   - Once analysis is complete, results will be shown:
   - The uploaded image and a segmented version of it. 
   - The estimated volume in mm^3 along with the estimated x,y and z dimensions

4. **Analyze Another Image**
   - To analyze a new image, click the Analyze Another Image button and repeat the upload process.  

5. **Download Report**
   - After analysis, click the Save Result button. 
   - A PDF report containing the analysis result, highlighted image will be generated to be downloaded on your device.


# Labeling System (Using Blender)

## Overview

This labeling system is designed to automatically measure the **size** and **volume** of polyps inside the colon by analyzing 3D intersections between objects in `.obj` files. Given synthetic 3D models that contain both a colon and a polyp, the script:

1. Imports the 3D models into Blender
2. Calculates the intersection between the polyp and colon using a boolean operation
3. Computes the volume of the intersection
4. Extracts the bounding box dimensions (X, Y, Z)
5. Saves the results for each file into a structured `.csv` file

---

## How to Run the Labeling Script

To use this script, ensure you have [Blender](https://www.blender.org/download/) installed (tested with **Blender 3.x**) and that your `.obj` files are located inside the `data/sample_data/mesh` directory.


You can run the labeling in **one of two ways**:

---

## Option 1: Run Using the `.blend` File

1. Open **Blender**.
2. Click **Open**, then select the file:  
   `label_polyp_data.blend` (located in the `AlphaPolyp/` main folder).
3. Go to the **Scripting** workspace.
4. Click **Run Script** to process the `.obj` files.

---

## Option 2: Run Using the Python Script Directly

1. Open **Blender**.
2. Go to the **Scripting** workspace.
3. Click **Open**, and select the Python script:  
   `labeling/label_polyp_data.py`
4. Click **Run Script** to start processing.

---

## Output

The script will:
- Process all `.obj` files in `data/sample_data/meshes/`
- Save the results to:  `data/annotations/sample_polyp_intersection_results.csv`


⚠️ **Important**: Make sure you open either the `.blend` file or the `.py` file from the **project root folder** (`AlphaPolyp/`) to ensure the paths work correctly.
