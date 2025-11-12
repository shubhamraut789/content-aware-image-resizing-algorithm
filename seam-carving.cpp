#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

struct Pixel {
    uchar r, g, b;
};

void readImage(const string& imagePath, Pixel**& image, int& height, int& width) {
    Mat3b mat = imread(imagePath);

    if (mat.empty()) {
        cout << "Error: Could not open or find the image!" << '\n';
        return;
    }

    height = mat.rows;
    width = mat.cols;
    
    image = new Pixel*[height];
    for (int i = 0; i < height; ++i) {
        image[i] = new Pixel[width];
        for (int j = 0; j < width; ++j) {
            image[i][j].r = mat.at<Vec3b>(i, j)[2];
            image[i][j].g = mat.at<Vec3b>(i, j)[1];
            image[i][j].b = mat.at<Vec3b>(i, j)[0];
        }
    }
}

void calculateEnergy(Pixel** image, int height, int width, int** energy) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int dx_r = 0, dx_g = 0, dx_b = 0;
            int dy_r = 0, dy_g = 0, dy_b = 0;
            
            // X gradient (horizontal)
            int xLeft = (j > 0) ? j - 1 : j;
            int xRight = (j < width - 1) ? j + 1 : j;
            dx_r = abs(image[i][xRight].r - image[i][xLeft].r);
            dx_g = abs(image[i][xRight].g - image[i][xLeft].g);
            dx_b = abs(image[i][xRight].b - image[i][xLeft].b);
            
            // Y gradient (vertical)
            int yTop = (i > 0) ? i - 1 : i;
            int yBottom = (i < height - 1) ? i + 1 : i;
            dy_r = abs(image[yBottom][j].r - image[yTop][j].r);
            dy_g = abs(image[yBottom][j].g - image[yTop][j].g);
            dy_b = abs(image[yBottom][j].b - image[yTop][j].b);

            // Combined energy
            int dx = dx_r + dx_g + dx_b;
            int dy = dy_r + dy_g + dy_b;
            energy[i][j] = dx + dy;
        }
    }
}

void findSeam(int** energy, int height, int width, int* seam, bool isVertical) {
    int** cumulativeEnergy = new int*[height];
    for (int i = 0; i < height; ++i) {
        cumulativeEnergy[i] = new int[width];
    }

    if (isVertical) {
        for (int j = 0; j < width; ++j) {
            cumulativeEnergy[0][j] = energy[0][j];
        }
        for (int i = 1; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int minEnergy = cumulativeEnergy[i - 1][j];
                if (j > 0) minEnergy = min(minEnergy, cumulativeEnergy[i - 1][j - 1]);
                if (j < width - 1) minEnergy = min(minEnergy, cumulativeEnergy[i - 1][j + 1]);
                cumulativeEnergy[i][j] = energy[i][j] + minEnergy;
            }
        }
        
        // Backtrack to find seam
        int minIndex = 0;
        for (int j = 1; j < width; ++j) {
            if (cumulativeEnergy[height - 1][j] < cumulativeEnergy[height - 1][minIndex]) {
                minIndex = j;
            }
        }
        seam[height - 1] = minIndex;
        for (int i = height - 2; i >= 0; --i) {
            int previous = seam[i + 1];
            int minIndex = previous;
            if (previous > 0 && cumulativeEnergy[i][previous - 1] < cumulativeEnergy[i][minIndex]) {
                minIndex = previous - 1;
            }
            if (previous < width - 1 && cumulativeEnergy[i][previous + 1] < cumulativeEnergy[i][minIndex]) {
                minIndex = previous + 1;
            }
            seam[i] = minIndex;
        }
    } else {  // Horizontal seam
        for (int i = 0; i < height; ++i) {
            cumulativeEnergy[i][0] = energy[i][0];
        }
        for (int j = 1; j < width; ++j) {
            for (int i = 0; i < height; ++i) {
                int minEnergy = cumulativeEnergy[i][j - 1];
                if (i > 0) minEnergy = min(minEnergy, cumulativeEnergy[i - 1][j - 1]);
                if (i < height - 1) minEnergy = min(minEnergy, cumulativeEnergy[i + 1][j - 1]);
                cumulativeEnergy[i][j] = energy[i][j] + minEnergy;
            }
        }
        
        // Backtrack to find seam
        int minIndex = 0;
        for (int i = 1; i < height; ++i) {
            if (cumulativeEnergy[i][width - 1] < cumulativeEnergy[minIndex][width - 1]) {
                minIndex = i;
            }
        }
        seam[0] = minIndex;
        for (int j = 1; j < width; ++j) {
            int previous = seam[j - 1];
            int minIndex = previous;
            if (previous > 0 && cumulativeEnergy[previous - 1][j] < cumulativeEnergy[minIndex][j]) {
                minIndex = previous - 1;
            }
            if (previous < height - 1 && cumulativeEnergy[previous + 1][j] < cumulativeEnergy[minIndex][j]) {
                minIndex = previous + 1;
            }
            seam[j] = minIndex;
        }
    }

    // Clean up
    for (int i = 0; i < height; ++i) {
        delete[] cumulativeEnergy[i];
    }
    delete[] cumulativeEnergy;
}

// Visualize the seam before removing it
void visualizeSeam(Pixel** image, int height, int width, int* seam, bool isVertical, const string& windowName) {
    Mat3b displayMat(height, width);
    
    // Copy image to Mat
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            displayMat.at<Vec3b>(i, j)[0] = image[i][j].b;
            displayMat.at<Vec3b>(i, j)[1] = image[i][j].g;
            displayMat.at<Vec3b>(i, j)[2] = image[i][j].r;
        }
    }
    
    // Highlight the seam in red
    if (isVertical) {
        for (int i = 0; i < height; ++i) {
            int j = seam[i];
            if (j >= 0 && j < width) {
                displayMat.at<Vec3b>(i, j)[0] = 0;   // B
                displayMat.at<Vec3b>(i, j)[1] = 0;   // G
                displayMat.at<Vec3b>(i, j)[2] = 255; // R
            }
        }
    } else {
        for (int j = 0; j < width; ++j) {
            int i = seam[j];
            if (i >= 0 && i < height) {
                displayMat.at<Vec3b>(i, j)[0] = 0;   // B
                displayMat.at<Vec3b>(i, j)[1] = 0;   // G
                displayMat.at<Vec3b>(i, j)[2] = 255; // R
            }
        }
    }
    
    imshow(windowName, displayMat);
    waitKey(1); // Short delay for visualization
}

// Visualize energy map
void visualizeEnergy(int** energy, int height, int width, const string& windowName) {
    // Find max energy for normalization
    int maxEnergy = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            maxEnergy = max(maxEnergy, energy[i][j]);
        }
    }
    
    Mat energyMat(height, width, CV_8UC1);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            energyMat.at<uchar>(i, j) = (uchar)((energy[i][j] * 255) / maxEnergy);
        }
    }
    
    Mat colorMap;
    applyColorMap(energyMat, colorMap, COLORMAP_JET);
    imshow(windowName, colorMap);
}

void removeSeam(Pixel**& image, int& height, int& width, int* seam, bool isVertical) {
    if (isVertical) {
        for (int i = 0; i < height; ++i) {
            for (int j = seam[i]; j < width - 1; ++j) {
                image[i][j] = image[i][j + 1];
            }
        }
        --width;
    } else {
        for (int j = 0; j < width; ++j) {
            for (int i = seam[j]; i < height - 1; ++i) {
                image[i][j] = image[i + 1][j];
            }
        }
        --height;
    }
}

void resizeImage(Pixel**& image, int& height, int& width, int newHeight, int newWidth) {
    // Validation
    if (newWidth > width || newHeight > height) {
        cout << "Error: Can only shrink image, not expand!" << '\n';
        return;
    }
    
    int originalHeight = height;
    int originalWidth = width;
    int totalSeams = (width - newWidth) + (height - newHeight);
    int seamsRemoved = 0;
    
    cout << "Starting seam carving..." << '\n';
    cout << "Original size: " << originalWidth << "x" << originalHeight << '\n';
    cout << "Target size: " << newWidth << "x" << newHeight << '\n';
    cout << "Total seams to remove: " << totalSeams << '\n' << '\n';
    
    // Create windows
    namedWindow("Current Image", WINDOW_AUTOSIZE);
    namedWindow("Energy Map", WINDOW_AUTOSIZE);
    namedWindow("Seam Highlight", WINDOW_AUTOSIZE);
    
    while (height > newHeight || width > newWidth) {
        bool isVertical = (width > newWidth);
        int seamLength = isVertical ? height : width;
        int* seam = new int[seamLength];
        
        int** energy = new int*[height];
        for (int i = 0; i < height; ++i) {
            energy[i] = new int[width];
        }

        calculateEnergy(image, height, width, energy);
        
        // Visualize energy map
        visualizeEnergy(energy, height, width, "Energy Map");
        
        findSeam(energy, height, width, seam, isVertical);
        
        // Visualize the seam before removing
        visualizeSeam(image, height, width, seam, isVertical, "Seam Highlight");
        
        removeSeam(image, height, width, seam, isVertical);
        
        // Show current state after removal
        Mat3b currentMat(height, width);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                currentMat.at<Vec3b>(i, j)[0] = image[i][j].b;
                currentMat.at<Vec3b>(i, j)[1] = image[i][j].g;
                currentMat.at<Vec3b>(i, j)[2] = image[i][j].r;
            }
        }
        imshow("Current Image", currentMat);
        
        seamsRemoved++;
        cout << "\rProgress: " << seamsRemoved << "/" << totalSeams 
             << " seams removed | Current size: " << width << "x" << height << flush;
        
        // Control visualization speed (adjust delay as needed)
        int key = waitKey(10); // 10ms delay between seams
        if (key == 27) { // ESC key to stop
            cout << "\n\nStopped by user!" << '\n';
            break;
        }
        
        delete[] seam;
        for (int i = 0; i < height; ++i) {
            delete[] energy[i];
        }
        delete[] energy;
    }
    
    cout << "\n\nFinal size: " << width << "x" << height << '\n';
    
    // Save and display final result
    Mat3b resizedMat(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            resizedMat.at<Vec3b>(i, j)[0] = image[i][j].b;
            resizedMat.at<Vec3b>(i, j)[1] = image[i][j].g;
            resizedMat.at<Vec3b>(i, j)[2] = image[i][j].r;
        }
    }

    destroyWindow("Energy Map");
    destroyWindow("Seam Highlight");
    
    imshow("Final Resized Image", resizedMat);
    cout << "Press any key to save and exit..." << '\n';
    waitKey(0);
    
    imwrite("resized_image.jpg", resizedMat);
    cout << "Image saved as 'resized_image.jpg'" << '\n';
}

int main(int argc, char* argv[]) {
    string imagePath;
    int newWidth, newHeight;

    cout << "=== Seam Carving with Real-time Visualization ===" << '\n' << '\n';
    cout << "Enter the path to the image: ";
    cin >> imagePath;
    
    Pixel** image;
    int height, width;

    readImage(imagePath, image, height, width);
    
    if (image == nullptr) {
        return -1;
    }

    cout << "Original image size: " << width << "x" << height << '\n';
    cout << "Enter new width (must be <= " << width << "): ";
    cin >> newWidth;
    cout << "Enter new height (must be <= " << height << "): ";
    cin >> newHeight;
    
    if (newWidth > width || newHeight > height) {
        cout << "Error: New dimensions must be smaller than or equal to original!" << '\n';
        return -1;
    }
    
    resizeImage(image, height, width, newHeight, newWidth);

    // Cleanup
    for (int i = 0; i < height; ++i) {
        delete[] image[i];
    }
    delete[] image;

    cout << "DONE!!" << '\n';
    return 0;
}
