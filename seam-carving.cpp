#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct Pixel {
    uchar r, g, b;
};


void readImage(const string& imagePath, Pixel**& image, int& height, int& width) {
    Mat3b mat = imread(imagePath);

    if (mat.empty()) {
        cout<< "Error: Could not open or find the image!" << '\n';
        return ;
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
            int rGrad = 0, gGrad = 0, bGrad = 0;
            
            if (i > 0) {
                rGrad += abs(image[i][j].r - image[i - 1][j].r);
                gGrad += abs(image[i][j].g - image[i - 1][j].g);
                bGrad += abs(image[i][j].b - image[i - 1][j].b);
            }
            if (j > 0) {
                rGrad += abs(image[i][j].r - image[i][j - 1].r);
                gGrad += abs(image[i][j].g - image[i][j - 1].g);
                bGrad += abs(image[i][j].b - image[i][j - 1].b);
            }

            energy[i][j] = rGrad + gGrad + bGrad;
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
    } else {  
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
    }


    if (isVertical) {
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
    } else { 
        int minIndex = 0;
        for (int i = 1; i < height; ++i) {
            if (cumulativeEnergy[i][width - 1] < cumulativeEnergy[minIndex][width - 1]) {
                minIndex = i;
            }
        }
        seam[width - 1] = minIndex;
        for (int j = width - 2; j >= 0; --j) {
            int previous = seam[j + 1];
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

    // Clean
    for (int i = 0; i < height; ++i) {
        delete[] cumulativeEnergy[i];
    }
    delete[] cumulativeEnergy;
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
    while (height > newHeight || width > newWidth) {
        int* seam = (height > newHeight)? new int[width] : new int[height];
        int** energy = new int*[height];

        
        for (int i = 0; i < height; ++i) {
            energy[i] = new int[width];
        }

        calculateEnergy(image, height, width, energy);

        bool isVertical = (width > newWidth);
        findSeam(energy, height, width, seam, isVertical);
        
        removeSeam(image, height, width, seam, isVertical);

        delete[] seam;
        for (int i = 0; i < height; ++i) {
            delete[] energy[i];
        }
        delete[] energy;
    }


    Mat3b resizedMat(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            resizedMat.at<Vec3b>(i, j)[0] = image[i][j].b;
            resizedMat.at<Vec3b>(i, j)[1] = image[i][j].g;
            resizedMat.at<Vec3b>(i, j)[2] = image[i][j].r;
        }
    }

    imshow("Resized Image", resizedMat);
    waitKey(0);
    imwrite("resized_image.jpg", resizedMat);
    
}


int main(int argc, char* argv[]) {
    string imagePath;
    int newWidth, newHeight;

    cout << "Enter the path to the image: ";
    cin >> imagePath;
    cout << "Enter new width: ";
    cin >> newWidth;
    cout << "Enter new height: ";
    cin >> newHeight;

    Pixel** image;
    int height, width;

    readImage(imagePath, image, height, width);

    resizeImage(image, height, width, newHeight, newWidth);

    cout<<"DONE!!"<<'\n';
    return 0;
}

