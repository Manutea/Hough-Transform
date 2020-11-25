#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include <stdio.h>
#include "Bitmap_image.h"

#define DEG2RAD (3.1415926535f/180.0f)

/** @struct line
* @brief list possible lines.
* @details Used to store data to retrace a line.
*/
struct line
{
    int x1, y1, x2, y2;
    bool isAline = false;
};

const thrust::host_vector<rgb_t> getPixels(const bitmap_image&, const int, const int);
void drawResult(const int, const int, const std::string&, const int, const line*, const rgb_t*);
__global__ void cudaAccumulator(const int, const int, const int, const double, const double, const double, const rgb_t*, int*);
__global__ void cudaGetLines(const int, const int, const int, const int, const int, const int*, line*);

/**
* @brief main.
*
* @param[in] argv[1], the input image path.
* @param[in] argv[2], the output result image path.
*/
int main(int argc, char* argv[])
{
    const std::string inputImagePath = argv[1];
    const std::string outputImagePath = argv[2];

    bitmap_image image(inputImagePath);
    if (!image)
    {
        printf("Error - Failed to open: input.bmp\n");
        return 1;
    }

    const auto theshold = 250;
    const auto height = image.height();
    const auto width  = image.width();

    const thrust::host_vector<rgb_t> h_pixels = getPixels(image, width, height);

    const auto houghH   = ((sqrt(2.0) * (height > width ? height : width)) / 2.0);
    const auto centerX  = width / 2.0;
    const auto centerY  = height / 2.0;
    const auto accuH = (int)(houghH * 2.0);
    const auto accuW = 180;
    const auto accuSize = accuH * accuW;

    const auto threadX = 8;
    const auto threadY = 8;

    const thrust::device_vector<rgb_t> d_pixels = h_pixels;
    const rgb_t* pixelsBufferArray = thrust::raw_pointer_cast(&d_pixels[0]);

    thrust::device_vector<int> d_accu(accuSize, 0);
    auto* accuBufferArray = thrust::raw_pointer_cast(&d_accu[0]);

    const dim3 accBlocks(width / threadX + 1, height / threadY + 1);
    const dim3 accThreads(threadX, threadY);

    cudaAccumulator << <accBlocks, accThreads >> > (theshold, width, height, centerX, centerY, houghH, pixelsBufferArray, accuBufferArray);
    cudaDeviceSynchronize();

    const dim3 linesBlocks(accuW / threadX + 1, accuH / threadY + 1);
    const dim3 linesThreads(threadX, threadY);

    thrust::device_vector<line> d_lines(accuSize);
    line* linesBufferArray = thrust::raw_pointer_cast(&d_lines[0]);
    cudaGetLines << <linesBlocks, linesThreads >> > (theshold, accuW, accuH, width, height, accuBufferArray, linesBufferArray);
    cudaDeviceSynchronize();

    const thrust::host_vector<line> h_lines = d_lines;

    drawResult(width, height, outputImagePath, accuSize, h_lines.data(), h_pixels.data());

    return 0;
}

/**
* @brief Get the pixels value from the image.
* 
* @param[in] image is a reference of te bitmap_image variable.
* @param[in] width, the width of the image to analyze.
* @param[in] height, the height of the image to analyze.
* 
* @returns a host vector contanining the rgb values.
*/
const thrust::host_vector<rgb_t> getPixels(const bitmap_image& _image, const int _width, const int _height) {
    const auto size = _height * _width;
    thrust::host_vector<rgb_t> pixels(size);

    for (auto y = 0; y < _height; ++y) {
        for (auto x = 0; x < _width; ++x) {
            const auto index = _width * y + x;
            _image.get_pixel(x, y, pixels[index]);
        }
    }

    return pixels;
}

/**
* @brief draw Hough lines in a image file.
*
* @param[in] width, the width of the image to analyze.
* @param[in] height, the height of the image to analyze.
* @param[in] str, output direction to write the image.
* @param[in] linesSize, the array size of the lines.
* @param[in] lines, the lines to draw.
* @param[in] pixels, array contanining the rgb values from the image to analyse.
*/
void drawResult(const int _width, const int _height, const std::string& _str, const int _linesSize, const line* _lines, const rgb_t* _pixels) {
    bitmap_image imageLines(_width, _height);
    image_drawer draw(imageLines);

    for (auto y = 0; y < _height; ++y) {
        for (auto x = 0; x < _width; ++x) {
            const auto index = _width * y + x;
            const rgb_t pixel = _pixels[index];
            imageLines.set_pixel(x, y, pixel);
        }
    }

    draw.pen_color(255, 0, 0);

    for (auto i = 0; i < _linesSize; ++i) {
        const auto line = _lines[i];
        if (line.isAline)
            draw.line_segment(line.x1, line.y1, line.x2, line.y2);
    }

    imageLines.save_image(_str);
}

/**
* @brief compute the Hough accumulator in CUDA
* 
* @param[in] threshold, is the value to start counting a white pixel.
* @param[in] width, the width of the image to analyze.
* @param[in] height, the height of the image to analyze.
* @param[in] centerX, is width/2.
* @param[in] centerY, is height/2.
* @param[in] houghH, the maximum height depends on the image size.
* @param[in] pixels, array contanining the rgb values from the image to analyse.
* @param[out] accu, array contanining the accumulator score.
*/
__global__ void cudaAccumulator(const int _threshold, const int _width, const int _height, const double _centerX, const double _centerY, const double _houghH, const rgb_t* _pixels, int* _accu) {
    const auto row = threadIdx.y + blockIdx.y * blockDim.y;
    const auto col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= _width || row >= _height) return;

    const auto index = row * _width + col;
    if ((_pixels[index].red + _pixels[index].green + _pixels[index].blue)/3 > _threshold) {
        for (auto t = 0; t < 180; ++t) {
            const auto r = (((double)col - _centerX) * cos((double)t * DEG2RAD)) + (((double)row - _centerY) * sin((double)t * DEG2RAD));
            const auto accuIndex = (int)((round(r + _houghH) * 180.0)) + t;
            atomicAdd(&_accu[accuIndex], 1);
        }
    }
}

/**
* @brief compute the Hough accumulator in CUDA
*
* @param[in] threshold, is the value to start counting a white pixel.
* @param[in] accuW, the width of the accumulator array.
* @param[in] accuH, the height of the accumulator array.
* @param[in] width, the width of the image to analyze.
* @param[in] height, the height of the image to analyze.
* @param[in] accu, array contanining the accumulator score.
*
* @param[out] lines, vector contanining the accumulator score.
*/
__global__ void cudaGetLines(const int _threshold, const int _accuW, const int _accuH, const int _width, const int _height, const int* _accu, line* _lines) {
    const int colT = threadIdx.x + blockIdx.x * blockDim.x;
    const int rowR = threadIdx.y + blockIdx.y * blockDim.y;
    if (colT >= _accuW || rowR >= _accuH) return;

    const int index = rowR * _accuW + colT;

    if (_accu[index] >= _threshold) {
        int max = _accu[index];
        for (int ly = -4; ly <= 4; ++ly)
            for (int lx = -4; lx <= 4; ++lx)
                if ((ly + rowR >= 0 && ly + rowR < _accuH) && (lx + colT >= 0 && lx + colT < _accuW))
                    if ((int)_accu[((rowR + ly) * _accuW) + (colT + lx)] > max)
                    {
                        max = _accu[((rowR + ly) * _accuW) + (colT + lx)];
                        ly = lx = 5;
                    }
        if (max > _accu[index] == false)
        {
            int x1, y1, x2, y2;

            if (colT >= 45 && colT <= 135)
            {
                //y = (r - x cos(t)) / sin(t)  
                x1 = 0;
                y1 = ((double)(rowR - (_accuH / 2)) - ((x1 - (_width / 2)) * cos(colT * DEG2RAD))) / sin(colT * DEG2RAD) + (_height / 2);
                x2 = _width - 0;
                y2 = ((double)(rowR - (_accuH / 2)) - ((x2 - (_width / 2)) * cos(colT * DEG2RAD))) / sin(colT * DEG2RAD) + (_height / 2);
            }
            else
            {
                //x = (r - y sin(t)) / cos(t);  
                y1 = 0;
                x1 = ((double)(rowR - (_accuH / 2)) - ((y1 - (_height / 2)) * sin(colT * DEG2RAD))) / cos(colT * DEG2RAD) + (_width / 2);
                y2 = _height - 0;
                x2 = ((double)(rowR - (_accuH / 2)) - ((y2 - (_height / 2)) * sin(colT * DEG2RAD))) / cos(colT * DEG2RAD) + (_width / 2);
            }

            _lines[index] = line{ x1, y1, x2, y2, true };
        }
    }
}