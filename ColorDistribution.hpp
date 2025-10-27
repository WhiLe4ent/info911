#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

struct ColorDistribution
{
    float data[8][8][8]; // l'histogramme
    int nb;              // le nombre d'échantillons

    ColorDistribution() { reset(); }
    ColorDistribution &operator=(const ColorDistribution &other) = default;
    // Met à zéro l'histogramme
    void reset();
    // Ajoute l'échantillon color à l'histogramme:
    // met +1 dans la bonne case de l'histogramme et augmente le nb d'échantillons
    void add(Vec3b color);
    // Indique qu'on a fini de mettre les échantillons:
    // divise chaque valeur du tableau par le nombre d'échantillons
    // pour que case représente la proportion des picels qui ont cette couleur.
    void finished();
    // Retourne la distance entre cet histogramme et l'histogramme other
    float distance(const ColorDistribution &other) const;
};

ColorDistribution getColorDistribution(const Mat &input, Point pt1, Point pt2);

float minDistance(const ColorDistribution &h,
                  const std::vector<ColorDistribution> &hists);

cv::Mat recoObject(const cv::Mat &input,
                   const std::vector<ColorDistribution> &col_hists,
                   const std::vector<ColorDistribution> &col_hists_object,
                   const std::vector<cv::Vec3b> &colors,
                   const int bloc);

int closestObjectIndex(const ColorDistribution& h,
                       const std::vector<std::vector<ColorDistribution>>& all_hists);

cv::Mat recoObjectMulti(const cv::Mat &input,
                        const std::vector<std::vector<ColorDistribution>> &all_col_hists,
                        const std::vector<cv::Vec3b> &colors,
                        int bloc,
                        std::vector<std::vector<int>> &outLabels,
                        bool doRelax = true,
                        int superFactor = 4);

void addDistributionIfFar(std::vector<ColorDistribution> &hists,
                          const ColorDistribution &newHist,
                          float threshold);

void relaxLabels(std::vector<std::vector<int>>& labels, int nbRows, int nbCols, int passes = 1);

Mat computeMarkers(const std::vector<std::vector<int>> &labels, int bloc, int superFactor);

