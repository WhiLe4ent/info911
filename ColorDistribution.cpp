#include "ColorDistribution.hpp"
#include <cfloat>
#include <algorithm>

void ColorDistribution::reset()
{
    nb = 0;
    for (int r = 0; r < 8; r++)
        for (int g = 0; g < 8; g++)
            for (int b = 0; b < 8; b++)
                data[r][g][b] = 0.f;
}

void ColorDistribution::add(Vec3b color)
{
    int r = color[2] / 32;
    int g = color[1] / 32;
    int b = color[0] / 32;

    r = std::min(7, std::max(0, r));
    g = std::min(7, std::max(0, g));
    b = std::min(7, std::max(0, b));
    data[r][g][b] += 1.f;
    nb++;
}

void ColorDistribution::finished()
{
    if (nb == 0)
        return;
    for (int r = 0; r < 8; r++)
        for (int g = 0; g < 8; g++)
            for (int b = 0; b < 8; b++)
                data[r][g][b] /= static_cast<float>(nb);
}

float ColorDistribution::distance(const ColorDistribution &other) const
{
    float d = 0.f;
    for (int r = 0; r < 8; r++)
        for (int g = 0; g < 8; g++)
            for (int b = 0; b < 8; b++)
            {
                float a = data[r][g][b];
                float b2 = other.data[r][g][b];
                float denom = a + b2;
                if (denom > 0.f)
                    d += (a - b2) * (a - b2) / denom;
            }
    return d * 0.5f;
}

ColorDistribution getColorDistribution(const Mat &input, Point pt1, Point pt2)
{
    ColorDistribution cd;

    int x1 = std::max(0, std::min(pt1.x, input.cols - 1));
    int y1 = std::max(0, std::min(pt1.y, input.rows - 1));
    int x2 = std::max(0, std::min(pt2.x, input.cols));
    int y2 = std::max(0, std::min(pt2.y, input.rows));
    if (x2 <= x1 || y2 <= y1)
        return cd; 

    for (int y = y1; y < y2; y++)
        for (int x = x1; x < x2; x++)
            cd.add(input.at<Vec3b>(y, x));
    cd.finished();
    return cd;
}

float minDistance(const ColorDistribution &h, const std::vector<ColorDistribution> &hists)
{
    if (hists.empty())
        return FLT_MAX;
    float dmin = FLT_MAX;
    for (const auto &hh : hists)
    {
        float d = h.distance(hh);
        if (d < dmin)
            dmin = d;
    }
    return dmin;
}

void addDistributionIfFar(std::vector<ColorDistribution> &hists,
                          const ColorDistribution &newHist,
                          float threshold)
{
    if (newHist.nb == 0)
        return;
    if (hists.empty())
    {
        hists.push_back(newHist);
        return;
    }
    float d = minDistance(newHist, hists);
    if (d > threshold)
    {
        hists.push_back(newHist);
    }
}

cv::Mat recoObject(const cv::Mat &input,
                   const std::vector<ColorDistribution> &col_hists,
                   const std::vector<ColorDistribution> &col_hists_object,
                   const std::vector<cv::Vec3b> &colors,
                   const int bloc)
{
    Mat output = Mat::zeros(input.size(), CV_8UC3);

    for (int y = 0; y < input.rows; y += bloc)
    {
        for (int x = 0; x < input.cols; x += bloc)
        {
            Point p1(x, y);
            Point p2(std::min(x + bloc, input.cols), std::min(y + bloc, input.rows));
            ColorDistribution h = getColorDistribution(input, p1, p2);

            float d_fond = minDistance(h, col_hists);
            float d_obj = minDistance(h, col_hists_object);

            int label = (d_obj < d_fond) ? 1 : 0;
            Vec3b col = (label >= 0 && label < (int)colors.size()) ? colors[label] : Vec3b(0, 0, 0);
            rectangle(output, p1, p2, Scalar(col), FILLED);
        }
    }

    return output;
}

int closestObjectIndex(const ColorDistribution &h,
                       const std::vector<std::vector<ColorDistribution>> &all_hists)
{
    int best_index = -1;
    float best_dist = FLT_MAX;
    for (size_t i = 0; i < all_hists.size(); ++i)
    {
        if (all_hists[i].empty())
            continue;
        float d = minDistance(h, all_hists[i]);
        if (d < best_dist)
        {
            best_dist = d;
            best_index = static_cast<int>(i);
        }
    }
    if (best_index < 0)
    {
        best_index = 0;
    }
    return best_index;
}

cv::Mat recoObjectMulti(const cv::Mat &input,
                        const std::vector<std::vector<ColorDistribution>> &all_col_hists,
                        const std::vector<cv::Vec3b> &colors,
                        int bloc)
{
    Mat reco(input.size(), CV_8UC3, Scalar(0, 0, 0));
    for (int y = 0; y <= input.rows - bloc; y += bloc)
    {
        for (int x = 0; x <= input.cols - bloc; x += bloc)
        {
            ColorDistribution cd = getColorDistribution(input, Point(x, y), Point(x + bloc, y + bloc));
            int obj_idx = closestObjectIndex(cd, all_col_hists);
            Vec3b color = (obj_idx >= 0 && obj_idx < (int)colors.size()) ? colors[obj_idx] : Vec3b(0, 0, 0);
            rectangle(reco, Point(x, y), Point(x + bloc - 1, y + bloc - 1), Scalar(color), FILLED);
        }
    }
    return reco;
}
